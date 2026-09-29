"""可审计的 PPO 核心训练步（无现成 RL 训练器依赖）。

只依赖 numpy 与 torch。处理向量化自动重置环境的轨迹边界：

- terminated（真终止）：下一状态不存在，TD 目标 bootstrap 值为 0。
- truncated（仅时间限制截断）：episode 其实未结束，用截断前最终观测
  （final_observation）的估值 V(s_T) bootstrap；但优势递推不得跨过
  重置边界进入自动重置产生的下一条记录（那条记录已属于新 episode）。
- 批次末尾仍未终止：用该步 final_observation 的估值 bootstrap。
- 有效步掩码 valid 为 False 的填充步不参与归一化与损失。
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


@dataclass
class Rollout:
    """形状均为 (T, N) 的一批 rollout 数据（T 时间步，N 个并行环境）。

    rewards[t,n]            : 在 transition t 上得到的奖励 r_t
    values[t,n]             : 行为期价值 V_old(s_t)（与旧策略同时采样）
    old_log_probs[t,n]      : 旧策略对已选动作的 log pi_old(a_t|s_t)
    actions[t,n]            : 已选离散动作（整数）
    valid[t,n]              : 该记录是否为有效步（False=填充步）
    terminated[t,n]         : 真终止标志（MDP 结束，无下一状态）
    truncated[t,n]          : 时间限制截断标志（episode 逻辑上可继续）
    final_values[t,n]       : 该 transition 之后最终观测的价值 V(s_{t+1})；
                              真终止时应为 0；时间截断/批次截断时为对
                              “截断前最终观测”的估值。
    observations[t,n,:]     : 状态特征（可选，供真实更新使用）
    """

    rewards: np.ndarray
    values: np.ndarray
    old_log_probs: np.ndarray
    actions: np.ndarray
    valid: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_values: np.ndarray
    observations: np.ndarray = None

    def __post_init__(self):
        self.rewards = np.asarray(self.rewards, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)
        self.old_log_probs = np.asarray(self.old_log_probs, dtype=np.float64)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.terminated = np.asarray(self.terminated, dtype=bool)
        self.truncated = np.asarray(self.truncated, dtype=bool)
        self.final_values = np.asarray(self.final_values, dtype=np.float64)
        if self.observations is not None:
            self.observations = np.asarray(self.observations, dtype=np.float64)
        self._check_shapes()

    def _check_shapes(self):
        T, N = self.rewards.shape
        for name in ("values", "old_log_probs", "actions", "valid",
                     "terminated", "truncated", "final_values"):
            arr = getattr(self, name)
            if arr.shape != (T, N):
                raise ValueError(f"{name} 形状 {arr.shape} != {(T, N)}")
        # 终止/截断只在有效步上发生；真终止与时间截断互斥。
        if np.any(self.terminated & ~self.valid):
            raise ValueError("terminated 不能落在填充步上")
        if np.any(self.truncated & ~self.valid):
            raise ValueError("truncated 不能落在填充步上")
        if np.any(self.terminated & self.truncated):
            raise ValueError("同一步不能同时 terminated 与 truncated")

    @property
    def T(self):
        return self.rewards.shape[0]

    @property
    def N(self):
        return self.rewards.shape[1]

    def torch_tensors(self, dtype=torch.float64, device="cpu"):
        d = {}
        for name in ("rewards", "values", "old_log_probs"):
            d[name] = torch.tensor(getattr(self, name), dtype=dtype, device=device)
        d["actions"] = torch.tensor(self.actions, dtype=torch.long, device=device)
        for name in ("valid", "terminated", "truncated"):
            d[name] = torch.tensor(getattr(self, name), dtype=torch.bool, device=device)
        d["final_values"] = torch.tensor(self.final_values, dtype=dtype, device=device)
        return d


def derive_masks(valid, terminated, truncated):
    """推导两类掩码（均为 (T,N) bool，numpy 或 torch 均可）。

    kill[t]      = 该 transition 之后本 episode 是否“死亡”，
                   即下一状态价值必须按 0 处理（真终止）。
    bootstrap[t] = TD 目标的下一价值是否必须取自外部 final_values。
                   时间截断步 / 批次末尾最后有效步 -> True；
                   真终止 -> False；普通内部步 -> False（直接用 values[t+1]）。
    connect[t]   = GAE 递推是否把 t+1 步的优势接进来。
                   只有“同一 episode 内、且 t+1 也是有效步”才连接。
                   时间截断虽然允许 bootstrap，但自动重置后 t+1 的记录
                   已属于新 episode，connect 必须为 False。
    """
    xp = np if isinstance(valid, np.ndarray) else torch
    T, N = valid.shape[0], valid.shape[1]
    kill = terminated & valid
    # 批次内“最后一个有效步”（时间维）：批次截断候选。
    if xp is np:
        seen = np.zeros(N, dtype=bool)
        last_valid = np.zeros_like(valid)
        for t in range(T - 1, -1, -1):
            take = valid[t] & ~seen
            last_valid[t] = take
            seen |= valid[t]
        # t+1 是否为同一 episode 的有效后续记录。
        next_valid = np.zeros_like(valid)
        next_valid[:-1] = valid[1:]
    else:
        seen = torch.zeros(N, dtype=torch.bool, device=valid.device)
        last_valid = torch.zeros_like(valid)
        for t in range(T - 1, -1, -1):
            take = valid[t] & ~seen
            last_valid[t] = take
            seen |= valid[t]
        next_valid = torch.zeros_like(valid)
        next_valid[:-1] = valid[1:]
    # TD bootstrap：TD 目标的下一状态价值不来自缓冲区的 values[t+1]，
    # 而必须由外部 final_values 提供——时间截断步，或批次末尾最后有效步。
    bootstrap = valid & ~terminated & (truncated | last_valid)
    # 仅当 t+1 记录是同一 episode 的有效步时连接 GAE 递推。
    # 下列任一情况都必须断开：
    #   - t 自身是终止步（kill，无下一状态）；
    #   - t 自身是时间截断步（下一条记录是自动重置后新 episode 的第一步）；
    #   - t+1 落在边界标志上（承接自动重置后的新 episode 首记录）；
    #   - t+1 是填充步。
    # 自动重置约定下，record[t+1] 是否属于“同一 episode”完全由
    # transition t 的标志决定：t 真终止或时间截断 -> 重置 -> 下一条记录
    # 必属新 episode（或不存在），因此不连接；t+1 是填充步也不连接。
    cur_open = valid & ~terminated & ~truncated
    connect = cur_open & next_valid
    return kill, bootstrap, connect


def gae_reference(rollout, gamma=0.99, lam=0.95):
    """逐步标量参考实现（纯 numpy，逐环境逐时间步循环）。

    返回 Adv 与 Returns 两个 (T,N) 数组（填充步为 0）。

    对每个有效步 t：
        V_next = 0                          若 kill[t]（真终止）
                 final_values[t]            若 bootstrap[t]（时间/批次截断）
                 values[t+1]                否则（同一 episode 的内部步）
        delta  = r_t + gamma * V_next - V_t
        A_t    = delta_t + (gamma*lam * A_{t+1}) 仅当 connect[t]，否则只留 delta_t
        ret_t  = A_t + V_t                  （未归一化优势 + 旧 value）
    """
    T, N = rollout.T, rollout.N
    adv = np.zeros((T, N), dtype=np.float64)
    ret = np.zeros((T, N), dtype=np.float64)
    kill, bootstrap, connect = derive_masks(
        rollout.valid, rollout.terminated, rollout.truncated)
    for n in range(N):
        next_adv = 0.0
        for t in range(T - 1, -1, -1):
            if not rollout.valid[t, n]:
                adv[t, n] = 0.0
                ret[t, n] = 0.0
                next_adv = 0.0
                continue
            if kill[t, n]:
                v_next = 0.0                                  # 真终止：无下一状态
            elif bootstrap[t, n]:
                v_next = float(rollout.final_values[t, n])   # 截断/批次末尾：最终观测估值
            else:
                v_next = float(rollout.values[t + 1, n])     # 内部步：缓冲区已存的 V(s_{t+1})
            delta = (float(rollout.rewards[t, n])
                     + gamma * v_next
                     - float(rollout.values[t, n]))
            if connect[t, n]:
                a_t = delta + gamma * lam * next_adv
            else:
                a_t = delta
            adv[t, n] = a_t
            ret[t, n] = a_t + float(rollout.values[t, n])
            next_adv = a_t
    return adv, ret


def gae_torch(rollout, gamma=0.99, lam=0.95, device="cpu"):
    """与 gae_reference 数学等价的 torch 实现（时间维反向循环，便于审计）。"""
    d = rollout.torch_tensors(device=device)
    T, N = rollout.T, rollout.N
    kill, bootstrap, connect = derive_masks(d["valid"], d["terminated"], d["truncated"])
    zeros = torch.zeros_like(d["values"])
    buffered_next = torch.zeros_like(d["values"])
    buffered_next[:-1] = d["values"][1:]
    v_next = torch.where(kill, zeros,
                         torch.where(bootstrap, d["final_values"], buffered_next))
    delta = d["rewards"] + gamma * v_next - d["values"]
    adv = torch.zeros((T, N), dtype=d["values"].dtype, device=device)
    next_adv = torch.zeros(N, dtype=d["values"].dtype, device=device)
    for t in range(T - 1, -1, -1):
        cur = torch.where(connect[t], delta[t] + gamma * lam * next_adv, delta[t])
        cur = torch.where(d["valid"][t], cur, torch.zeros_like(cur))
        adv[t] = cur
        # 填充步之后的递推状态清零，防止跨“空洞”泄漏。
        next_adv = torch.where(d["valid"][t], cur, torch.zeros_like(cur))
    ret = adv + d["values"]
    ret = torch.where(d["valid"], ret, torch.zeros_like(ret))
    return adv, ret, {"kill": kill, "bootstrap": bootstrap,
                      "connect": connect, "delta": delta, "v_next": v_next}


def normalize_advantages(adv, valid):
    """优势归一化：只在有效步上做总体均值/方差（population std）。

    - 没有有效步：返回全 0。
    - 只有一个有效步：均值为该优势、方差为 0，归一化结果为 0
      （无法估计尺度；用 0 而不是除零）。
    归一化结果仅用于策略项；价值回归目标始终使用未归一化优势。
    """
    xp = np if isinstance(adv, np.ndarray) else torch
    if xp is np:
        if not np.any(valid):
            return np.zeros_like(adv)
        vals = adv[valid]
        mean = vals.mean()
        std = vals.std()
        if std < EPS:
            out = np.zeros_like(adv)
            out[valid] = 0.0
            return out
        return np.where(valid, (adv - mean) / (std + EPS), 0.0)
    if not bool(valid.any()):
        return torch.zeros_like(adv)
    vals = adv[valid]
    mean = vals.mean()
    std = vals.std(unbiased=False)
    if bool(std < EPS):
        return torch.zeros_like(adv)
    return torch.where(valid, (adv - mean) / (std + EPS), torch.zeros_like(adv))


def policy_value_loss(model, obs, actions, old_log_probs, adv_norm,
                      returns, values_old, valid, clip_range=0.2,
                      value_coef=0.5, entropy_coef=0.01):
    """离散动作的 clipped PPO 损失（仅对 valid 步求均值）。

    - 旧策略概率 old_log_probs、优势 adv_norm、回归目标 returns 在更新中固定
      （调用方应先 detach 并传入计算好的常量）。
    - 价值回归对 returns = V_old + A(未归一化) 做 MSE。
    - 策略项使用归一化优势 adv_norm。
    - 没有有效步时返回各分量为 0（保留 0 维张量与梯度图，便于安全 step）。
    """
    logits, v_pred = model(obs)            # (T,N,A), (T,N)
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    log_ratio = chosen - old_log_probs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_norm
    policy_loss = -torch.min(surr1, surr2)
    value_loss = (v_pred - returns) ** 2
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)

    mask = valid.to(v_pred.dtype)
    count = mask.sum().clamp_min(1.0)
    p_loss = (policy_loss * mask).sum() / count
    v_loss = (value_loss * mask).sum() / count
    ent = (entropy * mask).sum() / count
    total = p_loss + value_coef * v_loss - entropy_coef * ent
    return {
        "total": total,
        "policy": p_loss,
        "value": v_loss,
        "entropy": ent,
        "ratio": ratio,
        "approx_kl": (mask * (log_ratio.detach())).sum() / count,
        "clip_fraction": (mask * ((ratio - 1.0).abs() > clip_range).to(mask.dtype)).sum() / count,
        "v_pred": v_pred,
    }


class ActorCritic(nn.Module):
    """小型离散动作 actor-critic 网络（默认 float64，便于数值核验）。"""

    def __init__(self, obs_dim, n_actions, hidden=16, dtype=torch.float64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden, dtype=dtype),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions, dtype=dtype)
        self.value_head = nn.Linear(hidden, 1, dtype=dtype)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def ppo_update(rollout, gamma=0.99, lam=0.95, clip_range=0.2,
               value_coef=0.5, entropy_coef=0.01, lr=1e-2,
               n_epochs=1, minibatch_size=None, seed=0, verbose=False):
    """执行一次真实的 PPO 优化更新。

    采样期量（old_log_probs、A、returns）在所有 epoch/minibatch 上固定。
    返回更新前后损失、参数范数变化与审计信息。
    """
    torch.manual_seed(seed)
    if rollout.observations is None:
        raise ValueError("真实更新需要 observations")
    T, N = rollout.T, rollout.N
    obs_dim = rollout.observations.shape[-1]
    n_actions = int(rollout.actions.max()) + 1

    adv_t, ret_t, mask_info = gae_torch(rollout, gamma=gamma, lam=lam)
    adv_np, ret_np = gae_reference(rollout, gamma=gamma, lam=lam)
    assert np.allclose(adv_t.numpy(), adv_np, atol=1e-10)
    assert np.allclose(ret_t.numpy(), ret_np, atol=1e-10)

    valid_t = torch.tensor(rollout.valid)
    adv_norm_t = normalize_advantages(adv_t, valid_t)
    # 全部目标 detach：更新中固定旧策略量。
    fixed = {
        "obs": torch.tensor(rollout.observations),
        "actions": torch.tensor(rollout.actions, dtype=torch.long),
        "old_log_probs": torch.tensor(rollout.old_log_probs),
        "adv_norm": adv_norm_t.detach(),
        "returns": ret_t.detach(),
        "values_old": torch.tensor(rollout.values),
        "valid": valid_t,
    }
    model = ActorCritic(obs_dim, n_actions)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def evaluate():
        with torch.no_grad():
            out = policy_value_loss(
                model, fixed["obs"], fixed["actions"], fixed["old_log_probs"],
                fixed["adv_norm"], fixed["returns"], fixed["values_old"],
                fixed["valid"], clip_range, value_coef, entropy_coef)
        return {k: float(out[k]) for k in ("total", "policy", "value", "entropy",
                                          "approx_kl", "clip_fraction")}

    before = evaluate()
    params_before = [p.detach().clone() for p in model.parameters()]

    flat_idx = [idx for idx in range(T * N) if rollout.valid.reshape(-1)[idx]]
    rng = np.random.default_rng(seed)
    steps = 0
    for _ in range(n_epochs):
        order = rng.permutation(flat_idx).tolist()
        bs = minibatch_size or len(order)
        for start in range(0, len(order), bs):
            mb = order[start:start + bs]
            if not mb:
                continue
            idx = torch.tensor(mb, dtype=torch.long)
            def pick(x):
                flat = x.reshape(T * N, *x.shape[2:])
                return flat[idx]
            out = policy_value_loss(
                model, pick(fixed["obs"]), pick(fixed["actions"]),
                pick(fixed["old_log_probs"]), pick(fixed["adv_norm"]),
                pick(fixed["returns"]), pick(fixed["values_old"]),
                pick(fixed["valid"]), clip_range, value_coef, entropy_coef)
            opt.zero_grad()
            out["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            opt.step()
            steps += 1

    after = evaluate()
    param_move = float(sum(
        (p - pb).norm().item() ** 2 for p, pb in zip(model.parameters(), params_before)) ** 0.5)
    if verbose:
        print(f"[ppo_update] steps={steps} total {before['total']:.6f} -> "
              f"{after['total']:.6f} (Δ{after['total'] - before['total']:+.6f}), "
              f"param_move={param_move:.6f}")
    return {
        "model": model,
        "before": before,
        "after": after,
        "param_move": param_move,
        "steps": steps,
        "advantages": adv_np,
        "returns": ret_np,
        "masks": mask_info,
    }
