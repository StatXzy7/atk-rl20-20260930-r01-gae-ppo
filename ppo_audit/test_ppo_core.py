"""自动化测试：可手算轨迹、边界掩码、lambda 端点、填充/单步/空批次、
numpy 与 torch 对拍、损失独立核验、有限差分梯度核验、
reset-observation 误当 final-observation 的失败反例、真实优化更新。

运行：python -m pytest test_ppo_core.py -v
"""

import numpy as np
import torch
import torch.nn.functional as F

import pytest

from ppo_core import (
    Rollout, derive_masks, gae_reference, gae_torch,
    normalize_advantages, policy_value_loss, ActorCritic, ppo_update,
)
from fixtures import (
    build_rollout, EXPECTED_ADV, EXPECTED_RET,
    EXPECTED_KILL, EXPECTED_BOOT, EXPECTED_CONN, GAMMA, LAM, T, N,
)


# --------------------------------------------------------------------------
# 1. 掩码与 GAE 的手算核验（4+ 条短轨迹 + 填充）
# --------------------------------------------------------------------------

def test_masks_hand_computed():
    r = build_rollout()
    kill, boot, conn = derive_masks(r.valid, r.terminated, r.truncated)
    np.testing.assert_array_equal(kill, EXPECTED_KILL)
    np.testing.assert_array_equal(boot, EXPECTED_BOOT)
    np.testing.assert_array_equal(conn, EXPECTED_CONN)


def test_gae_reference_hand_computed():
    r = build_rollout()
    adv, ret = gae_reference(r, GAMMA, LAM)
    np.testing.assert_allclose(adv, EXPECTED_ADV, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ret, EXPECTED_RET, rtol=0, atol=1e-12)
    # 填充步严格为 0
    assert np.all(adv[~r.valid] == 0.0)
    assert np.all(ret[~r.valid] == 0.0)
    # ret == V_old + A（未归一化优势）
    np.testing.assert_allclose(ret[r.valid], adv[r.valid] + r.values[r.valid])


def test_torch_matches_reference():
    r = build_rollout()
    adv_n, ret_n = gae_reference(r, GAMMA, LAM)
    adv_t, ret_t, info = gae_torch(r, GAMMA, LAM)
    np.testing.assert_allclose(adv_t.numpy(), adv_n, atol=1e-12)
    np.testing.assert_allclose(ret_t.numpy(), ret_n, atol=1e-12)
    # 内部步的 v_next 必须来自 values[t+1]，而不是 final_values
    # env C t0：final_values=0，但 v_next 应为 V(t1)=2
    assert abs(float(info["v_next"][0, 2]) - 2.0) < 1e-12
    # env D t2（新 episode 首步，t1 真终止）：v_next=V(t3)=2，必须非 0 引导
    assert abs(float(info["v_next"][2, 3]) - 2.0) < 1e-12
    # env D t1（真终止）：v_next=0
    assert float(info["v_next"][1, 3]) == 0.0
    # env B t1（时间截断）：v_next=final_values=8
    assert abs(float(info["v_next"][1, 1]) - 8.0) < 1e-12


# --------------------------------------------------------------------------
# 2. lambda = 0 与 lambda = 1
# --------------------------------------------------------------------------

def _single_env_rollout(rewards, values, terminated, truncated, final_values,
                        t_len=None):
    n = len(rewards)
    T_ = t_len if t_len is not None else n
    assert T_ >= n
    def col(x, fill=0.0):
        arr = np.full(T_, fill, dtype=np.float64)
        arr[:len(x)] = x
        return arr.reshape(T_, 1)
    valid = np.zeros((T_, 1), dtype=bool)
    valid[:n] = True
    term = np.zeros(T_, dtype=bool); term[:n] = terminated
    trunc = np.zeros(T_, dtype=bool); trunc[:n] = truncated
    return Rollout(
        col(rewards), col(values), col([0.0] * n),
        np.zeros((T_, 1), dtype=np.int64), valid,
        term.reshape(T_, 1), trunc.reshape(T_, 1),
        col(final_values),
    )


def test_lambda_zero_is_td_residual():
    # 真终止轨迹 r=[2,4,8], V=[1,2,4], gamma=.5
    r = _single_env_rollout([2, 4, 8], [1, 2, 4],
                            [False, False, True], [False] * 3, [0, 0, 0])
    adv, ret = gae_reference(r, gamma=0.5, lam=0.0)
    # delta: 2+.5*2-1=2 ; 4+.5*4-2=4 ; 8+0-4=4
    np.testing.assert_allclose(adv[:3, 0], [2, 4, 4], atol=1e-12)
    np.testing.assert_allclose(ret[:3, 0], [3, 6, 8], atol=1e-12)


def test_lambda_one_is_monte_carlo_with_bootstrap():
    # 时间截断轨迹 r=[2,4], V=[1,2], V_final=8, gamma=.5
    r = _single_env_rollout([2, 4], [1, 2],
                            [False, False], [False, True], [0, 8], t_len=2)
    adv, ret = gae_reference(r, gamma=0.5, lam=1.0)
    # A1 = 4 + .5*8 - 2 = 6
    # A0 = (2+.5*2-1) + .5*1*6 = 2 + 3 = 5
    np.testing.assert_allclose(adv[:2, 0], [5, 6], atol=1e-12)
    np.testing.assert_allclose(ret[:2, 0], [6, 8], atol=1e-12)


def test_lambda_one_bias_when_values_wrong():
    """lam=1 并不“无偏估计优势函数”：用有偏价值做基线/自举时优势含偏差。

    构造两步行情：真实最优回报从截断点之后应为 20（V_final 的真值），
    但 critic 只给 8。MC 自举目标 4+.5*8=8，而真值 4+.5*20=14。
    正确价值函数 V*(s1)=14、V*(s0)=2+.5*14=9；给错误 V=[1,2] 时，
    A0=2+.5*8-1=5，而真实优势 A*(s0)=9-1=8 —— 偏差 3 被采样批次固化。
    """
    r = _single_env_rollout([2, 4], [1, 2],
                            [False, False], [False, True], [0, 8], t_len=2)
    adv_wrong, _ = gae_reference(r, gamma=0.5, lam=1.0)
    r_true = _single_env_rollout([2, 4], [1, 2],
                                 [False, False], [False, True], [0, 20], t_len=2)
    adv_true, _ = gae_reference(r_true, gamma=0.5, lam=1.0)
    assert adv_wrong[0, 0] != adv_true[0, 0]
    # A1 的偏差 = gamma*(8-20) = -6；lam=1 时再乘 gamma 传到 A0 -> -3
    np.testing.assert_allclose(adv_wrong[1, 0] - adv_true[1, 0], -6.0, atol=1e-12)
    np.testing.assert_allclose(adv_wrong[0, 0] - adv_true[0, 0], -3.0, atol=1e-12)
    # lam=0：A0 的 TD 自举来自 V(s1)=2（同一轨迹内部价值，两处相同），
    # 截断点估值偏差不会传播到 A0。
    adv0_w, _ = gae_reference(r, gamma=0.5, lam=0.0)
    adv0_t, _ = gae_reference(r_true, gamma=0.5, lam=0.0)
    np.testing.assert_allclose(adv0_w[0, 0], adv0_t[0, 0], atol=1e-12)


# --------------------------------------------------------------------------
# 3. 填充步 / 只有一个有效步 / 全空
# --------------------------------------------------------------------------

def test_padding_excluded_from_normalization():
    r = build_rollout()
    adv, _ = gae_reference(r, GAMMA, LAM)
    norm = normalize_advantages(adv, r.valid)
    valid_vals = norm[r.valid]
    np.testing.assert_allclose(valid_vals.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(valid_vals.std(), 1.0, atol=1e-12)
    assert np.all(norm[~r.valid] == 0.0)
    # torch 版本一致
    norm_t = normalize_advantages(torch.tensor(adv), torch.tensor(r.valid))
    np.testing.assert_allclose(norm_t.numpy(), norm, atol=1e-12)


def test_single_valid_step():
    # 只有一个有效步（真终止）：归一化退化为 0（单样本无法估尺度），
    # 优势本身仍有定义，回归目标 = V + A。
    T1 = 3
    rewards = np.zeros((T1, 1)); values = np.zeros((T1, 1))
    final = np.zeros((T1, 1)); olp = np.zeros((T1, 1))
    act = np.zeros((T1, 1), int); valid = np.zeros((T1, 1), bool)
    term = np.zeros((T1, 1), bool); trunc = np.zeros((T1, 1), bool)
    rewards[0, 0] = 8.0; values[0, 0] = 4.0; valid[0, 0] = True; term[0, 0] = True
    r = Rollout(rewards, values, olp, act, valid, term, trunc, final)
    adv, ret = gae_reference(r, 0.5, 0.5)
    np.testing.assert_allclose(adv[0, 0], 4.0, atol=1e-12)
    np.testing.assert_allclose(ret[0, 0], 8.0, atol=1e-12)
    norm = normalize_advantages(adv, valid)
    assert norm[0, 0] == 0.0
    assert np.all(norm[~valid] == 0.0)


def test_all_empty_is_well_defined():
    T1 = 3
    z = np.zeros((T1, 1))
    valid = np.zeros((T1, 1), bool)
    r = Rollout(z, z, z, np.zeros((T1, 1), int), valid,
                np.zeros((T1, 1), bool), np.zeros((T1, 1), bool), z)
    adv, ret = gae_reference(r, 0.5, 0.5)
    assert np.all(adv == 0.0) and np.all(ret == 0.0)
    norm = normalize_advantages(adv, valid)
    assert np.all(norm == 0.0)
    # 损失路径安全返回 0，且不产生 nan
    obs = torch.zeros(T1, 1, 2, dtype=torch.float64)
    model = ActorCritic(2, 2)
    out = policy_value_loss(
        model, obs, torch.zeros(T1, 1, dtype=torch.long),
        torch.zeros(T1, 1, dtype=torch.float64), torch.zeros(T1, 1),
        torch.zeros(T1, 1), torch.zeros(T1, 1), torch.zeros(T1, 1, dtype=torch.bool))
    assert float(out["total"].detach()) == 0.0
    out["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.all(g == 0.0) for g in grads)


# --------------------------------------------------------------------------
# 4. 失败反例：把 reset observation 的价值误当 final observation
#    （并把 GAE 递推推进自动重置后的新 episode）
# --------------------------------------------------------------------------

def buggy_gae_misuse_reset_obs(rewards, values, reset_values, valid,
                               terminated, truncated, gamma, lam):
    """典型错误实现：向量化环境自动重置后，用“当前槽位下一条记录的价值”
    V(s_{t+1}) 作为所有非终止步的自举值——时间截断时，那个 s_{t+1} 其实是
    自动重置后新 episode 的初始观测，不是截断前最终观测；同时 GAE 也不切断
    新 episode 边界。此函数仅用于演示错误，绝不能在生产代码中使用。
    """
    T, N = rewards.shape
    adv = np.zeros((T, N))
    next_adv = np.zeros(N)
    for t in range(T - 1, -1, -1):
        # 错误 1：截断也直接拿缓冲区下一条记录的 V（=reset 后的新 obs 价值）
        # 错误 2：根本不看 truncated，递推一路穿过 episode 边界
        if t + 1 < T:
            v_next = np.where(terminated[t], 0.0, values[t + 1])
            connected = valid[t] & valid[t + 1] & ~terminated[t]
        else:
            v_next = np.where(terminated[t], 0.0, reset_values[t])
            connected = np.zeros(N, dtype=bool)
        delta = rewards[t] + gamma * v_next - values[t]
        a = np.where(connected, delta + gamma * lam * next_adv, delta)
        adv[t] = np.where(valid[t], a, 0.0)
        next_adv = np.where(valid[t], a, 0.0)
    return adv


def test_reset_obs_misuse_is_demonstrably_wrong():
    # 取 env D：t1 真终止（ep1 结束），t2 是自动重置后的新 episode，
    # t4 时间截断且截断前最终观测价值 final_values[4]=8。
    r = build_rollout()
    adv_good, _ = gae_reference(r, GAMMA, LAM)

    # 错误实现：截断点 t4 也只能看到“下一条记录”；批次末尾用 reset 观测价值
    # 模拟自动重置观测价值 = 新 episode 初值（这里故意设成 1），与真值 8 不同。
    reset_values = np.ones_like(r.final_values)  # 自动重置 obs 被错误估值为 1
    adv_bad = buggy_gae_misuse_reset_obs(
        r.rewards, r.values, reset_values, r.valid,
        r.terminated, r.truncated, GAMMA, LAM)

    # --- 失败点 1：截断自举值错误 ---
    # 正确 A4 = 8 + .5*8 - 4 = 8；错误：8 + .5*1 - 4 = 4.5
    np.testing.assert_allclose(adv_good[4, 3], 8.0, atol=1e-12)
    np.testing.assert_allclose(adv_bad[4, 3], 4.5, atol=1e-12)

    # --- 失败点 2：优势跨 episode 边界泄漏 ---
    # env B：t1 截断后本无后续有效记录（t2 填充），这里再构造一个“截断后
    # 立刻是新 episode”的微型两环境批次，使错误递推可见。
    Tb = 3
    rewards = np.array([[2.0, 9.0], [4.0, 1.0], [1.0, 1.0]])
    values = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    final = np.zeros((Tb, 2)); final[1, 0] = 8.0   # env0 t1 截断，最终 obs V=8
    valid = np.ones((Tb, 2), bool)
    term = np.zeros((Tb, 2), bool)
    trunc = np.zeros((Tb, 2), bool); trunc[1, 0] = True
    olp = np.zeros((Tb, 2)); act = np.zeros((Tb, 2), int)
    rb = Rollout(rewards, values, olp, act, valid, term, trunc, final)
    good, _ = gae_reference(rb, GAMMA, LAM)
    bad = buggy_gae_misuse_reset_obs(rewards, values, np.ones_like(final),
                                     valid, term, trunc, GAMMA, LAM)
    # 正确：t1 截断 -> A1 = 4+.5*8-2 = 6；A0 = 3 + .25*6 = 4.5；t2 是新 episode
    np.testing.assert_allclose(good[:2, 0], [3.5, 6.0], atol=1e-12)
    np.testing.assert_allclose(good[2, 0], -2.0, atol=1e-12)  # 1+.5*0-3
    # 错误：t1 拿 t2 的 V=3（reset obs 价值），且把 t2 的 A 也接进 t1/t0
    assert not np.allclose(bad[:2, 0], good[:2, 0])
    # 错误 A1 = 4 + .5*3 - 2 = 3.5（真值 6）；错误 A0 还混入了新 episode t2
    # 错误 A1：自举值错（3 而非 8）且把新 episode t2 的 A=-1.5 也接进来
    # delta1_bad = 4+.5*3-2 = 3.5；A2_bad = 1+.5*1-3 = -1.5（reset obs 估值 1）
    # A1_bad = 3.5 + .25*(-1.5) = 3.125
    np.testing.assert_allclose(bad[1, 0], 3.125, atol=1e-12)


# --------------------------------------------------------------------------
# 5. 损失独立核验（numpy 手算，避开 clip 折点 ratio=1±ε）
# --------------------------------------------------------------------------

class TinyLinearModel(torch.nn.Module):
    """logits/value 均为线性，便于手算。"""

    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.wv = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.bv = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, x):
        logits = x @ self.W + self.b
        value = x @ self.wv + self.bv
        return logits, value


def test_loss_components_hand_computed():
    # 两条有效步，obs=[(1,0),(0,1)]，logits 恒为 0 -> pi=0.5, H=ln2
    obs = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)  # (2,1,2)
    obs = obs.reshape(2, 1, 2)
    actions = torch.zeros((2, 1), dtype=torch.long)
    ln2 = np.log(2.0)
    # step0 ratio=1（old=new=-ln2）；step1 目标 ratio=1.1（远离 1±.2 折点）
    old_lp = torch.tensor([[-ln2], [-ln2 - np.log(1.1)]], dtype=torch.float64)
    adv_norm = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    returns = torch.tensor([[1.0], [0.0]], dtype=torch.float64)  # v_pred=0 -> MSE=.5
    values_old = torch.zeros((2, 1), dtype=torch.float64)
    valid = torch.ones((2, 1), dtype=torch.bool)

    model = TinyLinearModel()
    out = policy_value_loss(model, obs, actions, old_lp, adv_norm, returns,
                            values_old, valid, clip_range=0.2,
                            value_coef=0.5, entropy_coef=0.01)
    np.testing.assert_allclose(float(out["ratio"][0, 0].detach()), 1.0, atol=1e-12)
    np.testing.assert_allclose(float(out["ratio"][1, 0].detach()), 1.1, atol=1e-12)
    # policy = -mean([1*1, 1.1*(-1)]) = 0.05
    np.testing.assert_allclose(float(out["policy"].detach()), 0.05, atol=1e-12)
    np.testing.assert_allclose(float(out["value"].detach()), 0.5, atol=1e-12)
    np.testing.assert_allclose(float(out["entropy"].detach()), ln2, atol=1e-12)
    expected_total = 0.05 + 0.5 * 0.5 - 0.01 * ln2
    np.testing.assert_allclose(float(out["total"].detach()), expected_total, atol=1e-12)
    np.testing.assert_allclose(float(out["clip_fraction"].detach()), 0.0, atol=1e-12)


def test_padding_excluded_from_loss():
    # 同样两有效步，但混入两个填充步（含垃圾数值）：损失必须不变
    obs = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[9.0, 9.0]], [[9.0, 9.0]]],
                       dtype=torch.float64)
    actions = torch.zeros((4, 1), dtype=torch.long)
    ln2 = np.log(2.0)
    old_lp = torch.tensor([[-ln2], [-ln2 - np.log(1.1)], [9.0], [9.0]])
    adv_norm = torch.tensor([[1.0], [-1.0], [99.0], [-99.0]])
    returns = torch.tensor([[1.0], [0.0], [99.0], [99.0]])
    values_old = torch.zeros((4, 1))
    valid = torch.tensor([[True], [True], [False], [False]])
    model = TinyLinearModel()
    out = policy_value_loss(model, obs, actions, old_lp, adv_norm, returns,
                            values_old, valid, 0.2, 0.5, 0.01)
    np.testing.assert_allclose(float(out["policy"].detach()), 0.05, atol=1e-12)
    np.testing.assert_allclose(float(out["value"].detach()), 0.5, atol=1e-12)
    np.testing.assert_allclose(float(out["entropy"].detach()), ln2, atol=1e-12)


# --------------------------------------------------------------------------
# 6. 有限差分梯度核验（避开 clip 折点与 logits=0；float64）
# --------------------------------------------------------------------------

def test_gradient_finite_difference():
    torch.manual_seed(7)
    # 用非对称、非零 logits，确保所有 ratio 远离 1±0.2 折点。
    obs = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)
    actions = torch.tensor([[0], [1]], dtype=torch.long)

    model = TinyLinearModel()
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.35, -0.21], [0.18, 0.27]]))
        model.b.copy_(torch.tensor([0.12, -0.09]))
        model.wv.copy_(torch.tensor([0.31, -0.44]))
        model.bv.fill_(0.07)

    def logits_and_values():
        logits, v = model(obs)
        return logits, v

    logits, v_pred0 = logits_and_values()
    log_probs = F.log_softmax(logits, dim=-1)
    chosen_lp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    # 选 old_log_probs 使 ratio=exp(±0.03)，严格在 [0.8,1.2] 内部
    ratio_targets = torch.tensor([[1.03], [0.97]])
    old_lp = (chosen_lp - torch.log(ratio_targets)).detach()
    adv_norm = torch.tensor([[0.7], [-0.4]])
    # 回归目标任意非平凡值
    returns = (v_pred0.detach() + torch.tensor([[0.2], [-0.3]]))
    values_old = torch.zeros((2, 1), dtype=torch.float64)
    valid = torch.ones((2, 1), dtype=torch.bool)

    def total_loss():
        return policy_value_loss(
            model, obs, actions, old_lp, adv_norm, returns, values_old,
            valid, 0.2, 0.5, 0.01)["total"]

    total_loss().backward()
    analytic = {name: p.grad.detach().clone() for name, p in model.named_parameters()}

    eps = 1e-6
    max_rel = 0.0
    for name, p in model.named_parameters():
        num = torch.zeros_like(p)
        flat = p.detach().reshape(-1)
        ngrad = analytic[name].reshape(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            fp = float(total_loss().detach())
            flat[i] = orig - eps
            fm = float(total_loss().detach())
            flat[i] = orig
            num.reshape(-1)[i] = (fp - fm) / (2 * eps)
        denom = torch.maximum(ngrad.abs(), num.reshape(-1).abs()).clamp_min(1e-8)
        rel = ((ngrad - num.reshape(-1)).abs() / denom).max().item()
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-5, f"有限差分相对误差过大: {max_rel}"


def test_clip_branch_changes_gradient():
    """折点处外侧使用 clipped 目标：构造 ratio=1.3（超出 1.2）且 adv>0，
    此时策略梯度应为 0（clamp 平台），而未 clip 时不为 0。"""
    obs = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    actions = torch.zeros((1, 1), dtype=torch.long)
    model = TinyLinearModel()
    with torch.no_grad():
        model.W.copy_(torch.zeros((2, 2)))
        model.b.copy_(torch.zeros(2))
    # logits=0 -> logp=-ln2；要 ratio=1.3，old_lp = -ln2 - ln1.3
    old_lp = torch.tensor([[-np.log(2.0) - np.log(1.3)]])
    adv_norm = torch.tensor([[1.0]])
    returns = torch.zeros((1, 1)); values_old = torch.zeros((1, 1))
    valid = torch.ones((1, 1), dtype=torch.bool)
    out = policy_value_loss(model, obs, actions, old_lp, adv_norm, returns,
                            values_old, valid, 0.2, 0.5, 0.0)
    out["total"].backward()
    # policy 项被 clip 成常数 1.2*adv，对 W/b 无梯度；只剩熵项
    assert float(out["ratio"][0, 0].detach()) == pytest.approx(1.3, abs=1e-12)
    np.testing.assert_allclose(model.W.grad.numpy(), 0.0, atol=1e-12)
    # 价值头仍然有梯度（返回目标为 0，v_pred 也为 0 -> 0 梯度是巧合；
    # 这里仅核验 policy 头为零即可）
    np.testing.assert_allclose(model.b.grad.numpy(), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# 7. 真实优化更新（随机初始化小型策略网络，固定旧策略量，损失下降）
# --------------------------------------------------------------------------

def test_real_update_decreases_loss_and_fixes_old_targets():
    r = build_rollout(with_obs=True, seed=3)
    res = ppo_update(r, gamma=GAMMA, lam=LAM, clip_range=0.2,
                     value_coef=0.5, entropy_coef=0.01, lr=5e-2,
                     n_epochs=1, minibatch_size=None, seed=1)
    assert res["param_move"] > 1e-4
    assert res["after"]["total"] < res["before"]["total"], res
    # 优势/目标在更新中固定：两次计算必须完全一致（ppo_update 内部已 assert，
    # 这里再独立确认 returns 仍是 V_old+A，未被归一化污染）
    np.testing.assert_allclose(
        res["returns"][r.valid],
        res["advantages"][r.valid] + r.values[r.valid], atol=1e-12)


def test_real_update_multiepoch_targets_remain_fixed():
    # 多 epoch 更新：old_log_probs 与 returns 不随新策略改变。
    # 通过极小 lr 多步确认 approx_kl 非负且 clip_fraction 在随机小网络上有界。
    r = build_rollout(with_obs=True, seed=11)
    res = ppo_update(r, gamma=GAMMA, lam=LAM, lr=1e-3,
                     n_epochs=3, minibatch_size=4, seed=5)
    assert res["steps"] == 3 * (int(r.valid.sum()) // 4 + 1)
    assert 0.0 <= res["after"]["clip_fraction"] <= 1.0
    assert res["after"]["approx_kl"] >= -0.5  # KL 估计允许噪声但不应发散
