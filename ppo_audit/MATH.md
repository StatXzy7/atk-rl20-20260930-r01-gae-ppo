# 可审计 PPO 核心训练步：数学说明

记号：缓冲区形状 `(T, N)`，记录 `(s_t, a_t, r_t, s'_t)` 位于槽位
`(t, n)`。向量化环境开启自动重置（autoreset）时，**第 `t` 条 transition
之后槽位里出现的下一帧 `record[t+1]` 可能已经属于新 episode**。因此
“时间上相邻”不等于“轨迹上相邻”，这是本实现处理的核心问题。

## 1. 三种 episode 结束语义

| 情况 | 标志 | MDP 是否结束 | TD 下一状态价值 | GAE 是否递推到 record[t+1] |
|---|---|---|---|---|
| 真终止 | `terminated[t]=1` | 是 | 0（无下一状态） | 否（下一条是新 episode 或无记录） |
| 时间限制截断 | `truncated[t]=1` | 否（episode 逻辑可继续） | `V(final_obs_t)`（截断前最终观测） | **否**（下一条记录是自动重置后的新 episode） |
| 批次末尾未终止 | 最后有效步且两标志皆 0 | 否 | `V(final_obs_t)`（采样批次边界） | 否（缓冲区里没有下一条记录） |
| 普通内部步 | 全 0 | 否 | `V(s_{t+1}) = values[t+1]` | 是 |

关键区分：时间截断只切断“数据记录的连续性”，**不切断 MDP 的价值依赖**，
所以允许 bootstrap（`V(final_obs_t)` 是对截断后未来回报的无偏条件期望，
在价值函数准确时无偏），但**优势递推不得把自动重置后新 episode 的
δ_{t+1} 接进来**——那是另一条轨迹的量。

## 2. 两类掩码的推导

对有效步集合 `M = {t : valid[t]=1}`：

- **TD bootstrap 掩码**
  ```
  kill[t]      = valid[t] ∧ terminated[t]
  last_valid   = 每列（沿 t 反向）遇到的第一个有效步
  bootstrap[t] = valid[t] ∧ ¬terminated[t] ∧ (truncated[t] ∨ last_valid[t])
  ```
  `bootstrap[t]=1` 表示 δ_t 中的下一状态价值必须由外部
  `final_values[t]` 提供，而不能用缓冲区里的 `values[t+1]`。

- **下一状态价值**
  ```
  V_next(t) = 0                         若 kill[t]
            = final_values[t]           若 bootstrap[t]
            = values[t+1]               否则（同一 episode 内部步）
  δ_t = r_t + γ V_next(t) − V(s_t)
  ```

- **GAE 递推连接掩码**（与 bootstrap 掩码**不同且必须分别实现**）
  ```
  connect[t] = valid[t] ∧ valid[t+1]
               ∧ ¬terminated[t] ∧ ¬truncated[t]
  ```
  自动重置约定下，`record[t+1]` 是否属于同一 episode 完全由 transition t
  自身的标志决定：一旦 t 终止或截断，环境必然在该 transition 之后重置，
  `record[t+1]` 必属新 episode，必须断开。注意终止步 `t=T_k` 的 δ 仍以
  γλ 权重进入其**前一步**（`connect[T_k−1]=1`）；断开的只是
  `connect[T_k]`。

  反向递推（标量参考实现逐 `(t,n)` 执行；torch 实现沿时间维反向循环）：
  ```
  A_t = δ_t + γλ · A_{t+1}   若 connect[t]
      = δ_t                  否则
  R_t = V(s_t) + A_t                         （回归目标，A 未归一化）
  ```
  填充步写 0，且反向递推状态在遇到填充步时清零，防止跨“空洞”泄漏。

## 3. 手算轨迹（γ = λ = 0.5，见 `fixtures.py`）

- env A（真终止 t=2，r=[2,4,8]，V=[1,2,4]）：
  δ=[2,4,4]，A=[2+.25·5, 4+.25·4, 4]=[3.25, 5, 4]，R=[4.25,7,8]。
- env B（时间截断 t=1，V_final=8）：
  δ_1=4+.5·8−2=6，δ_0=2+.5·2−1=2，A_0=2+.25·6=3.5；
  `connect[1]=0`，即使后面有记录也不接入新 episode。
- env C（批次截断 t=4，V_final=16）：A_4=2+.5·16−1=9，向前折扣传播。
- env D（同一缓冲区连续两 episode，t=1 真终止，t=4 时间截断）：
  边界 `connect[1]=0`；ep1 的 A=[2.5,2]，ep2 的 A=[3.5,6,8]。
- env E（仅 t=0,1 有效，其余填充）：t=1 为该列最后有效步，
  用 `final_values[1]=8` bootstrap；`connect[1]=0`（t+1 是填充）。

**失败反例**（`test_reset_obs_misuse_reset_obs_is_demonstrably_wrong`）：
错误地把自动重置观测的价值（槽位里 `values[t+1]=3`）当作截断前最终观测，
δ_1 从 6 错成 3.5，且错误实现不断开边界，使新 episode 的 δ_2=−1.5 以
γλ=0.25 泄漏进旧轨迹，A_1 进一步错成 3.125。两类错误（错误 bootstrap
来源 + 跨边界递推）都被测试钉住。

## 4. λ = 0、λ = 1 与有偏价值的含义（不声称 GAE 无偏）

- **λ = 0**：`A_t = δ_t`，单步 TD 残差。偏差只来自 t 自己的
  `V_next`；后续价值估计误差不沿轨迹传播。是对优势函数
  `A^π(s,a)=Q^π−V^π` 的有偏估计（除非 V=V^π），但方差最小。
- **λ = 1**：截断/终止前 `A_t = Σ γ^k δ_{t+k}`（望远镜求和）
  = 采样回报（含截断点 bootstrap 项）− V(s_t)，即蒙特卡洛优势。
  它对**该批次采样到的回报**是无偏的，但作为 `A^π` 的估计，
  **当价值函数有偏时仍然有偏**，且偏差会沿整条轨迹传播。
  测试 `test_lambda_one_bias_when_values_wrong` 量化了这一点：
  截断点估值从真值 20 错成 8，则 A_1 偏差 γ(8−20)=−6，
  λ=1 时 A_0 再乘 γ，偏差 −3；λ=0 时 A_0 不受影响。
- **0 < λ < 1**：指数加权折衷（λ 越小偏差越局部、方差越小；
  λ 越大偏差传播越远、方差越大）。GAAE/GAE 文献中“近似无偏”是指
  V=V^π 的特例；笼统声称“GAE 无偏”是错误的。价值回归目标
  `R_t = V_old + A_t` 继承同样的偏差结构。

## 5. 优势归一化与损失

- 归一化只在**有效步**上做总体均值/总体标准差（population std），
  填充步写 0 且不计入统计：
  ```
  Â_t = (A_t − mean_M A) / (std_M A + ε)
  ```
  - 有效步为 0：全部输出 0，损失安全返回 0（零梯度）。
  - 只有 1 个有效步：方差为 0，归一化优势取 0（无法估计尺度），
    但未归一化 A 与回归目标 R 仍有定义。
- **价值回归目标使用未归一化 A**：`R_t = V_old(s_t) + A_t`；
  归一化优势**只**进入策略项。
- 离散动作 clipped surrogate + 价值 MSE + 熵：
  ```
  ℓ_policy = mean_M [ − min(ρ_t Â_t, clip(ρ_t,1−ε,1+ε) Â_t) ],
             ρ_t = exp(log π_θ(a_t|s_t) − log π_old(a_t|s_t))
  ℓ_value  = mean_M [ (V_θ(s_t) − R_t)^2 ]
  H        = mean_M [ −Σ_a π_θ(a|s_t) log π_θ(a|s_t) ]
  L        = ℓ_policy + c_v ℓ_value − c_H H
  ```
  `old_log_probs`、`Â`、`R_t` 在更新前一次性计算并 `detach`，
  在所有 epoch/minibatch 上保持固定（PPO 的 on-policy 约束）。
  填充步不参与任何均值；有限差分梯度核验在所有 ρ 严格落在
  (1−ε,1+ε) 内部、logits 非零的参数点上进行（避开 clip 折点与
  softmax 对称点），float64 下相对误差 < 1e-5。
