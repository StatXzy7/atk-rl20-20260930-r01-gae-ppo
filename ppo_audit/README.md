# 可审计 PPO 核心训练步（自含实现，无现成 RL 训练器）

处理向量化自动重置环境的轨迹边界：真终止 / 时间截断 / 批次截断 /
同一缓冲区连续 episode / 填充步。只依赖 `numpy` 与 `torch`。

## 文件
- `ppo_core.py` — `Rollout`、两类掩码推导、逐步标量 numpy 参考 GAE、
  torch 向量化 GAE、优势归一化、clipped surrogate/价值/熵损失、
  小型 actor-critic 网络与一次真实 PPO 更新。
- `fixtures.py` — 5 列可手算短轨迹（真终止、时间截断、批次截断、
  连续两 episode、含填充）及手算期望数组。
- `test_ppo_core.py` — 16 个自动化测试。
- `demo.py` — 端到端演示（掩码、GAE、归一化、λ 端点、真实更新）。
- `MATH.md` — 数学推导、λ 语义与无偏性说明、失败反例。

## 运行
```bash
python3 -m pytest test_ppo_core.py -v
python3 demo.py
```

## 覆盖要点
- TD bootstrap 掩码与 GAE 递推连接掩码分别推导、分别测试；
- 时间截断用 `final_observation` 价值 bootstrap，但优势不跨重置边界；
- 批次末尾未终止轨迹正确 bootstrap；填充步不参与归一化/损失；
- 单有效步（归一化退化为 0）与全空（安全零损失零梯度）明确定义；
- 回归目标 `V_old + A`（未归一化），归一化优势仅用于策略项；
- λ=0 / λ=1 / 价值有偏的数值演示，不做“GAE 无偏”的笼统断言；
- numpy 手算优势/目标/损失 ↔ torch 对拍；float64 有限差分梯度核验
  （避开 clip 折点）；reset obs 误当 final obs 的失败反例；
- 随机初始化小网络完成一次真实优化更新，旧策略量在更新中固定。
