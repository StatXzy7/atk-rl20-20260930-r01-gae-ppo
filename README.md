# atk-rl20-20260930-r01-gae-ppo

可审计的 PPO 核心训练步：不调用现成 RL 训练器，纯 NumPy 实现，重点
正确处理向量化环境自动重置造成的轨迹边界问题。

## 布局

- `ppo_audit/gae.py` — TD bootstrap 掩码与 GAE 递推连接掩码的推导与
  实现（向量化 + 逐步标量参考 `compute_gae_scalar`），优势归一化
- `ppo_audit/losses.py` — 离散动作 clipped surrogate、价值回归、熵项
- `ppo_audit/network.py` — 手写反向传播的小型 tanh-MLP 策略/价值网络
  与 Adam（无 autograd，便于逐行审计）
- `ppo_audit/update.py` — 两阶段更新：先冻结 old log-prob / 优势 /
  回归目标，再做梯度步
- `tests/` — 手算轨迹、标量对照、有限差分梯度核验、reset-observation
  误用失败反例
- `docs/PPO_MATH.md` — 掩码推导、λ 与偏差–方差的精确讨论、手算过程
- `demo_update.py` — 自造小数据上的真实优化更新演示

## 运行

```bash
python3 -m pytest tests/ -v   # 自动化测试（29 个）
python3 demo_update.py        # 边界掩码审计表 + 真实更新 + 反例
```

仅依赖 numpy（测试用 pytest）。所有数据均为代码内自造的小规模数据。
