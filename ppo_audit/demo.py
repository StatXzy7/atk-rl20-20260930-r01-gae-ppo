"""端到端演示：自造小数据 -> 掩码 -> 逐步参考 GAE -> torch GAE 对拍
-> 归一化 -> 一次真实 PPO 更新。运行：python demo.py
"""

import numpy as np
import torch

from ppo_core import gae_reference, gae_torch, derive_masks, normalize_advantages, ppo_update
from fixtures import build_rollout, GAMMA, LAM

ENV_NAMES = ["A 真终止", "B 时间截断", "C 批次截断", "D 连续两episode", "E 填充截断"]


def show_grid(name, arr, env_axis=True):
    print(f"{name}（列=环境，行=时间步）:")
    a = np.asarray(arr)
    if a.dtype == bool:
        print(a.astype(int).T)
    else:
        print(np.round(a.T, 6))


def main():
    torch.manual_seed(0)
    r = build_rollout(with_obs=True, seed=42)

    kill, boot, conn = derive_masks(r.valid, r.terminated, r.truncated)
    print("=" * 70)
    print("环境：", ENV_NAMES)
    print("=" * 70)
    show_grid("kill 掩码（真终止，自举值=0）", kill)
    show_grid("bootstrap 掩码（必须用 final_values 自举）", boot)
    show_grid("connect 掩码（GAE 递推连接；截断/重置边界断开）", conn)

    adv_n, ret_n = gae_reference(r, GAMMA, LAM)
    adv_t, ret_t, _ = gae_torch(r, GAMMA, LAM)
    assert np.allclose(adv_t.numpy(), adv_n, atol=1e-12)
    assert np.allclose(ret_t.numpy(), ret_n, atol=1e-12)
    show_grid("GAE 优势 A（逐步标量参考实现 == torch 实现）", adv_n)
    show_grid("价值回归目标 R = V_old + A（未归一化）", ret_n)

    norm = normalize_advantages(adv_n, r.valid)
    show_grid("归一化优势（仅策略项使用；填充步为0）", norm)
    print(f"有效步归一化后均值={norm[r.valid].mean():.3e} "
          f"标准差={norm[r.valid].std():.6f}")

    print("-" * 70)
    print("lambda 端点演示（env A 同数据）:")
    a0, _ = gae_reference(r, GAMMA, 0.0)
    a1, _ = gae_reference(r, GAMMA, 1.0)
    print("  lam=0（纯单步TD残差） A[:3,0] =", np.round(a0[:3, 0], 6))
    print("  lam=.5               A[:3,0] =", np.round(adv_n[:3, 0], 6))
    print("  lam=1（蒙特卡洛回报-基线）A[:3,0] =", np.round(a1[:3, 0], 6))

    print("-" * 70)
    print("一次真实 PPO 更新（小型 MLP，旧策略量固定，Adam）:")
    res = ppo_update(r, gamma=GAMMA, lam=LAM, lr=5e-2, n_epochs=1,
                     minibatch_size=None, seed=1, verbose=True)
    b, a = res["before"], res["after"]
    print(f"  policy 损失 {b['policy']:.6f} -> {a['policy']:.6f}")
    print(f"  value  损失 {b['value']:.6f} -> {a['value']:.6f}")
    print(f"  熵          {b['entropy']:.6f} -> {a['entropy']:.6f}")
    print(f"  approx_kl={a['approx_kl']:.6f} clip_fraction={a['clip_fraction']:.3f}")


if __name__ == "__main__":
    main()
