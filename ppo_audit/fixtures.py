"""自造的小规模、可手算轨迹数据（gamma=lam=0.5）。

布局 (T=5, N=5)，逐列环境含义：

env A (n=0) 真终止于 t=2：        r=[2,4,8],  V=[1,2,4]
env B (n=1) 时间截断于 t=1：      r=[2,4],    V=[1,2], final_V(t1)=8
env C (n=2) 批次截断于末尾 t=4：  r=[2,4,1,0,2], V=[1,2,1,1,1], final_V(t4)=16
env D (n=3) 同缓冲区连续两 episode：
               episode1 真终止于 t=1：r=[2,4], V=[1,2]
               episode2 自动重置，时间截断于 t=4：r=[2,4,8], V=[1,2,4], final_V=8
env E (n=4) 只有 t0,t1 两个有效步，t2..t4 为填充；
               最后有效步 t1 未终止（批次截断语义），final_V=8

手算结果（gamma=lam=0.5），逐环境（列）：
  A（真终止 t2）: adv=[3.25, 5, 4], ret=[4.25, 7, 8]
  B（时间截断 t1, V_final=8）: adv=[3.5, 6], ret=[4.5, 8]
     注意 A_B(t0)=delta0+gl*delta1=3+0.25*2=3.5，连接在截断前一步仍成立，
     但截断步自身 t1 不再向 t2（新 episode）递推。
  C（批次截断 t4, V_final=16）: adv=[2.68359375, 2.734375, 0.9375, 1.75, 9],
                                ret=[3.68359375, 4.734375, 1.9375, 2.75, 10]
  D（连续两 episode）:
     ep1 t0..t1 真终止: adv=[2.5, 2], ret=[3.5, 4]
     ep2 t2..t4 时间截断(V_final=8): adv=[3.5, 6, 8], ret=[4.5, 8, 12]
     关键：t1(终止)与 t2(新episode)之间 connect=False；
           若错误地把 t2 的 V=1 当作 t1 的 final 观测，A(t1) 会从 2 变成 0.5。
  E（t0,t1 有效，其后填充；t1 批次截断 V_final=8）:
     adv=[3.5, 6], ret=[4.5, 8]；t1 与填充步 t2 之间 connect=False。
"""

import numpy as np

T, N = 5, 5
GAMMA, LAM = 0.5, 0.5


def build_rollout(with_obs=False, seed=0, obs_dim=3):
    rewards = np.zeros((T, N))
    values = np.zeros((T, N))
    final = np.zeros((T, N))
    old_lp = np.zeros((T, N))
    actions = np.zeros((T, N), dtype=np.int64)
    valid = np.zeros((T, N), dtype=bool)
    term = np.zeros((T, N), dtype=bool)
    trunc = np.zeros((T, N), dtype=bool)

    # env A
    n = 0
    valid[:3, n] = True
    rewards[:3, n] = [2, 4, 8]
    values[:3, n] = [1, 2, 4]
    term[2, n] = True
    # env B
    n = 1
    valid[:2, n] = True
    rewards[:2, n] = [2, 4]
    values[:2, n] = [1, 2]
    trunc[1, n] = True
    final[1, n] = 8.0
    # env C
    n = 2
    valid[:, n] = True
    rewards[:, n] = [2, 4, 1, 0, 2]
    values[:, n] = [1, 2, 1, 1, 1]
    final[4, n] = 16.0
    # env D
    n = 3
    valid[:, n] = True
    rewards[:, n] = [2, 4, 2, 4, 8]
    values[:, n] = [1, 2, 1, 2, 4]
    term[1, n] = True
    trunc[4, n] = True
    final[4, n] = 8.0
    # env E
    n = 4
    valid[:2, n] = True
    rewards[:2, n] = [2, 4]
    values[:2, n] = [1, 2]
    final[1, n] = 8.0

    rng = np.random.default_rng(seed)
    old_lp[valid] = -np.log(2.0)  # pi_old=0.5
    actions[valid] = rng.integers(0, 2, size=int(valid.sum()))

    obs = None
    if with_obs:
        rng_o = np.random.default_rng(seed + 1)
        obs = rng_o.standard_normal((T, N, obs_dim))
        obs[~valid] = 0.0

    from ppo_core import Rollout
    return Rollout(rewards, values, old_lp, actions, valid, term, trunc, final, obs)


EXPECTED_ADV = np.array([
    [3.25,      3.5,  2.68359375, 2.5,  3.5],
    [5.0,       6.0,  2.734375,   2.0,  6.0],
    [4.0,       0.0,  0.9375,     3.5,  0.0],
    [0.0,       0.0,  1.75,       6.0,  0.0],
    [0.0,       0.0,  9.0,        8.0,  0.0],
])

EXPECTED_RET = np.array([
    [4.25,      4.5,  3.68359375, 3.5,  4.5],
    [7.0,       8.0,  4.734375,   4.0,  8.0],
    [8.0,       0.0,  1.9375,     4.5,  0.0],
    [0.0,       0.0,  2.75,       8.0,  0.0],
    [0.0,       0.0,  10.0,       12.0, 0.0],
])

EXPECTED_KILL = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=bool)

EXPECTED_BOOT = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
], dtype=bool)

EXPECTED_CONN = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=bool)
# env D: t0(ep1内部)->t1 连接；t1 终止 -> t2(新ep) 断开；
#         t2->t3->t4 连接（t4 的 A_t4 被 t3 接入）；t4 截断无后续。
