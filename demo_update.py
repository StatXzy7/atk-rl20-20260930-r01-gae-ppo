"""End-to-end demo on self-made small data:

1. builds one buffer containing all four boundary cases + padding,
2. derives the boundary masks and GAE (printed for audit),
3. freezes old log-probs / advantages / value targets,
4. runs real Adam updates on a randomly initialized tiny network,
5. shows the reset-observation misuse counterexample.

Run:  python3 demo_update.py
"""

import numpy as np

from ppo_audit import (
    Adam,
    build_training_batch,
    compute_gae,
    forward,
    init_params,
    loss_and_grads,
    ppo_update,
)

OBS_DIM, HIDDEN, N_ACTIONS = 4, 6, 2
GAMMA, LAM = 0.99, 0.95
CLIP_EPS, VF_COEF, ENT_COEF = 0.2, 0.5, 0.01


def build_buffer(seed=0):
    """T=10 buffer: ep A steps 0-2 (true terminal), ep B steps 3-4
    (time-limit truncation), ep C steps 5-7 (cut by batch end),
    steps 8-9 padding."""
    rng = np.random.default_rng(seed)
    t_steps = 10
    obs = rng.normal(size=(t_steps, OBS_DIM))
    actions = rng.integers(0, N_ACTIONS, size=t_steps)
    rewards = rng.normal(size=t_steps)
    terminated = np.zeros(t_steps, dtype=bool)
    truncated = np.zeros(t_steps, dtype=bool)
    valid = np.ones(t_steps, dtype=bool)
    terminated[2] = True
    truncated[4] = True
    valid[8:] = False

    next_obs = rng.normal(size=(t_steps, OBS_DIM))
    next_obs[0], next_obs[1] = obs[1], obs[2]   # episode A continues
    next_obs[2] = rng.normal(size=OBS_DIM)      # auto-reset obs (must be ignored)
    next_obs[3] = obs[4]                        # episode B continues
    next_obs[4] = rng.normal(size=OBS_DIM)      # final obs of truncated episode B
    next_obs[5], next_obs[6] = obs[6], obs[7]   # episode C continues
    next_obs[7] = rng.normal(size=OBS_DIM)      # batch-end bootstrap obs
    return obs, actions, rewards, next_obs, terminated, truncated, valid


def main():
    obs, actions, rewards, next_obs, terminated, truncated, valid = build_buffer()
    params = init_params(OBS_DIM, HIDDEN, N_ACTIONS, seed=0)
    next_values = forward(params, next_obs)["values"]

    batch = build_training_batch(
        params, obs, actions, rewards, next_values,
        terminated, truncated, valid, gamma=GAMMA, lam=LAM,
    )

    print("=== boundary audit (per recorded step) ===")
    print(f"{'t':>3} {'valid':>6} {'term':>5} {'trunc':>6} "
          f"{'bootstrap':>10} {'connect':>8} {'adv':>9} {'target':>9}")
    for t in range(obs.shape[0]):
        print(f"{t:>3} {str(valid[t]):>6} {str(terminated[t]):>5} "
              f"{str(truncated[t]):>6} "
              f"{batch['bootstrap_mask'][t]:>10.1f} "
              f"{batch['connection_mask'][t]:>8.1f} "
              f"{batch['adv_raw'][t]:>9.4f} {batch['value_targets'][t]:>9.4f}")

    print("\n=== real PPO update (targets frozen, 25 Adam epochs) ===")
    opt = Adam(params, lr=5e-3)
    history = ppo_update(params, opt, batch, CLIP_EPS, VF_COEF, ENT_COEF, epochs=25)
    for epoch in [0, 1, 2, 4, 9, 24]:
        p = history[epoch]
        print(f"epoch {epoch:>3}: total={p['total']:+.5f} "
              f"policy={p['policy']:+.5f} value={p['value']:.5f} "
              f"entropy={p['entropy']:.5f}")
    final_total, _, _ = loss_and_grads(
        params, batch["obs"], batch["actions"], batch["old_log_probs"],
        batch["adv_norm"], batch["value_targets"], batch["valid"],
        CLIP_EPS, VF_COEF, ENT_COEF,
    )
    print(f"final total loss after update: {final_total:+.5f} "
          f"(started at {history[0]['total']:+.5f})")

    print("\n=== counterexample: reset obs misused as final obs ===")
    buffer = dict(
        rewards=[1.0, 1.0], values=[0.1, 0.2],
        terminated=[False, True], truncated=[False, False],
        valid=[True, True],
    )
    correct = compute_gae(**buffer, next_values=[0.2, 9.9], gamma=1.0, lam=1.0)
    # buggy variant: bootstrap the terminal step from V(reset obs) = 9.9
    deltas_buggy = [1.0 + 0.2 - 0.1, 1.0 + 9.9 - 0.2]
    adv_buggy = [deltas_buggy[0] + deltas_buggy[1], deltas_buggy[1]]
    print(f"correct advantages : {correct['advantages'].round(4).tolist()}")
    print(f"buggy   advantages : {[round(a, 4) for a in adv_buggy]}"
          "  <- terminal step absorbed V(reset obs) = 9.9")


if __name__ == "__main__":
    main()
