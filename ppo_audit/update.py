"""Glue: rollout batch -> fixed PPO targets -> real gradient updates.

The two-phase structure is the point: phase 1 evaluates the *old* network
once and freezes old log-probs, advantages and value targets; phase 2 runs
gradient steps that can only move the network parameters.
"""

from __future__ import annotations

import numpy as np

from .gae import compute_gae, normalize_advantages
from .network import Adam, forward, loss_and_grads


def build_training_batch(
    params: dict[str, np.ndarray],
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    valid: np.ndarray,
    gamma: float,
    lam: float,
) -> dict[str, np.ndarray]:
    """Phase 1: everything the update is allowed to condition on, frozen."""
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    valid = np.asarray(valid, dtype=bool)
    cache = forward(params, obs)
    actions = np.asarray(actions, dtype=np.int64)
    old_log_probs = cache["log_probs"][np.arange(obs.shape[0]), actions]
    old_values = cache["values"]

    gae = compute_gae(
        rewards=rewards,
        values=old_values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        valid=valid,
        gamma=gamma,
        lam=lam,
    )
    adv_norm = normalize_advantages(gae["advantages"], valid)
    return {
        "obs": obs,
        "actions": actions,
        "valid": valid,
        "old_log_probs": old_log_probs,       # frozen
        "old_values": old_values,             # frozen
        "adv_raw": gae["advantages"],         # frozen, unnormalized
        "adv_norm": adv_norm,                 # frozen, policy term only
        "value_targets": gae["returns"],      # frozen = adv_raw + old_values
        "bootstrap_mask": gae["bootstrap_mask"],
        "connection_mask": gae["connection_mask"],
    }


def ppo_update(
    params: dict[str, np.ndarray],
    optimizer: Adam,
    batch: dict[str, np.ndarray],
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int = 1,
) -> list[dict[str, float]]:
    """Phase 2: gradient steps with all targets held fixed.

    Returns the per-epoch loss components (recorded *before* each step).
    An all-padding batch is a defined no-op: losses are 0 and the
    parameters do not move.
    """
    history = []
    for _ in range(epochs):
        _, parts, grads = loss_and_grads(
            params,
            obs=batch["obs"],
            actions=batch["actions"],
            old_log_probs=batch["old_log_probs"],
            adv_norm=batch["adv_norm"],
            value_targets=batch["value_targets"],
            valid=batch["valid"],
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
        )
        optimizer.step(params, grads)
        history.append(parts)
    return history
