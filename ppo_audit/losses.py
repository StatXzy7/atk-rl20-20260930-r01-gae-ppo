"""PPO loss terms for discrete actions, masked to valid steps.

Everything here is a pure function of already-computed quantities.  During
an update the old log-probabilities, the (normalized) advantages and the
value-regression targets are computed once from the rollout and then held
fixed; only the network parameters move.

Conventions
-----------
* policy term   : PPO-Clip surrogate  L_pi  = -E[min(r A, clip(r) A)]
* value term    : regression onto the *unnormalized* lambda-return target
                  R_t = A_t^{raw} + V_old(s_t);  L_v = E[(V_pred - R)^2]
* entropy term  : H of the categorical policy, added as a bonus
* total         : L = L_pi + vf_coef * L_v - ent_coef * E[H]

All expectations are means over valid (non-padding) steps only.  With zero
valid steps every term is exactly 0.0 and every gradient is exactly 0.
"""

from __future__ import annotations

import numpy as np


def _valid_count(valid: np.ndarray) -> int:
    return int(np.asarray(valid, dtype=bool).sum())


def clipped_surrogate_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    adv_norm: np.ndarray,
    valid: np.ndarray,
    clip_eps: float,
) -> float:
    """L_pi = -mean_valid[ min(ratio*A, clip(ratio, 1-e, 1+e)*A) ]."""
    n = _valid_count(valid)
    if n == 0:
        return 0.0
    log_probs = np.asarray(log_probs, dtype=np.float64)
    old_log_probs = np.asarray(old_log_probs, dtype=np.float64)
    adv_norm = np.asarray(adv_norm, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    ratio = np.exp(log_probs[valid] - old_log_probs[valid])
    adv = adv_norm[valid]
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return float(-np.minimum(unclipped, clipped).sum() / n)


def value_regression_loss(
    values_pred: np.ndarray,
    value_targets: np.ndarray,
    valid: np.ndarray,
) -> float:
    """L_v = mean_valid[ (V_pred - R)^2 ] with R = A_raw + V_old."""
    n = _valid_count(valid)
    if n == 0:
        return 0.0
    values_pred = np.asarray(values_pred, dtype=np.float64)
    value_targets = np.asarray(value_targets, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    err = values_pred[valid] - value_targets[valid]
    return float((err**2).sum() / n)


def entropy_bonus(
    action_probs: np.ndarray,
    valid: np.ndarray,
    log_floor: float = 1e-12,
) -> float:
    """mean_valid[ H(pi) ] with H = -sum_a p log p."""
    n = _valid_count(valid)
    if n == 0:
        return 0.0
    action_probs = np.asarray(action_probs, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    probs = action_probs[valid]
    ent = -(probs * np.log(np.maximum(probs, log_floor))).sum(axis=-1)
    return float(ent.sum() / n)


def total_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    adv_norm: np.ndarray,
    values_pred: np.ndarray,
    value_targets: np.ndarray,
    action_probs: np.ndarray,
    valid: np.ndarray,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> dict[str, float]:
    """Combine the three terms; returns each component for auditing."""
    l_pi = clipped_surrogate_loss(log_probs, old_log_probs, adv_norm, valid, clip_eps)
    l_v = value_regression_loss(values_pred, value_targets, valid)
    ent = entropy_bonus(action_probs, valid)
    return {
        "total": l_pi + vf_coef * l_v - ent_coef * ent,
        "policy": l_pi,
        "value": l_v,
        "entropy": ent,
    }


def clipped_surrogate_per_step(
    log_prob: float,
    old_log_prob: float,
    adv_norm: float,
    clip_eps: float,
) -> float:
    """Scalar reference for a single transition (used by tests/docs)."""
    ratio = float(np.exp(log_prob - old_log_prob))
    unclipped = ratio * adv_norm
    clipped = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps) * adv_norm
    return -min(unclipped, clipped)
