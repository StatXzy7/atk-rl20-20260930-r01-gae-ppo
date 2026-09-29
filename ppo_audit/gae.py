"""GAE with explicit trajectory-boundary masks for vectorized auto-reset envs.

Two masks are derived and exposed separately (see docs/PPO_MATH.md):

* ``bootstrap_mask[t]`` -- TD bootstrap switch inside delta_t.
    delta_t = r_t + gamma * bootstrap_t * V(final_obs_t) - V(s_t)
  It is 1 for every valid step whose transition did not reach a *true*
  terminal state.  Time-limit truncations keep it at 1 (the episode would
  have continued, so V(final_obs) is a valid bootstrap), and so does an
  unfinished trajectory at the end of the sampled batch.

* ``connection_mask[t]`` -- GAE recurrence switch between A_t and A_{t+1}.
    A_t = delta_t + gamma * lam * connection_t * A_{t+1}
  It is 0 whenever record t+1 no longer belongs to the same episode as
  record t: after a true termination, after a time-limit truncation (the
  env auto-reset, so the next stored record starts a new episode), after
  the last record of the batch, and across padding.  A truncation may
  bootstrap its own TD target but must NOT let the advantage recurse into
  the next episode.

``next_values[t]`` is the value of the observation *after* transition t,
i.e. of ``final_observation_t``:
  - continuing step          -> V(s_{t+1})
  - time-limit truncation    -> V of the truncated episode's final obs
                                (NOT the reset observation of the new episode)
  - true termination         -> unused (masked out); may hold anything
  - last step of an unfinished batch -> V(s_T), the batch bootstrap value
"""

from __future__ import annotations

import numpy as np


def compute_boundary_masks(
    terminated: np.ndarray,
    truncated: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the TD bootstrap mask and the GAE connection mask.

    Parameters
    ----------
    terminated, truncated, valid : bool arrays of shape (T,)

    Returns
    -------
    bootstrap_mask : float array (T,), 1.0 where delta_t may bootstrap
        from ``next_values[t]``.
    connection_mask : float array (T,), 1.0 where A_t recurses into A_{t+1}.
        ``connection_mask[T-1]`` is always 0.0 (no next record in batch).
    """
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if not (terminated.shape == truncated.shape == valid.shape):
        raise ValueError("terminated, truncated and valid must have equal shapes")
    if terminated.ndim != 1:
        raise ValueError("expected 1-D (time-major) inputs")

    t_steps = terminated.shape[0]
    # A true terminal state has no successor value; truncation is only an
    # artifact of the time limit, so the final observation still carries a
    # well-defined state value and may be bootstrapped.
    bootstrap_mask = (valid & ~terminated).astype(np.float64)

    connection_mask = np.zeros(t_steps, dtype=np.float64)
    if t_steps > 1:
        # Record t+1 continues the episode of record t only if both are real
        # steps and step t did not end the episode in any way.  After a
        # truncation the env auto-resets, so the next record is a new episode.
        connection_mask[:-1] = (
            valid[:-1] & valid[1:] & ~terminated[:-1] & ~truncated[:-1]
        ).astype(np.float64)
    return bootstrap_mask, connection_mask


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    valid: np.ndarray,
    gamma: float,
    lam: float,
) -> dict[str, np.ndarray]:
    """Vectorized GAE.  All inputs are time-major arrays of shape (T,).

    Returns a dict with keys ``advantages``, ``returns`` (lambda-return value
    targets, i.e. advantages + old values), ``deltas``, ``bootstrap_mask``
    and ``connection_mask``.  Padded steps get exactly 0 advantage/return.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    t_steps = rewards.shape[0]
    if not (values.shape == next_values.shape == rewards.shape):
        raise ValueError("rewards, values and next_values must have equal shapes")

    bootstrap_mask, connection_mask = compute_boundary_masks(
        terminated, truncated, valid
    )

    # Select before multiplying: 0 * inf would be nan, and the contract is
    # that masked slots (true terminals, padding) may hold anything.
    next_values_safe = np.where(bootstrap_mask > 0.0, next_values, 0.0)
    deltas = rewards + gamma * bootstrap_mask * next_values_safe - values
    deltas = np.where(valid, deltas, 0.0)

    advantages = np.zeros(t_steps, dtype=np.float64)
    running = 0.0
    for t in range(t_steps - 1, -1, -1):
        if valid[t]:
            running = deltas[t] + gamma * lam * connection_mask[t] * running
            advantages[t] = running
        else:
            # Padding never carries advantage mass to earlier steps: the
            # connection mask into a padded step is already 0, and we reset
            # the accumulator defensively.
            running = 0.0

    returns = np.where(valid, advantages + values, 0.0)
    return {
        "advantages": advantages,
        "returns": returns,
        "deltas": deltas,
        "bootstrap_mask": bootstrap_mask,
        "connection_mask": connection_mask,
    }


def compute_gae_scalar(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    valid,
    gamma: float,
    lam: float,
) -> dict[str, list[float]]:
    """Step-by-step scalar reference implementation (plain Python floats).

    Mirrors :func:`compute_gae` exactly, one transition at a time, so every
    boundary decision is auditable without any vectorization.
    """
    t_steps = len(rewards)
    advantages = [0.0] * t_steps
    deltas = [0.0] * t_steps
    bootstraps = [0.0] * t_steps
    connections = [0.0] * t_steps

    next_adv = 0.0  # A_{t+1} while sweeping backwards
    for t in range(t_steps - 1, -1, -1):
        if not valid[t]:
            # Padding: no TD error, and the recurrence must not cross it.
            next_adv = 0.0
            continue
        # TD bootstrap mask: only a true terminal state forbids bootstrapping.
        bootstrap = 0.0 if terminated[t] else 1.0
        bootstraps[t] = bootstrap
        # select before multiplying so a garbage next_values slot (e.g. inf
        # at a terminal step) cannot produce 0 * inf = nan
        bootstrap_value = gamma * next_values[t] if bootstrap else 0.0
        delta = rewards[t] + bootstrap_value - values[t]
        deltas[t] = delta
        # GAE connection mask: is record t+1 still the same episode?
        is_last_record = t == t_steps - 1 or not valid[t + 1]
        ends_episode = bool(terminated[t]) or bool(truncated[t])
        connection = 0.0 if (is_last_record or ends_episode) else 1.0
        connections[t] = connection
        next_adv = delta + gamma * lam * connection * next_adv
        advantages[t] = next_adv

    returns = [
        (advantages[t] + values[t]) if valid[t] else 0.0 for t in range(t_steps)
    ]
    return {
        "advantages": advantages,
        "returns": returns,
        "deltas": deltas,
        "bootstrap_mask": bootstraps,
        "connection_mask": connections,
    }


def normalize_advantages(
    advantages: np.ndarray, valid: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Normalize advantages over *valid* steps only; padding stays 0.

    Defined edge-case behaviour:
      * 0 valid steps -> all-zero result (caller must skip the update).
      * 1 valid step  -> 0.0 (a single sample centered on its own mean has
        no policy signal; we do not invent a scale for it).
    """
    advantages = np.asarray(advantages, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = np.zeros_like(advantages)
    n_valid = int(valid.sum())
    if n_valid <= 1:
        return out
    sel = advantages[valid]
    mean = sel.mean()
    std = sel.std()  # population std (ddof=0), the PPO convention
    out[valid] = (sel - mean) / (std + eps)
    return out
