"""Tiny tanh MLP policy/value network in pure NumPy with manual backprop.

Deliberately avoids autograd/RL libraries so every gradient in the PPO
update is auditable (and is verified against finite differences in tests).

Architecture
------------
    h      = tanh(W1 x + b1)                      (shared trunk)
    logits = W2 h + b2          -> categorical policy over discrete actions
    value  = wv . h + bv                          (scalar state value)

Total loss (means over valid steps only, N = number of valid steps):
    L = L_pi + vf_coef * L_v - ent_coef * H
with the PPO-Clip surrogate L_pi, value regression L_v and entropy bonus H
as defined in ppo_audit.losses.
"""

from __future__ import annotations

import numpy as np

PARAM_NAMES = ("W1", "b1", "W2", "b2", "wv", "bv")


def init_params(
    obs_dim: int, hidden_dim: int, n_actions: int, seed: int
) -> dict[str, np.ndarray]:
    """Random small-init parameters (fan-in scaled), fully deterministic."""
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0.0, 1.0 / np.sqrt(obs_dim), size=(hidden_dim, obs_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(n_actions, hidden_dim)),
        "b2": np.zeros(n_actions),
        "wv": rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=hidden_dim),
        "bv": np.zeros(1),
    }
    return {k: v.astype(np.float64) for k, v in params.items()}


def forward(
    params: dict[str, np.ndarray], obs: np.ndarray
) -> dict[str, np.ndarray]:
    """obs: (T, obs_dim) -> cache with logits, log_probs, probs, values, hidden."""
    x = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    z1 = x @ params["W1"].T + params["b1"]
    hidden = np.tanh(z1)
    logits = hidden @ params["W2"].T + params["b2"]
    logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    logsumexp = np.log(np.exp(logits).sum(axis=1, keepdims=True))
    log_probs = logits - logsumexp
    probs = np.exp(log_probs)
    values = hidden @ params["wv"] + params["bv"]
    return {
        "hidden": hidden,
        "logits": logits,
        "log_probs": log_probs,
        "probs": probs,
        "values": values,
    }


def loss_and_grads(
    params: dict[str, np.ndarray],
    obs: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    adv_norm: np.ndarray,
    value_targets: np.ndarray,
    valid: np.ndarray,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    """Total PPO loss and exact analytic gradients w.r.t. all parameters.

    ``old_log_probs``, ``adv_norm`` and ``value_targets`` are treated as
    constants (they are computed once from the rollout and frozen for the
    whole update).  Padding rows contribute exactly nothing.
    """
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    actions = np.asarray(actions, dtype=np.int64)
    old_log_probs = np.asarray(old_log_probs, dtype=np.float64)
    adv_norm = np.asarray(adv_norm, dtype=np.float64)
    value_targets = np.asarray(value_targets, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    t_steps = obs.shape[0]

    grads = {name: np.zeros_like(params[name]) for name in PARAM_NAMES}
    n_valid = int(valid.sum())
    cache = forward(params, obs)
    if n_valid == 0:
        # Defined behaviour for an empty batch: zero loss, zero gradients,
        # the caller's optimizer step becomes a no-op.
        parts = {"total": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
        return 0.0, parts, grads

    mask = valid.astype(np.float64)
    probs = cache["probs"]
    log_probs = cache["log_probs"]
    values = cache["values"]
    hidden = cache["hidden"]

    # ---- policy term: L_pi = -(1/N) sum m_t min(r A, clip(r) A) ----------
    logp_taken = log_probs[np.arange(t_steps), actions]
    ratio = np.exp(logp_taken - old_log_probs)
    obj_unclipped = ratio * adv_norm
    obj_clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    use_unclipped = obj_unclipped <= obj_clipped  # False where clip is active
    l_pi = -float((mask * np.minimum(obj_unclipped, obj_clipped)).sum() / n_valid)
    # d ratio / d logp_taken = ratio; gradient vanishes where the clip binds.
    dlogp = -mask * adv_norm * ratio * use_unclipped / n_valid

    # ---- value term: vf_coef * (1/N) sum m_t (v_t - R_t)^2 ---------------
    v_err = values - value_targets
    l_v = float((mask * v_err**2).sum() / n_valid)
    dvalues = 2.0 * vf_coef * mask * v_err / n_valid

    # ---- entropy term: -(ent_coef/N) sum m_t H_t -------------------------
    entropy_per_step = -(probs * log_probs).sum(axis=1)
    ent = float((mask * entropy_per_step).sum() / n_valid)

    total = l_pi + vf_coef * l_v - ent_coef * ent
    parts = {"total": total, "policy": l_pi, "value": l_v, "entropy": ent}

    # ---- backprop to logits ----------------------------------------------
    dlogits = np.zeros_like(log_probs)
    # policy gradient through log-softmax: d logp_a / d z = onehot(a) - p
    dlogits[np.arange(t_steps), actions] += dlogp
    dlogits -= dlogp[:, None] * probs
    # entropy: d H / d z_i = -p_i (log p_i + H)
    dlogits += (
        ent_coef
        * (mask / n_valid)[:, None]
        * probs
        * (log_probs + entropy_per_step[:, None])
    )

    # ---- backprop through the trunk --------------------------------------
    dhidden = dlogits @ params["W2"] + dvalues[:, None] * params["wv"][None, :]
    grads["W2"] = dlogits.T @ hidden
    grads["b2"] = dlogits.sum(axis=0)
    grads["wv"] = dvalues @ hidden
    grads["bv"] = np.array([dvalues.sum()])
    dz1 = dhidden * (1.0 - hidden**2)
    grads["W1"] = dz1.T @ obs
    grads["b1"] = dz1.sum(axis=0)
    return total, parts, grads


class Adam:
    """Minimal Adam optimizer over a flat dict of parameter arrays."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
    ) -> None:
        self.t += 1
        for name in PARAM_NAMES:
            g = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * g * g
            m_hat = self.m[name] / (1.0 - self.beta1**self.t)
            v_hat = self.v[name] / (1.0 - self.beta2**self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
