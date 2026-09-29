"""Auditable PPO core: explicit boundary masks, hand-derived gradients."""

from .gae import (
    compute_boundary_masks,
    compute_gae,
    compute_gae_scalar,
    normalize_advantages,
)
from .losses import (
    clipped_surrogate_loss,
    clipped_surrogate_per_step,
    entropy_bonus,
    total_loss,
    value_regression_loss,
)
from .network import Adam, forward, init_params, loss_and_grads
from .update import build_training_batch, ppo_update

__all__ = [
    "compute_boundary_masks",
    "compute_gae",
    "compute_gae_scalar",
    "normalize_advantages",
    "clipped_surrogate_loss",
    "clipped_surrogate_per_step",
    "entropy_bonus",
    "total_loss",
    "value_regression_loss",
    "Adam",
    "forward",
    "init_params",
    "loss_and_grads",
    "build_training_batch",
    "ppo_update",
]
