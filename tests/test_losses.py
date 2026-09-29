"""Hand-computed loss values, clip behaviour, padding exclusion."""

import math

import numpy as np

from ppo_audit import (
    clipped_surrogate_loss,
    clipped_surrogate_per_step,
    entropy_bonus,
    total_loss,
    value_regression_loss,
)

LOG = math.log


def hand_batch():
    # two valid steps; ratios 1.1 and 0.9 both inside clip_eps=0.2
    return dict(
        log_probs=np.array([LOG(0.55), LOG(0.45)]),
        old_log_probs=np.array([LOG(0.5), LOG(0.5)]),
        adv_norm=np.array([1.0, -1.0]),
        values_pred=np.array([1.0, 2.0]),
        value_targets=np.array([1.5, 1.0]),
        action_probs=np.array([[0.6, 0.4], [0.6, 0.4]]),
        valid=np.array([True, True]),
    )


def test_policy_loss_hand_computed():
    b = hand_batch()
    # step 0: -min(1.1*1, clip(1.1)*1) = -1.1
    # step 1: -min(0.9*(-1), 0.9*(-1)) = +0.9
    # mean over 2 valid steps: (-1.1 + 0.9) / 2 = -0.1
    assert math.isclose(
        clipped_surrogate_loss(
            b["log_probs"], b["old_log_probs"], b["adv_norm"], b["valid"], 0.2
        ),
        -0.1,
        rel_tol=1e-12,
    )


def test_value_loss_hand_computed():
    b = hand_batch()
    # ((1.0-1.5)^2 + (2.0-1.0)^2) / 2 = (0.25 + 1.0) / 2 = 0.625
    assert math.isclose(
        value_regression_loss(b["values_pred"], b["value_targets"], b["valid"]),
        0.625,
        rel_tol=1e-12,
    )


def test_entropy_hand_computed():
    b = hand_batch()
    h = -(0.6 * LOG(0.6) + 0.4 * LOG(0.4))  # 0.6730116670092565
    assert math.isclose(
        entropy_bonus(b["action_probs"], b["valid"]), h, rel_tol=1e-12
    )


def test_total_loss_hand_computed():
    b = hand_batch()
    h = -(0.6 * LOG(0.6) + 0.4 * LOG(0.4))
    expected = -0.1 + 0.5 * 0.625 - 0.01 * h
    parts = total_loss(**b, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    assert math.isclose(parts["total"], expected, rel_tol=1e-12)
    assert math.isclose(parts["policy"], -0.1, rel_tol=1e-12)
    assert math.isclose(parts["value"], 0.625, rel_tol=1e-12)
    assert math.isclose(parts["entropy"], h, rel_tol=1e-12)


def test_clip_bounds_the_objective():
    # ratio above 1+eps with positive advantage: objective capped at 1+eps
    val = clipped_surrogate_per_step(LOG(1.5), LOG(1.0), adv_norm=1.0, clip_eps=0.2)
    assert math.isclose(val, -1.2, rel_tol=1e-12)
    # ratio below 1-eps with negative advantage: capped at -(1-eps)*(-1)
    val = clipped_surrogate_per_step(LOG(0.5), LOG(1.0), adv_norm=-1.0, clip_eps=0.2)
    assert math.isclose(val, 0.8, rel_tol=1e-12)
    # scalar reference matches the vectorized loss on the same numbers
    vec = clipped_surrogate_loss(
        np.array([LOG(1.5), LOG(0.5)]),
        np.array([0.0, 0.0]),
        np.array([1.0, -1.0]),
        np.array([True, True]),
        clip_eps=0.2,
    )
    assert math.isclose(vec, (-1.2 + 0.8) / 2, rel_tol=1e-12)


def test_per_step_reference_matches_vectorized():
    rng = np.random.default_rng(7)
    t_steps = 50
    logp = rng.uniform(-2.0, -0.1, size=t_steps)
    old = logp + rng.uniform(-0.5, 0.5, size=t_steps)  # ratios across the clip
    adv = rng.normal(size=t_steps)
    valid = rng.random(t_steps) < 0.8
    vec = clipped_surrogate_loss(logp, old, adv, valid, clip_eps=0.2)
    ref_terms = [
        clipped_surrogate_per_step(logp[t], old[t], adv[t], 0.2)
        for t in range(t_steps)
        if valid[t]
    ]
    expected = sum(ref_terms) / len(ref_terms) if ref_terms else 0.0
    assert math.isclose(vec, expected, rel_tol=1e-12)


def test_padding_rows_do_not_change_losses():
    b = hand_batch()
    clean = total_loss(**b, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    padded = total_loss(
        log_probs=np.append(b["log_probs"], 50.0),
        old_log_probs=np.append(b["old_log_probs"], -50.0),
        adv_norm=np.append(b["adv_norm"], 1000.0),
        values_pred=np.append(b["values_pred"], 1e6),
        value_targets=np.append(b["value_targets"], -1e6),
        action_probs=np.vstack([b["action_probs"], [0.9, 0.1]]),
        valid=np.array([True, True, False]),
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )
    for key in clean:
        assert math.isclose(clean[key], padded[key], rel_tol=1e-12), key


def test_all_padding_gives_zero_losses():
    parts = total_loss(
        log_probs=np.array([0.3, 0.4]),
        old_log_probs=np.array([0.0, 0.0]),
        adv_norm=np.array([1.0, -1.0]),
        values_pred=np.array([1.0, 2.0]),
        value_targets=np.array([1.0, 2.0]),
        action_probs=np.array([[0.5, 0.5], [0.5, 0.5]]),
        valid=np.array([False, False]),
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )
    assert parts == {"total": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
