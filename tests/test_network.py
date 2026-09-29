"""Finite-difference gradient verification and real-update behaviour."""

import numpy as np

from ppo_audit import (
    Adam,
    build_training_batch,
    forward,
    init_params,
    loss_and_grads,
    normalize_advantages,
    ppo_update,
)

OBS_DIM, HIDDEN, N_ACTIONS = 3, 5, 2
CLIP_EPS, VF_COEF, ENT_COEF = 0.2, 0.5, 0.01


def make_batch(seed, t_steps=6, ratio_jitter=0.05):
    """A small batch whose importance ratios stay strictly inside the clip
    region (away from the kinks at 1 +/- clip_eps)."""
    rng = np.random.default_rng(seed)
    params = init_params(OBS_DIM, HIDDEN, N_ACTIONS, seed=seed)
    obs = rng.normal(size=(t_steps, OBS_DIM))
    actions = rng.integers(0, N_ACTIONS, size=t_steps)
    valid = np.array([True, True, True, True, True, False])
    cache = forward(params, obs)
    logp_now = cache["log_probs"][np.arange(t_steps), actions]
    # ratios = exp(-jitter) in [e^-0.05, e^0.05]: deep inside (0.8, 1.2)
    old_log_probs = logp_now + rng.uniform(-ratio_jitter, ratio_jitter, t_steps)
    adv_norm = rng.normal(size=t_steps) * valid
    value_targets = rng.normal(size=t_steps)
    return params, obs, actions, old_log_probs, adv_norm, value_targets, valid


def finite_difference_grads(params, batch_args, h=1e-5):
    """Central differences of the total loss over every parameter entry."""
    grads_fd = {}
    for name in params:
        grads_fd[name] = np.zeros_like(params[name])
        it = np.nditer(params[name], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = params[name][idx]
            params[name][idx] = orig + h
            plus, _, _ = loss_and_grads(params, *batch_args)
            params[name][idx] = orig - h
            minus, _, _ = loss_and_grads(params, *batch_args)
            params[name][idx] = orig
            grads_fd[name][idx] = (plus - minus) / (2.0 * h)
            it.iternext()
    return grads_fd


def loss_args(obs, actions, old_log_probs, adv_norm, value_targets, valid):
    return (obs, actions, old_log_probs, adv_norm, value_targets, valid,
            CLIP_EPS, VF_COEF, ENT_COEF)


def test_finite_difference_gradients_inside_clip():
    params, obs, actions, old_lp, adv, targets, valid = make_batch(seed=1)
    args = loss_args(obs, actions, old_lp, adv, targets, valid)
    _, _, grads = loss_and_grads(params, *args)
    grads_fd = finite_difference_grads(params, args)
    for name in grads:
        np.testing.assert_allclose(
            grads[name], grads_fd[name], rtol=1e-4, atol=1e-6,
            err_msg=f"analytic vs finite-difference mismatch in {name}",
        )


def test_finite_difference_gradients_with_clip_active():
    # old_log_probs shifted by +0.5 -> ratios ~ e^-0.5 = 0.607, below 1-eps:
    # for negative advantages the clipped branch is active (zero gradient
    # through the ratio), for positive ones the unclipped branch is.  Both
    # regions are smooth here (far from the kink), so FD must still match.
    params, obs, actions, old_lp, adv, targets, valid = make_batch(seed=2)
    old_lp = old_lp + 0.5
    args = loss_args(obs, actions, old_lp, adv, targets, valid)
    _, _, grads = loss_and_grads(params, *args)
    grads_fd = finite_difference_grads(params, args)
    for name in grads:
        np.testing.assert_allclose(
            grads[name], grads_fd[name], rtol=1e-4, atol=1e-6,
            err_msg=f"clip-active mismatch in {name}",
        )


def test_clipped_step_has_zero_policy_gradient():
    # one valid step, ratio forced to 1.5 > 1 + eps with positive advantage
    params = init_params(OBS_DIM, HIDDEN, N_ACTIONS, seed=3)
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(1, OBS_DIM))
    actions = np.array([0])
    cache = forward(params, obs)
    old_lp = cache["log_probs"][0, 0] - np.log(1.5)  # ratio = 1.5
    _, _, grads = loss_and_grads(
        params, obs, actions,
        old_log_probs=np.array([old_lp]),
        adv_norm=np.array([1.0]),
        value_targets=np.array([0.0]),
        valid=np.array([True]),
        clip_eps=CLIP_EPS, vf_coef=0.0, ent_coef=0.0,
    )
    for name, grad in grads.items():
        np.testing.assert_array_equal(
            grad, np.zeros_like(grad),
            err_msg=f"clip-active step leaked gradient into {name}",
        )


def test_all_padding_batch_is_noop():
    params = init_params(OBS_DIM, HIDDEN, N_ACTIONS, seed=4)
    before = {k: v.copy() for k, v in params.items()}
    rng = np.random.default_rng(4)
    t_steps = 4
    total, parts, grads = loss_and_grads(
        params,
        obs=rng.normal(size=(t_steps, OBS_DIM)),
        actions=np.zeros(t_steps, dtype=np.int64),
        old_log_probs=np.zeros(t_steps),
        adv_norm=np.zeros(t_steps),
        value_targets=np.zeros(t_steps),
        valid=np.zeros(t_steps, dtype=bool),
        clip_eps=CLIP_EPS, vf_coef=VF_COEF, ent_coef=ENT_COEF,
    )
    assert total == 0.0
    assert parts == {"total": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
    for name, grad in grads.items():
        np.testing.assert_array_equal(grad, np.zeros_like(grad), err_msg=name)
    # and a full update on the empty batch must not move anything
    opt = Adam(params, lr=1e-2)
    batch = dict(
        obs=rng.normal(size=(t_steps, OBS_DIM)),
        actions=np.zeros(t_steps, dtype=np.int64),
        old_log_probs=np.zeros(t_steps),
        adv_norm=np.zeros(t_steps),
        value_targets=np.zeros(t_steps),
        valid=np.zeros(t_steps, dtype=bool),
    )
    history = ppo_update(params, opt, batch, CLIP_EPS, VF_COEF, ENT_COEF, epochs=3)
    assert all(p["total"] == 0.0 for p in history)
    for name in before:
        np.testing.assert_array_equal(params[name], before[name])


def test_single_valid_step_has_zero_policy_signal_but_runs():
    rng = np.random.default_rng(5)
    params = init_params(OBS_DIM, HIDDEN, N_ACTIONS, seed=5)
    obs = rng.normal(size=(3, OBS_DIM))
    valid = np.array([False, True, False])
    adv_norm = normalize_advantages(np.array([9.0, 4.0, 9.0]), valid)
    assert adv_norm[1] == 0.0  # documented single-sample behaviour
    total, parts, grads = loss_and_grads(
        params, obs,
        actions=np.array([0, 1, 0]),
        old_log_probs=np.zeros(3),
        adv_norm=adv_norm,
        value_targets=np.array([0.0, 1.0, 0.0]),
        valid=valid,
        clip_eps=CLIP_EPS, vf_coef=VF_COEF, ent_coef=ENT_COEF,
    )
    assert np.isfinite(total)
    assert parts["policy"] == 0.0
    assert parts["value"] > 0.0
    for grad in grads.values():
        assert np.all(np.isfinite(grad))


def synthetic_rollout_buffer(seed=11):
    """One buffer containing all four boundary cases plus padding:
    steps 0-2 episode A (true terminal), 3-4 episode B (time truncation),
    5-7 episode C (cut by the batch end), 8-9 padding."""
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
    # next observations as a vectorized collector would store them
    next_obs = rng.normal(size=(t_steps, OBS_DIM))
    next_obs[0], next_obs[1] = obs[1], obs[2]          # episode A continues
    next_obs[2] = rng.normal(size=OBS_DIM)             # reset obs (must be ignored)
    next_obs[3] = obs[4]                               # episode B continues
    next_obs[4] = rng.normal(size=OBS_DIM)             # final obs of truncated ep B
    next_obs[5], next_obs[6] = obs[6], obs[7]          # episode C continues
    next_obs[7] = rng.normal(size=OBS_DIM)             # batch-end bootstrap obs
    return obs, actions, rewards, next_obs, terminated, truncated, valid


def test_real_update_reduces_loss_with_frozen_targets():
    obs, actions, rewards, next_obs, terminated, truncated, valid = (
        synthetic_rollout_buffer()
    )
    params = init_params(OBS_DIM, HIDDEN, N_ACTIONS, seed=11)
    next_values = forward(params, next_obs)["values"]
    batch = build_training_batch(
        params, obs, actions, rewards, next_values,
        terminated, truncated, valid, gamma=0.99, lam=0.95,
    )
    frozen = {k: batch[k].copy() for k in
              ("old_log_probs", "adv_raw", "adv_norm", "value_targets")}

    opt = Adam(params, lr=5e-3)
    history = ppo_update(
        params, opt, batch, CLIP_EPS, VF_COEF, ENT_COEF, epochs=25
    )
    final_total, _, _ = loss_and_grads(
        params, batch["obs"], batch["actions"], batch["old_log_probs"],
        batch["adv_norm"], batch["value_targets"], batch["valid"],
        CLIP_EPS, VF_COEF, ENT_COEF,
    )
    assert final_total < history[0]["total"]
    # the frozen targets really stayed frozen through the whole update
    for name, values in frozen.items():
        np.testing.assert_array_equal(batch[name], values)
    # value targets are the *unnormalized* lambda-returns
    np.testing.assert_allclose(
        batch["value_targets"][valid],
        batch["adv_raw"][valid] + batch["old_values"][valid],
    )
