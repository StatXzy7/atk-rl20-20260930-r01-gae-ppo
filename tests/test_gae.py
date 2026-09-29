"""Hand-computed GAE trajectories + scalar/vectorized cross-checks.

Every expected number below is derived by hand in docs/PPO_MATH.md; the
tests exist to prove the implementation matches the math, not itself.
"""

import numpy as np

from ppo_audit import (
    compute_boundary_masks,
    compute_gae,
    compute_gae_scalar,
    normalize_advantages,
)


def test_true_termination_no_bootstrap():
    # gamma=1, lam=1; episode of 3 steps, true terminal at the last step.
    # next_values[2] is garbage on purpose (e.g. value of the reset obs):
    # a correct implementation must never read it.
    out = compute_gae(
        rewards=[1.0, 2.0, 3.0],
        values=[0.5, 0.5, 0.5],
        next_values=[0.5, 0.5, 7.7],
        terminated=[False, False, True],
        truncated=[False, False, False],
        valid=[True, True, True],
        gamma=1.0,
        lam=1.0,
    )
    # deltas: 1+0.5-0.5=1, 2+0.5-0.5=2, 3+0-0.5=2.5
    np.testing.assert_allclose(out["deltas"], [1.0, 2.0, 2.5])
    # A_2=2.5; A_1=2+2.5=4.5; A_0=1+4.5=5.5
    np.testing.assert_allclose(out["advantages"], [5.5, 4.5, 2.5])
    # returns = A + V = reward-to-go [6, 5, 3]
    np.testing.assert_allclose(out["returns"], [6.0, 5.0, 3.0])
    np.testing.assert_allclose(out["bootstrap_mask"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(out["connection_mask"], [1.0, 1.0, 0.0])


def test_time_truncation_bootstraps_final_obs_value():
    # gamma=0.5, lam=1; 2-step episode cut by the time limit.
    # next_values[1] = V(final_observation) = 2.0 (pre-reset final obs).
    out = compute_gae(
        rewards=[1.0, 1.0],
        values=[0.4, 0.6],
        next_values=[0.6, 2.0],
        terminated=[False, False],
        truncated=[False, True],
        valid=[True, True],
        gamma=0.5,
        lam=1.0,
    )
    # delta_1 = 1 + 0.5*2.0 - 0.6 = 1.4 ; delta_0 = 1 + 0.5*0.6 - 0.4 = 0.9
    np.testing.assert_allclose(out["deltas"], [0.9, 1.4])
    # A_1 = 1.4 ; A_0 = 0.9 + 0.5*1*1.4 = 1.6
    np.testing.assert_allclose(out["advantages"], [1.6, 1.4])
    # returns: R_1 = 1 + 0.5*2 = 2.0 ; R_0 = 1 + 0.5*1 + 0.25*2 = 2.0
    np.testing.assert_allclose(out["returns"], [2.0, 2.0])
    # truncation keeps the TD bootstrap but closes the recurrence
    np.testing.assert_allclose(out["bootstrap_mask"], [1.0, 1.0])
    np.testing.assert_allclose(out["connection_mask"], [1.0, 0.0])


def test_batch_end_bootstrap_unfinished_trajectory():
    # gamma=1, lam=1; batch ends mid-episode: must bootstrap V(s_T)=1.5.
    out = compute_gae(
        rewards=[1.0, 1.0, 1.0],
        values=[0.0, 0.0, 0.0],
        next_values=[0.0, 0.0, 1.5],
        terminated=[False, False, False],
        truncated=[False, False, False],
        valid=[True, True, True],
        gamma=1.0,
        lam=1.0,
    )
    # A_2 = 1+1.5 = 2.5 ; A_1 = 1+2.5 = 3.5 ; A_0 = 1+3.5 = 4.5
    np.testing.assert_allclose(out["advantages"], [4.5, 3.5, 2.5])
    np.testing.assert_allclose(out["returns"], [4.5, 3.5, 2.5])
    np.testing.assert_allclose(out["bootstrap_mask"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["connection_mask"], [1.0, 1.0, 0.0])


TWO_EP = dict(
    rewards=[1.0, 1.0, 2.0, 2.0, 2.0],
    values=[0.1, 0.2, 0.3, 0.4, 0.5],
    # 9.9 sits at the terminated step: it is the reset-obs value and must
    # be ignored; 3.0 is the final-obs value of the truncated episode.
    next_values=[0.2, 9.9, 0.4, 0.5, 3.0],
    terminated=[False, True, False, False, False],
    truncated=[False, False, False, False, True],
    valid=[True, True, True, True, True],
)


def test_two_episodes_in_one_buffer_no_leakage():
    out = compute_gae(**TWO_EP, gamma=1.0, lam=1.0)
    # episode A (steps 0-1, terminated):
    #   delta_1 = 1 - 0.2 = 0.8 ; delta_0 = 1 + 0.2 - 0.1 = 1.1
    #   A_1 = 0.8 ; A_0 = 1.1 + 0.8 = 1.9
    # episode B (steps 2-4, truncated, V(final)=3.0):
    #   delta_4 = 2 + 3.0 - 0.5 = 4.5 ; delta_3 = 2 + 0.5 - 0.4 = 2.1
    #   delta_2 = 2 + 0.4 - 0.3 = 2.1
    #   A_4 = 4.5 ; A_3 = 2.1 + 4.5 = 6.6 ; A_2 = 2.1 + 6.6 = 8.7
    np.testing.assert_allclose(out["advantages"], [1.9, 0.8, 8.7, 6.6, 4.5])
    np.testing.assert_allclose(out["bootstrap_mask"], [1, 0, 1, 1, 1])
    np.testing.assert_allclose(out["connection_mask"], [1, 0, 1, 1, 0])

    # episode B computed standalone must give identical advantages:
    # proves no advantage mass leaked across the auto-reset boundary.
    ep_b = compute_gae(
        rewards=[2.0, 2.0, 2.0],
        values=[0.3, 0.4, 0.5],
        next_values=[0.4, 0.5, 3.0],
        terminated=[False, False, False],
        truncated=[False, False, True],
        valid=[True, True, True],
        gamma=1.0,
        lam=1.0,
    )
    np.testing.assert_allclose(out["advantages"][2:], ep_b["advantages"])


def test_lambda_zero_is_pure_td():
    out = compute_gae(**TWO_EP, gamma=0.9, lam=0.0)
    # with lam=0 the recurrence is cut everywhere: A_t == delta_t
    np.testing.assert_allclose(out["advantages"], out["deltas"])
    np.testing.assert_allclose(
        out["deltas"],
        [
            1.0 + 0.9 * 0.2 - 0.1,   # 1.08
            1.0 - 0.2,               # 0.8  (terminated: no bootstrap)
            2.0 + 0.9 * 0.4 - 0.3,   # 2.06
            2.0 + 0.9 * 0.5 - 0.4,   # 2.05
            2.0 + 0.9 * 3.0 - 0.5,   # 4.2  (truncated: bootstrap)
        ],
    )


def test_closed_form_without_boundaries():
    # no resets, no padding: A_t = sum_k (gamma*lam)^k delta_{t+k}
    rng = np.random.default_rng(0)
    t_steps = 6
    gamma, lam = 0.9, 0.7
    rewards = rng.normal(size=t_steps)
    values = rng.normal(size=t_steps)
    next_values = rng.normal(size=t_steps)
    out = compute_gae(
        rewards, values, next_values,
        terminated=np.zeros(t_steps, bool),
        truncated=np.zeros(t_steps, bool),
        valid=np.ones(t_steps, bool),
        gamma=gamma, lam=lam,
    )
    deltas = rewards + gamma * next_values - values
    expected = np.array([
        sum((gamma * lam) ** k * deltas[t + k] for k in range(t_steps - t))
        for t in range(t_steps)
    ])
    np.testing.assert_allclose(out["deltas"], deltas)
    np.testing.assert_allclose(out["advantages"], expected, rtol=1e-12)


def test_scalar_reference_matches_vectorized():
    rng = np.random.default_rng(42)
    for _ in range(25):
        t_steps = int(rng.integers(1, 15))
        rewards = rng.normal(size=t_steps)
        values = rng.normal(size=t_steps)
        next_values = rng.normal(size=t_steps)
        terminated = rng.random(t_steps) < 0.25
        truncated = rng.random(t_steps) < 0.25
        valid = rng.random(t_steps) < 0.8
        gamma, lam = float(rng.uniform(0.8, 1.0)), float(rng.uniform(0.0, 1.0))
        vec = compute_gae(
            rewards, values, next_values, terminated, truncated, valid,
            gamma=gamma, lam=lam,
        )
        ref = compute_gae_scalar(
            rewards.tolist(), values.tolist(), next_values.tolist(),
            terminated.tolist(), truncated.tolist(), valid.tolist(),
            gamma=gamma, lam=lam,
        )
        np.testing.assert_allclose(vec["advantages"], ref["advantages"], atol=1e-12)
        np.testing.assert_allclose(vec["returns"], ref["returns"], atol=1e-12)
        np.testing.assert_allclose(vec["deltas"], ref["deltas"], atol=1e-12)
        np.testing.assert_allclose(vec["bootstrap_mask"], ref["bootstrap_mask"])
        np.testing.assert_allclose(vec["connection_mask"], ref["connection_mask"])


def test_masks_directly():
    bootstrap, connection = compute_boundary_masks(
        terminated=np.array([False, True, False, False]),
        truncated=np.array([False, False, True, False]),
        valid=np.array([True, True, True, False]),
    )
    # step 3 is padding: no bootstrap; connection into it is also closed.
    np.testing.assert_array_equal(bootstrap, [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(connection, [1.0, 0.0, 0.0, 0.0])


def test_normalization_excludes_padding():
    adv = np.array([1.0, 2.0, 3.0, 100.0])
    valid = np.array([True, True, True, False])
    out = normalize_advantages(adv, valid)
    std = np.std([1.0, 2.0, 3.0])
    np.testing.assert_allclose(out[:3], ([-1.0, 0.0, 1.0]) / std, rtol=1e-6)
    assert out[3] == 0.0  # padding never receives a normalized value


def test_normalization_single_valid_step_is_zero():
    out = normalize_advantages(np.array([5.0]), np.array([True]))
    np.testing.assert_array_equal(out, [0.0])


def test_normalization_all_padding_is_zero():
    out = normalize_advantages(np.array([1.0, 2.0]), np.array([False, False]))
    np.testing.assert_array_equal(out, [0.0, 0.0])


def test_masked_slots_may_hold_anything_even_inf():
    # the collector left inf/nan in slots the masks must ignore
    out = compute_gae(
        rewards=[1.0, 2.0, 3.0],
        values=[0.5, 0.5, 0.5],
        next_values=[0.5, 0.5, np.inf],   # terminal step: must be ignored
        terminated=[False, False, True],
        truncated=[False, False, False],
        valid=[True, True, True],
        gamma=1.0,
        lam=1.0,
    )
    np.testing.assert_allclose(out["advantages"], [5.5, 4.5, 2.5])
    ref = compute_gae_scalar(
        [1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [0.5, 0.5, np.inf],
        [False, False, True], [False, False, False], [True, True, True],
        gamma=1.0, lam=1.0,
    )
    np.testing.assert_allclose(ref["advantages"], [5.5, 4.5, 2.5])


def test_value_targets_use_unnormalized_advantages():
    out = compute_gae(**TWO_EP, gamma=1.0, lam=1.0)
    valid = np.array(TWO_EP["valid"])
    values = np.array(TWO_EP["values"])
    # regression target = raw advantage + old value, exactly the lambda-return
    np.testing.assert_allclose(
        out["returns"][valid], out["advantages"][valid] + values[valid]
    )
