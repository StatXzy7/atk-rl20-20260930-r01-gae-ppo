"""Failure counterexamples: what goes wrong if the auto-reset observation
is (mis)used as the transition's final observation.

In a vectorized collector, after ``terminated[t]`` or ``truncated[t]`` the
environment has already reset, so the "next observation" sitting in the
buffer is the *first observation of the new episode*.  Feeding its value
into the TD target of the ending transition is the classic boundary bug.
These tests pin down the exact error it would introduce and prove the
real implementation avoids it.
"""

import numpy as np

from ppo_audit import compute_gae

# Same hand-computed buffer as tests/test_gae.py::TWO_EP.
BUFFER = dict(
    rewards=[1.0, 1.0, 2.0, 2.0, 2.0],
    values=[0.1, 0.2, 0.3, 0.4, 0.5],
    terminated=[False, True, False, False, False],
    truncated=[False, False, False, False, True],
    valid=[True, True, True, True, True],
)
CORRECT_NEXT_VALUES = [0.2, 9.9, 0.4, 0.5, 3.0]
V_RESET = 9.9   # value of the reset observation after the true terminal
V_FINAL_TRUNC = 3.0  # value of the truncated episode's real final obs
V_RESET_AFTER_TRUNC = 0.7  # value of the new episode's first obs


def buggy_gae_uses_reset_value_at_termination(gamma, lam):
    """The bug: bootstrap every non-padding step, even true terminals."""
    t_steps = len(BUFFER["rewards"])
    adv = [0.0] * t_steps
    running = 0.0
    for t in range(t_steps - 1, -1, -1):
        # WRONG: no bootstrap mask -- the reset-obs value leaks into the
        # terminal transition's TD target.
        delta = (
            BUFFER["rewards"][t]
            + gamma * CORRECT_NEXT_VALUES[t]
            - BUFFER["values"][t]
        )
        ends = BUFFER["terminated"][t] or BUFFER["truncated"][t]
        conn = 0.0 if (t == t_steps - 1 or ends) else 1.0
        running = delta + gamma * lam * conn * running
        adv[t] = running
    return adv


def test_reset_obs_value_at_termination_inflates_advantages():
    gamma = lam = 1.0
    correct = compute_gae(**BUFFER, next_values=CORRECT_NEXT_VALUES,
                          gamma=gamma, lam=lam)
    buggy = buggy_gae_uses_reset_value_at_termination(gamma, lam)

    # hand-computed truth (see docs/PPO_MATH.md)
    np.testing.assert_allclose(correct["advantages"], [1.9, 0.8, 8.7, 6.6, 4.5])

    # the terminal transition's delta picks up exactly gamma * V(reset):
    # delta_1: correct 1 - 0.2 = 0.8  vs  buggy 1 + 9.9 - 0.2 = 10.7
    assert buggy[1] - correct["advantages"][1] == gamma * V_RESET
    # and the error recurses into every earlier step of the same episode:
    # A_0: correct 1.9  vs  buggy 1.1 + 10.7 = 11.8
    assert buggy[0] - correct["advantages"][0] == gamma * lam * gamma * V_RESET
    # episode B is untouched (the recurrence mask still blocks the leak)
    np.testing.assert_allclose(buggy[2:], correct["advantages"][2:])


def test_reset_obs_value_at_truncation_loses_the_bootstrap():
    # variant of the same bug on the truncation boundary: the collector
    # stores V(reset obs) = 0.7 instead of V(final obs) = 3.0 at step 4.
    wrong_next = [0.2, 9.9, 0.4, 0.5, V_RESET_AFTER_TRUNC]
    correct = compute_gae(**BUFFER, next_values=CORRECT_NEXT_VALUES,
                          gamma=1.0, lam=1.0)
    buggy = compute_gae(**BUFFER, next_values=wrong_next, gamma=1.0, lam=1.0)

    # delta_4 loses gamma * (V_final - V_reset) = 3.0 - 0.7 = 2.3 of value
    diff = correct["advantages"] - buggy["advantages"]
    assert diff[4] == V_FINAL_TRUNC - V_RESET_AFTER_TRUNC
    # and the whole truncated episode is shifted by the same amount
    np.testing.assert_allclose(
        diff[2:], 2.3
    )
    # episode A is unaffected
    np.testing.assert_allclose(correct["advantages"][:2], buggy["advantages"][:2])


def test_sign_flip_counterexample():
    # One-step episode ending in a penalized terminal state; the reset obs
    # of the next episode happens to have a high value.  Using it flips the
    # advantage sign: the agent would learn to *seek* the bad terminal.
    kwargs = dict(
        rewards=[-1.0],
        values=[0.0],
        terminated=[True],
        truncated=[False],
        valid=[True],
        gamma=1.0,
        lam=1.0,
    )
    correct = compute_gae(next_values=[5.0], **kwargs)  # 5.0 must be ignored
    assert correct["advantages"][0] == -1.0
    buggy = -1.0 + 1.0 * 5.0 - 0.0  # buggy collector bootstraps V(reset)
    assert buggy == 4.0  # -1 + 5: the penalty turned into a bonus
