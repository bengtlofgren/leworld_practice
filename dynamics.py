"""Toy environments: ground-truth transitions, rollouts, and batch samplers.

The linear-dynamics worlds (param recovery, circular embedding) build on
linear_predict; the rotation world feeds the SIGReg experiments.
"""

import numpy as np
import torch

from models import linear_predict


def noisy_transition(state, action, rng, params_true=np.array([0.8, 0.9, 0.85]), noise_scale=1.0):
    """Ground-truth environment: linear_predict with the TRUE parameters, plus an
    irreducible noise term epsilon ~ N(0, noise_scale). noise_scale <= 0 gives the
    deterministic transition (no noise).

    The model's predictor is graded against trajectories produced here; when the
    params match, the prediction loss bottoms out at the noise floor, not zero.
    """
    without_noise = linear_predict(state, action, params_true)
    if noise_scale <= 0:
        return without_noise
    return without_noise + rng.normal(0.0, noise_scale, size=np.shape(state))


def wrapped_transition(state, action, rng, params_true, noise_scale=0.5, period=100.0):
    """Periodic ground-truth: the same map wrapped to [0, period) plus noise."""
    nxt = linear_predict(state, action, params_true) % period
    return nxt + rng.normal(0.0, noise_scale, size=np.shape(state))


def rollout(state_zero, actions, transition):
    """Autoregressively roll a transition forward under an action sequence.

    `transition(state, action) -> next_state`. `actions` has shape (T, d); returns
    the (T + 1, d) state trajectory z_0 .. z_T.
    """
    states = [np.asarray(state_zero, dtype=float)]
    for action in actions:
        states.append(transition(states[-1], action))
    return np.stack(states)


def angle_batch(B):
    """Observation = wrap-safe (cos, sin) of a uniformly random angle; returns the
    angle too (for probing)."""
    theta = torch.rand(B) * 2 * torch.pi
    return torch.stack([theta.cos(), theta.sin()], 1), theta


def rotation_batch(B):
    """Pure-rotation world: angle + angular-velocity action, wrap-safe (cos, sin)."""
    theta = torch.rand(B) * 2 * torch.pi
    action = torch.rand(B) - 0.5
    theta_next = theta + action
    obs = torch.stack([theta.cos(), theta.sin()], 1)
    obs_next = torch.stack([theta_next.cos(), theta_next.sin()], 1)
    return obs, action[:, None], obs_next
