"""Circular-embedding variant — the hard mod-100 dynamics become learnable.

Same toy as main.py, but the TRUE dynamics keep the modulo (`% 100`). We learn
the predictor's params two ways, changing ONLY how the loss measures error:

  * plain MSE      — wraps the prediction with `% 100`, then squares the gap.
                     The wrap makes the loss surface discontinuous -> stuck.
  * circular MSE   — embeds both prediction and target on a circle of period 100,
                     so 99.9 and 0.1 are neighbours. Smooth everywhere -> recovers.

Same model, same data: the only thing that changes is the *representation* of the
error. That is the whole point — the mod was never too complex to model, just the
wrong (discontinuous) thing to put inside a squared-error loss. Real LeWM keeps
everything smooth for exactly this reason (see papers/leworldmodel.pdf).
"""

import numpy as np
import torch

PERIOD = 100.0


def phi(state, action, params):
    """Raw linear predictor params * z + a (no modulo).

    The predictor outputs the unwrapped value; wrap-equivalence is the loss's job.
    """
    return params * state + action


def true_state_transition(state, action, rng, params_true, noise_scale=0.5):
    """Frozen environment: the same map but wrapped to [0, PERIOD) plus noise."""
    nxt = (params_true * state + action) % PERIOD
    return nxt + rng.normal(0.0, noise_scale, size=np.shape(state))


def rollout(state_zero, actions, transition):
    states = [np.asarray(state_zero, dtype=float)]
    for action in actions:
        states.append(transition(states[-1], action))
    return np.stack(states)


def circle(x):
    """Embed a scalar on a circle of circumference PERIOD -> (cos, sin).

    Periodic with period PERIOD, so values straddling a wrap are continuous.
    """
    angle = 2 * torch.pi * x / PERIOD
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def plain_loss(pred, target):
    """MSE after wrapping the prediction — the `% 100` injects the discontinuity."""
    return ((pred % PERIOD - target) ** 2).sum()


def circular_loss(pred, target):
    """MSE between circle embeddings — smooth across the wrap."""
    return ((circle(pred) - circle(target)) ** 2).sum()


def fit(inputs, actions, targets, loss_fn, init=(2.0, 3.0, 3.0), steps=3000, lr=0.05):
    params = torch.tensor(init, requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(phi(inputs, actions, params), targets)
        loss.backward()
        opt.step()
    return params.detach().numpy().round(3), loss.item()


def main():
    rng = np.random.default_rng(0)
    actions_np = rng.uniform(-1.0, 1.0, size=(200, 3))  # chaotic mod dynamics mix well
    z0 = np.array([1.0, 2.0, 3.0])
    params_true = np.array([2.2, 2.8, 3.1])

    # Frozen ground-truth trajectory under the wrapped (mod-100) dynamics.
    true_np = rollout(z0, actions_np, lambda s, a: true_state_transition(s, a, rng, params_true))
    actions = torch.tensor(actions_np, dtype=torch.float32)
    states = torch.tensor(true_np, dtype=torch.float32)
    inputs, targets = states[:-1], states[1:]  # teacher-forced (z_t, z_{t+1}) pairs

    plain_params, plain_l = fit(inputs, actions, targets, plain_loss)
    circ_params, circ_l = fit(inputs, actions, targets, circular_loss)

    print(f"true params:             {params_true}")
    print(f"plain MSE     -> params {plain_params}   (stuck,     loss {plain_l:8.2f})")
    print(f"circular MSE  -> params {circ_params}   (recovered, loss {circ_l:8.4f})")


if __name__ == "__main__":
    main()
