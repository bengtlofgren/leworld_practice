"""Toy LeWorldModel (LeWM) prediction loss — see papers/leworldmodel.pdf.

LeWM trains an encoder z_t = enc(o_t) and a predictor z_hat_{t+1} = pred(z_t, a_t).
The full objective is L = L_pred + lambda * SIGReg. Here we only implement L_pred
(Eq. 1), the next-embedding prediction loss; SIGReg comes later.

`phi` stands in for the predictor; its `params` are the *learnable* variable.
`true_state_transition` is the frozen environment (phi with the true params plus
noise). We generate a frozen ground-truth trajectory, then minimise the
teacher-forced prediction loss to recover the true params down to the noise floor.
"""

import numpy as np
import torch


def phi(state, action=0.0, params=np.array([0.8, 0.9, 0.85])):
    """Toy predictor pred(z_t, a_t): params * z_t + a_t, a smooth linear map.

    The action is an additive control input, the simplest stand-in for the
    action conditioning the paper applies via AdaLN inside the predictor.
    No modulo: a hard wrap makes the loss discontinuous and unlearnable by
    gradient descent. params < 1 keeps the rollout from blowing up instead.
    """
    return params * state + action


def true_state_transition(state, action, rng, params_true=np.array([0.8, 0.9, 0.85]), noise_scale=1.0):
    """Ground-truth environment dynamics: the predictor's map with the TRUE
    parameters, plus an irreducible noise term epsilon ~ N(0, noise_scale).

    The model's `phi` is graded against trajectories produced here. When
    params_true matches phi's params the only gap left is the noise, so the
    prediction loss bottoms out at a noise floor rather than zero.
    """
    without_noise_state = phi(state, action, params_true)
    if noise_scale <= 0:
        return without_noise_state
    epsilon = rng.normal(0.0, noise_scale, size=np.shape(state))
    return without_noise_state + epsilon


def rollout(state_zero, actions, transition=phi):
    """Autoregressively roll a transition forward under an action sequence.

    `transition(state, action) -> next_state` defaults to the model's `phi`;
    pass the true dynamics to generate the environment's trajectory instead.
    `actions` has shape (T, d); returns the (T + 1, d) state trajectory z_0 .. z_T.
    """
    states = [np.asarray(state_zero, dtype=float)]
    for action in actions:
        states.append(transition(states[-1], action))
    return np.stack(states)


def prediction_loss(predicted_states, true_states):
    """LeWM prediction loss (Eq. 1): summed squared L2 norm over timesteps.

    The paper uses the squared L2 norm ||z_hat - z||_2^2 (no square root).
    Works on numpy arrays or torch tensors (both support `**` and `.sum()`).
    """
    diff = predicted_states - true_states
    return (diff**2).sum()


def main():
    rng = np.random.default_rng(0)
    actions_np = rng.uniform(-1.0, 1.0, size=(100, 3))  # longer trajectory = more signal
    z0 = np.array([1.0, 2.0, 3.0])

    # Frozen ground-truth dynamics. params_true is what the model must recover;
    # the gap from phi's params is the reducible model misspecification, and the
    # noise term is the irreducible floor the loss cannot go below.
    params_true = np.array([0.7, 0.95, 0.6])
    true_np = rollout(z0, actions_np, lambda s, a: true_state_transition(s, a, rng, params_true, noise_scale=0.1))

    # Hand the frozen trajectory to torch as the supervision signal.
    actions = torch.tensor(actions_np, dtype=torch.float32)
    true_states = torch.tensor(true_np, dtype=torch.float32)
    inputs, targets = true_states[:-1], true_states[1:]  # teacher-forced (z_t, z_{t+1}) pairs

    # The predictor's params are the only learnable variable; start at phi's default.
    params_model = torch.tensor([0.8, 0.9, 0.85], requires_grad=True)
    optimizer = torch.optim.Adam([params_model], lr=0.05)

    for step in range(2001):
        optimizer.zero_grad()
        predicted = phi(inputs, actions, params_model)  # one teacher-forced step per pair
        loss = prediction_loss(predicted, targets)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            print(f"step {step:4d}  loss {loss.item():9.3f}  params {params_model.detach().numpy().round(3)}")

    print(f"\nrecovered params: {params_model.detach().numpy().round(3)}")
    print(f"true params:      {params_true}")


if __name__ == "__main__":
    main()
