"""Toy LeWorldModel (LeWM) prediction loss — see papers/leworldmodel.pdf.

LeWM trains an encoder z_t = enc(o_t) and a predictor z_hat_{t+1} = pred(z_t, a_t).
The full objective is L = L_pred + lambda * SIGReg. Here we only implement L_pred
(Eq. 1), the next-embedding prediction loss; SIGReg comes later.

To keep things concrete, `phi` stands in for the predictor: a fixed toy map from
one latent state to the next. We roll it out from two different starting states to
get a "true" trajectory and a "predicted" one, then measure the prediction loss
between them.
"""

import numpy as np


def phi(state, params=np.array([2.0, 3.0, 3.0])):
    """Toy one-step predictor: elementwise (params * state) mod 100."""
    return (params * state) % 100


def rollout(state_zero, iterations=10):
    """Autoregressively roll the predictor forward, returning a (iterations, d) array."""
    states = [np.asarray(state_zero, dtype=float)]
    for _ in range(1, iterations):
        states.append(phi(states[-1]))
    return np.stack(states)


def prediction_loss(predicted_states, true_states):
    """LeWM prediction loss (Eq. 1): summed squared L2 norm over timesteps.

    The paper uses the squared L2 norm ||z_hat - z||_2^2 (no square root).
    """
    diff = predicted_states - true_states
    return np.sum(np.sum(diff**2, axis=1))


def main():
    true_states = rollout([1.2, 2.1, 3.3])
    predicted_states = rollout([1.0, 2.0, 3.0])
    loss = prediction_loss(predicted_states, true_states)
    print(f"prediction loss: {loss:.4f}")


if __name__ == "__main__":
    main()
