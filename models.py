"""Model building blocks: the toy predictor and a generic MLP.

`linear_predict` is the paper's predictor pred_phi(z_t, a_t); its `params` are the
learnable variable in the param-recovery and circular-embedding experiments. `mlp`
builds the small encoders / predictors / decoders used by the SIGReg experiments.
"""

import numpy as np
import torch


def linear_predict(state, action=0.0, params=np.array([0.8, 0.9, 0.85])):
    """Toy predictor pred(z_t, a_t): params * z_t + a_t, a smooth linear map.

    The action is an additive control input, the simplest stand-in for the action
    conditioning the paper applies via AdaLN inside the predictor. No modulo: a
    hard wrap makes the loss discontinuous and unlearnable by gradient descent;
    params < 1 keeps the rollout from blowing up instead. Works on numpy arrays or
    torch tensors.
    """
    return params * state + action


def mlp(d_in, d_out, hidden=64):
    """Two-hidden-layer tanh MLP."""
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, d_out),
    )
