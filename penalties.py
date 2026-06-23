"""Losses and regularisers.

prediction_loss is LeWM's L_pred. sigreg / circular_uniformity / vicreg are the
three anti-collapse penalties compared in the penalty experiment. circle_embed /
circle_loss / wrapped_mse are the scalar wrap losses used by the circular-embedding
experiment.
"""

import torch


def prediction_loss(predicted_states, true_states):
    """LeWM prediction loss (Eq. 1): summed squared L2 norm over timesteps.

    The paper uses the squared L2 norm ||z_hat - z||_2^2 (no square root). Works on
    numpy arrays or torch tensors (both support `**` and `.sum()`).
    """
    diff = predicted_states - true_states
    return (diff**2).sum()


def sigreg(Z, n_proj=128, kappa=1.0, t_min=0.2, t_max=4.0, n_t=64):
    """Sketched isotropic-Gaussian regulariser (paper App. A).

    Project Z onto random unit directions and score each 1-D projection by the
    Epps-Pulley statistic T = int w(t) |phi_emp(t) - phi_0(t)|^2 dt, where phi_emp
    is the empirical characteristic function and phi_0(t) = exp(-t^2/2) is the
    standard-normal one. Differentiable, so it can be a training loss.
    """
    B, d = Z.shape
    U = torch.randn(d, n_proj)
    U = U / U.norm(dim=0, keepdim=True)          # unit-norm directions on S^{d-1}
    H = Z @ U                                     # (B, n_proj) 1-D projections
    t = torch.linspace(t_min, t_max, n_t)         # quadrature nodes in [0.2, 4]
    th = H[:, :, None] * t[None, None, :]         # (B, n_proj, n_t)
    re = th.cos().mean(0) - torch.exp(-0.5 * t**2)  # Re[phi_emp - phi_0]
    im = th.sin().mean(0)                          # Im[phi_emp]   (phi_0 is real)
    w = torch.exp(-0.5 * t**2 / kappa**2)         # Epps-Pulley weighting
    return torch.trapz(w * (re**2 + im**2), t, dim=1).mean()


def circular_uniformity(Z, n_harmonics=4, eps=1e-6):
    """Uniformity-on-the-circle penalty (directional-statistics analog of a
    normality test). Take each embedding's direction on the unit circle as a
    complex number w; for angles uniform on S^1, E[w^k] = 0 for all harmonics k.
    Penalise the squared magnitudes. Zero iff uniform; maximal when all directions
    coincide (i.e. collapse)."""
    u = Z / (Z.norm(dim=1, keepdim=True) + eps)
    w = torch.complex(u[:, 0], u[:, 1])
    return sum((w ** k).mean().abs() ** 2 for k in range(1, n_harmonics + 1))


def vicreg(Z, gamma=1.0, eps=1e-4):
    """VICReg-style: hinge each dim's std up to 1 (variance term) and decorrelate
    dims (covariance term). Shape-agnostic - a ring satisfies it perfectly."""
    std = (Z.var(0) + eps).sqrt()
    var_term = torch.relu(gamma - std).mean()
    Zc = Z - Z.mean(0)
    cov = (Zc.T @ Zc) / (Z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_term + (off ** 2).sum() / Z.shape[1]


def circle_embed(x, period=100.0):
    """Embed a scalar on a circle of circumference `period` -> (cos, sin).

    Periodic with that period, so values straddling a wrap are continuous.
    """
    angle = 2 * torch.pi * x / period
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def circle_loss(pred, target, period=100.0):
    """MSE between circle embeddings - smooth across the wrap."""
    return ((circle_embed(pred, period) - circle_embed(target, period)) ** 2).sum()


def wrapped_mse(pred, target, period=100.0):
    """MSE after wrapping the prediction - the modulo injects the discontinuity."""
    return ((pred % period - target) ** 2).sum()
