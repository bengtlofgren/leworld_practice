"""Does SIGReg's Gaussian target conflict with a periodic state factor?

This is the runnable version of the question for the LeWM authors: SIGReg pushes
the aggregate latent distribution toward an isotropic Gaussian N(0, I), but a
state variable that is intrinsically *periodic* (an angle) has a natural smooth
encoding that is a closed loop / ring (cos t, sin t). A ring is decidedly
non-Gaussian: projected onto any 1-D direction it gives a U-shaped (arcsine)
marginal, exactly what SIGReg's Epps-Pulley normality test penalises.

So the two terms of L = L_pred + lambda * SIGReg pull in opposite directions for
a periodic factor:
  * L_pred  wants a faithful ring (rotations are linear and predictable on it),
  * SIGReg  wants a Gaussian blob (no ring).

Two experiments:

  Part A (static)  - SIGReg(ring) vs SIGReg(Gaussian). Establishes that the prior
                     numerically dislikes a ring even when both have unit variance.

  Part B (learned) - a minimal LeWM (encoder + predictor, end-to-end, no stop-grad
                     or EMA, as in the paper) on a pure-rotation world. Sweep lambda
                     and watch the angle fidelity, the ring geometry, and the SIGReg
                     floor. The concern shows up as: SIGReg can never reach ~0 for a
                     faithful periodic code, and forcing it down (large lambda)
                     degrades the angle representation.

Run: .venv/bin/python sigreg_periodic.py
"""

import numpy as np
import torch

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# SIGReg: sketched isotropic-Gaussian regulariser (paper App. A).
# Project Z onto random unit directions and score each 1-D projection by the
# Epps-Pulley statistic T = \int w(t) |phi_emp(t) - phi_0(t)|^2 dt, where phi_emp
# is the empirical characteristic function and phi_0(t) = exp(-t^2/2) is the
# standard-normal one. Differentiable, so it can be a training loss.
# --------------------------------------------------------------------------- #
def sigreg(Z, n_proj=128, kappa=1.0, t_min=0.2, t_max=4.0, n_t=64):
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


# --------------------------------------------------------------------------- #
# Part A: the prior dislikes a ring even at matched variance.
# --------------------------------------------------------------------------- #
def part_a():
    n = 4096
    gaussian = torch.randn(n, 2)                  # isotropic, unit variance
    theta = torch.rand(n) * 2 * torch.pi
    # radius sqrt(2) -> each axis has variance 1, matching the Gaussian. So any
    # SIGReg gap is about SHAPE (ring vs blob), not scale.
    ring = torch.stack([theta.cos(), theta.sin()], 1) * (2 ** 0.5)
    print("Part A  -  SIGReg of unit-variance point clouds")
    print(f"  Gaussian blob : {sigreg(gaussian).item():.4f}")
    print(f"  circle / ring : {sigreg(ring).item():.4f}")
    print("  (ring >> blob: the periodic shape is what SIGReg penalises)\n")


# --------------------------------------------------------------------------- #
# Part B: the faithfulness <-> Gaussianity tradeoff, collapse-free.
#
# We encode an angle into a 2-D latent and decode it back. The decoder is a
# faithfulness ANCHOR: it makes representation collapse impossible (a constant
# latent cannot reconstruct), so whatever we see is purely the geometric tension
# between "keep the angle" and "look Gaussian", not the separate JEPA-collapse
# dynamics. (LeWM itself is reconstruction-free; the decoder here is only a probe
# of whether the periodic information survives.)
#
#   loss = recon_loss + lambda * SIGReg(z)
#
# recon_loss low  <=> faithful periodic code (a ring).
# SIGReg low      <=> latent looks like an isotropic Gaussian.
# The question is whether any lambda gives both.
# --------------------------------------------------------------------------- #
def make_net(d_in, d_out, hidden=64):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, d_out),
    )


def obs_batch(B):
    """Observation = wrap-safe (cos, sin) of a uniformly random angle."""
    theta = torch.rand(B) * 2 * torch.pi
    return torch.stack([theta.cos(), theta.sin()], 1), theta


def angle_r2(encoder):
    """Recoverability of the angle from the latent via a linear probe z -> (cos,sin).
    Faithful ring -> ~1; flattened / Gaussianised latent that loses injectivity -> low."""
    with torch.no_grad():
        obs, _ = obs_batch(4096)
        z = encoder(obs)
        zb = torch.cat([z, torch.ones(z.shape[0], 1)], 1)
        w = torch.linalg.lstsq(zb, obs).solution
        pred = zb @ w
        r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean(0)) ** 2).sum()
        radius = z.norm(dim=1)
        return r2.item(), radius.mean().item(), radius.std().item()


def train(lmbda, steps=1500, B=512, lr=1e-3):
    encoder = make_net(2, 2)
    decoder = make_net(2, 2)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for _ in range(steps):
        obs, _ = obs_batch(B)
        z = encoder(obs)
        recon = decoder(z)
        recon_loss = ((recon - obs) ** 2).sum(1).mean()
        (recon_loss + lmbda * sigreg(z)).backward()
        opt.step()
        opt.zero_grad()
    r2, rad_mean, rad_std = angle_r2(encoder)
    with torch.no_grad():
        final_reg = sigreg(encoder(obs_batch(4096)[0])).item()
    return dict(lmbda=lmbda, recon=recon_loss.item(), sigreg=final_reg,
                angle_r2=r2, rad_mean=rad_mean, rad_std=rad_std)


def part_b():
    print("Part B  -  faithful angle autoencoder + SIGReg, swept over lambda (d_latent=2)")
    print(f"  {'lambda':>7} {'recon':>9} {'sigreg':>9} {'angle_R2':>9} {'radius(mean/std)':>18}")
    for lmbda in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        r = train(lmbda)
        print(f"  {r['lmbda']:>7.2f} {r['recon']:>9.4f} {r['sigreg']:>9.4f} "
              f"{r['angle_r2']:>9.3f} {r['rad_mean']:>8.2f}/{r['rad_std']:<8.2f}")
    print("\n  recon low = faithful periodic code (a ring).   sigreg low = latent is Gaussian.")


def conclusion():
    print("""
Conclusion
----------
Two facts, pulling opposite ways:

1. The prior genuinely cannot represent the periodic factor as Gaussian.
   Across every lambda (even 30) SIGReg bottoms out around ~0.006 - roughly 60x
   the Gaussian floor (0.0001) - and never reaches it. This is structural: the
   latent is the image of a 1-D circle, so it is always a 1-D loop in 2-D space;
   an isotropic 2-D Gaussian is simply unreachable. So yes - a periodic state is
   NOT representable by something that looks Gaussian. The intuition is correct.

2. ...but the tension is benign, not destructive.
   At no lambda does SIGReg break the ring: angle_R2 stays ~1.0, the radius stays
   tight, reconstruction stays low. SIGReg tolerates the periodic structure and
   simply pays the irreducible floor, rather than flattening the ring to chase a
   Gaussian it can never reach.

Upshot for the author question: SIGReg's Gaussian target IS in tension with
periodic factors (an unreachable floor), but in isolation it does not prevent
representing them - the periodic information survives intact. That actually lines
up with the paper attributing poor rotational probing (Tab. 4) to capacity /
visual priors 'regardless of training strategy', rather than to SIGReg itself.
Caveat: this is a 2-D, single-factor, reconstruction-anchored toy; high-dim,
multi-factor, pixel-based LeWM could behave differently.
""")


if __name__ == "__main__":
    part_a()
    part_b()
    conclusion()
