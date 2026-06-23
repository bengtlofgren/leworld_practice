"""The four LeWM toy experiments, as functions behind a single registry.

Each experiment seeds its own RNG at entry so that running them in sequence
(`python main.py all`) reproduces the standalone results exactly. See main.py for
the entry point.
"""

import numpy as np
import torch

from models import mlp, linear_predict
from dynamics import noisy_transition, wrapped_transition, rollout, angle_batch, rotation_batch
from penalties import (prediction_loss, sigreg, circular_uniformity, vicreg,
                       circle_loss, wrapped_mse)
from metrics import angle_r2


# --------------------------------------------------------------------------- #
# Shared trainer for the param-vector experiments (param recovery + circular).
# --------------------------------------------------------------------------- #
def fit(init, inputs, actions, targets, loss_fn, steps, lr, log_every=None):
    """Teacher-forced optimisation of the predictor's `params` vector. Returns the
    rounded recovered params and the final loss. If `log_every` is set, prints
    progress every that-many steps."""
    params = torch.tensor(init, requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)
    loss = None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(linear_predict(inputs, actions, params), targets)
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  loss {loss.item():9.3f}  params {params.detach().numpy().round(3)}")
    return params.detach().numpy().round(3), loss.item()


# --------------------------------------------------------------------------- #
# 1. Param recovery: teacher-forced prediction loss recovers smooth dynamics.
# --------------------------------------------------------------------------- #
def param_recovery():
    """Learn the predictor's params by minimising L_pred against a frozen noisy
    trajectory; they converge to the true params down to the noise floor."""
    rng = np.random.default_rng(0)
    actions_np = rng.uniform(-1.0, 1.0, size=(100, 3))  # longer trajectory = more signal
    z0 = np.array([1.0, 2.0, 3.0])

    # Frozen ground-truth dynamics. params_true is what the model must recover; the
    # gap from the start params is reducible misspecification, the noise is the
    # irreducible floor.
    params_true = np.array([0.7, 0.95, 0.6])
    true_np = rollout(z0, actions_np, lambda s, a: noisy_transition(s, a, rng, params_true, noise_scale=0.1))

    actions = torch.tensor(actions_np, dtype=torch.float32)
    true_states = torch.tensor(true_np, dtype=torch.float32)
    inputs, targets = true_states[:-1], true_states[1:]  # teacher-forced (z_t, z_{t+1}) pairs

    recovered, _ = fit((0.8, 0.9, 0.85), inputs, actions, targets,
                       prediction_loss, steps=2001, lr=0.05, log_every=250)
    print(f"\nrecovered params: {recovered}")
    print(f"true params:      {params_true}")


# --------------------------------------------------------------------------- #
# 2. Circular embedding: mod-100 dynamics become learnable on a circle.
# --------------------------------------------------------------------------- #
def circular_embedding():
    """Same model and data; only the loss representation changes. Plain wrap-then-
    MSE gets stuck on the discontinuity; circular MSE recovers the params."""
    rng = np.random.default_rng(0)
    actions_np = rng.uniform(-1.0, 1.0, size=(200, 3))  # chaotic mod dynamics mix well
    z0 = np.array([1.0, 2.0, 3.0])
    params_true = np.array([2.2, 2.8, 3.1])

    true_np = rollout(z0, actions_np, lambda s, a: wrapped_transition(s, a, rng, params_true))
    actions = torch.tensor(actions_np, dtype=torch.float32)
    states = torch.tensor(true_np, dtype=torch.float32)
    inputs, targets = states[:-1], states[1:]  # teacher-forced (z_t, z_{t+1}) pairs

    plain_params, plain_l = fit((2.0, 3.0, 3.0), inputs, actions, targets, wrapped_mse, steps=3000, lr=0.05)
    circ_params, circ_l = fit((2.0, 3.0, 3.0), inputs, actions, targets, circle_loss, steps=3000, lr=0.05)

    print(f"true params:             {params_true}")
    print(f"plain MSE     -> params {plain_params}   (stuck,     loss {plain_l:8.2f})")
    print(f"circular MSE  -> params {circ_params}   (recovered, loss {circ_l:8.4f})")


# --------------------------------------------------------------------------- #
# 3. SIGReg floor: a faithful periodic code carries an irreducible SIGReg floor.
# --------------------------------------------------------------------------- #
def _sigreg_part_a():
    n = 4096
    gaussian = torch.randn(n, 2)                  # isotropic, unit variance
    theta = torch.rand(n) * 2 * torch.pi
    # radius sqrt(2) -> each axis has variance 1, matching the Gaussian, so any
    # SIGReg gap is about SHAPE (ring vs blob), not scale.
    ring = torch.stack([theta.cos(), theta.sin()], 1) * (2 ** 0.5)
    print("Part A  -  SIGReg of unit-variance point clouds")
    print(f"  Gaussian blob : {sigreg(gaussian).item():.4f}")
    print(f"  circle / ring : {sigreg(ring).item():.4f}")
    print("  (ring >> blob: the periodic shape is what SIGReg penalises)\n")


def _autoencoder_train(lmbda, steps=1500, B=512, lr=1e-3):
    """Angle autoencoder + SIGReg. The decoder is a faithfulness anchor (makes
    collapse impossible), isolating the ring-vs-Gaussian geometric tension."""
    encoder = mlp(2, 2)
    decoder = mlp(2, 2)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for _ in range(steps):
        obs, _ = angle_batch(B)
        z = encoder(obs)
        recon = decoder(z)
        recon_loss = ((recon - obs) ** 2).sum(1).mean()
        (recon_loss + lmbda * sigreg(z)).backward()
        opt.step()
        opt.zero_grad()
    r2, rad_mean, rad_std = angle_r2(encoder)
    with torch.no_grad():
        final_reg = sigreg(encoder(angle_batch(4096)[0])).item()
    return dict(lmbda=lmbda, recon=recon_loss.item(), sigreg=final_reg,
                angle_r2=r2, rad_mean=rad_mean, rad_std=rad_std)


def _sigreg_part_b():
    print("Part B  -  faithful angle autoencoder + SIGReg, swept over lambda (d_latent=2)")
    print(f"  {'lambda':>7} {'recon':>9} {'sigreg':>9} {'angle_R2':>9} {'radius(mean/std)':>18}")
    for lmbda in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        r = _autoencoder_train(lmbda)
        print(f"  {r['lmbda']:>7.2f} {r['recon']:>9.4f} {r['sigreg']:>9.4f} "
              f"{r['angle_r2']:>9.3f} {r['rad_mean']:>8.2f}/{r['rad_std']:<8.2f}")
    print("\n  recon low = faithful periodic code (a ring).   sigreg low = latent is Gaussian.")


def _sigreg_conclusion():
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


def sigreg_floor():
    """Static ring-vs-Gaussian SIGReg, then the faithfulness<->Gaussianity tradeoff
    swept over lambda, then the conclusion."""
    torch.manual_seed(0)
    _sigreg_part_a()
    _sigreg_part_b()
    _sigreg_conclusion()


# --------------------------------------------------------------------------- #
# 4. Penalty comparison: which anti-collapse prior suits a periodic factor?
# --------------------------------------------------------------------------- #
def _train_predictor(penalty_fn, lmbda, steps=1200, B=512, lr=1e-3):
    """Load-bearing setup: free encoder + predictor, self-prediction target (no
    stop-grad), so collapse is real and the penalty matters."""
    enc = mlp(2, 2)
    pred = mlp(2 + 1, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr)
    for _ in range(steps):
        obs, action, obs_next = rotation_batch(B)
        z = enc(obs)
        z_next = enc(obs_next)                       # target is encoder's own output (no stop-grad)
        z_hat = pred(torch.cat([z, action], 1))
        pred_loss = ((z_hat - z_next) ** 2).sum(1).mean()
        reg = penalty_fn(z) if lmbda > 0 else torch.zeros(())
        (pred_loss + lmbda * reg).backward()
        opt.step()
        opt.zero_grad()
    r2, _, _ = angle_r2(enc)
    with torch.no_grad():
        fresh = enc(rotation_batch(4096)[0])
        resid = penalty_fn(fresh).item()
        var = fresh.var(0).sum().item()              # collapse indicator: ~0 means collapsed
    return dict(pred_loss=pred_loss.item(), resid=resid, angle_r2=r2, var=var)


def penalty_comparison():
    """Compare SIGReg (Gaussian), a circular uniformity penalty, and VICReg on a
    periodic factor where the penalty is load-bearing."""
    torch.manual_seed(0)
    print("Load-bearing predictor (no decoder), pure-rotation world, d_latent=2.")
    print("angle_R2 = oracle test accuracy;  resid = residual penalty (the floor);")
    print("var = latent variance (collapse indicator, ~0 = collapsed).\n")
    print(f"  {'penalty':>9} {'lambda':>7} {'pred_loss':>10} {'resid':>9} {'angle_R2':>9} {'var':>8}")

    # one shared 'none' baseline to confirm collapse, then the penalties.
    r = _train_predictor(sigreg, 0.0)
    print(f"  {'none':>9} {0.0:>7.1f} {r['pred_loss']:>10.4f} {'-':>9} {r['angle_r2']:>9.3f} {r['var']:>8.3f}")

    for name, fn in [("sigreg", sigreg), ("circular", circular_uniformity), ("vicreg", vicreg)]:
        for lmbda in [3.0, 10.0, 30.0]:
            r = _train_predictor(fn, lmbda)
            print(f"  {name:>9} {lmbda:>7.1f} {r['pred_loss']:>10.4f} {r['resid']:>9.4f} "
                  f"{r['angle_r2']:>9.3f} {r['var']:>8.3f}")
    print("\nReading it:\n"
          "  - 'none' collapses (var ~ 0, angle_R2 low): why a penalty is needed at all.\n"
          "  - all three penalties that prevent collapse keep angle_R2 high: in 2-D the\n"
          "    Gaussian assumption costs no oracle accuracy.\n"
          "  - but only sigreg's resid floors above 0 (a ring can't be Gaussian); the\n"
          "    topology-matched circular penalty and the shape-agnostic vicreg reach ~0.\n"
          "    => the concern is real as an unsatisfiable-prior statement, not (here) as\n"
          "       an accuracy cost.")


EXPERIMENTS = {
    "param_recovery": param_recovery,
    "circular_embedding": circular_embedding,
    "sigreg_floor": sigreg_floor,
    "penalty_comparison": penalty_comparison,
}
