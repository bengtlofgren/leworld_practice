"""Which anti-collapse prior suits a KNOWN-periodic factor? (load-bearing setup)

Extends sigreg_periodic.py. Here there is NO decoder: the encoder is free and the
prediction target is the encoder's own next output (real LeWM, no stop-grad), so
representation collapse is a live failure mode and the penalty is load-bearing.

We compare three penalties that all prevent collapse but assume different latent
shapes, on a pure-rotation (periodic) world:

  * sigreg   - 1-D projections should be standard Gaussian (the paper). Isotropic
               Gaussian target; a ring can never match it  -> irreducible floor.
  * circular - the embedding angles should be uniform on the circle (Fourier /
               Rayleigh uniformity, the directional-statistics analog of a
               normality test). A faithful ring IS uniform   -> no floor.
  * vicreg   - unit variance per dim + decorrelation (shape-agnostic). A ring
               satisfies it                                   -> no floor.

We report the oracle test accuracy (angle_R2: linear probe z -> (cos,sin) on fresh
angles) and the residual penalty value (the floor). The question: does matching the
prior's topology to the data remove the floor, and does the floor cost any accuracy?

Run: .venv/bin/python penalty_comparison.py
"""

import torch
from sigreg_periodic import sigreg, make_net, obs_batch, angle_r2

torch.manual_seed(0)


def circular(Z, n_harmonics=4, eps=1e-6):
    """Uniformity-on-the-circle penalty. Take each embedding's direction on the
    unit circle as a complex number w; for angles uniform on S^1, E[w^k] = 0 for
    all harmonics k. Penalise the squared magnitudes. Zero iff uniform; maximal
    when all directions coincide (i.e. collapse)."""
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


def rot_batch(B):
    """Pure-rotation world: angle + angular-velocity action, wrap-safe (cos,sin)."""
    theta = torch.rand(B) * 2 * torch.pi
    action = torch.rand(B) - 0.5
    theta_next = theta + action
    obs = torch.stack([theta.cos(), theta.sin()], 1)
    obs_next = torch.stack([theta_next.cos(), theta_next.sin()], 1)
    return obs, action[:, None], obs_next


def train_predictor(penalty_fn, lmbda, steps=1200, B=512, lr=1e-3):
    enc = make_net(2, 2)
    pred = make_net(2 + 1, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr)
    for _ in range(steps):
        obs, action, obs_next = rot_batch(B)
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
        fresh = enc(rot_batch(4096)[0])
        resid = penalty_fn(fresh).item()
        var = fresh.var(0).sum().item()              # collapse indicator: ~0 means collapsed
    return dict(pred_loss=pred_loss.item(), resid=resid, angle_r2=r2, var=var)


def main():
    print("Load-bearing predictor (no decoder), pure-rotation world, d_latent=2.")
    print("angle_R2 = oracle test accuracy;  resid = residual penalty (the floor);")
    print("var = latent variance (collapse indicator, ~0 = collapsed).\n")
    print(f"  {'penalty':>9} {'lambda':>7} {'pred_loss':>10} {'resid':>9} {'angle_R2':>9} {'var':>8}")

    # one shared 'none' baseline to confirm collapse, then the penalties.
    r = train_predictor(sigreg, 0.0)
    print(f"  {'none':>9} {0.0:>7.1f} {r['pred_loss']:>10.4f} {'-':>9} {r['angle_r2']:>9.3f} {r['var']:>8.3f}")

    for name, fn in [("sigreg", sigreg), ("circular", circular), ("vicreg", vicreg)]:
        for lmbda in [3.0, 10.0, 30.0]:
            r = train_predictor(fn, lmbda)
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


if __name__ == "__main__":
    main()
