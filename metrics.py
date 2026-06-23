"""Representation-quality metrics."""

import torch


def angle_r2(encoder, n=4096):
    """How recoverable is the angle from the latent? Encode fresh uniform angles,
    fit a linear probe z -> (cos, sin), and report R^2 plus the latent radius
    stats. A faithful ring -> ~1; a collapsed or flattened latent -> low.
    """
    with torch.no_grad():
        theta = torch.rand(n) * 2 * torch.pi
        obs = torch.stack([theta.cos(), theta.sin()], 1)
        z = encoder(obs)
        zb = torch.cat([z, torch.ones(z.shape[0], 1)], 1)   # augment with bias
        w = torch.linalg.lstsq(zb, obs).solution
        pred = zb @ w
        r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean(0)) ** 2).sum()
        radius = z.norm(dim=1)
        return r2.item(), radius.mean().item(), radius.std().item()
