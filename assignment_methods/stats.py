"""Small LayerStats builders for assignment-method tests and demos."""

from __future__ import annotations

import torch

from eigenflip.statistics.trust_region import LayerStats


@torch.no_grad()
def identity_layer_stats(
    d: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
    keep_sigma: bool = True,
) -> LayerStats:
    """Build a simple H = I statistics object for small smoke tests.

    GPTQ and TFIC require a materialized ``Sigma``. This helper is not a
    calibration replacement; it only provides a deterministic toy metric for
    checking that assignment methods run end-to-end on a fixed grid.
    """
    dev = torch.device(device)
    mu = torch.zeros(d, device=dev, dtype=dtype)
    diag = torch.ones(d, device=dev, dtype=dtype)
    sigma = torch.eye(d, device=dev, dtype=dtype)
    return LayerStats(
        d=d,
        mu_hat=mu,
        diag_H=diag,
        diag_Sigma=diag,
        U_k=None,
        Lam_k=None,
        eps=eps,
        Sigma=sigma if keep_sigma else None,
        backend="identity",
    ).build()
