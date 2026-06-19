"""TFIC assignment baseline wrapper."""

from __future__ import annotations

import torch

from eigenflip.encoders.tfic_fast import TFICEncoder
from eigenflip.statistics.trust_region import LayerStats
from grid_baselines import VanillaQuantizationGrid

from .state_adapter import state_from_vanilla_grid


class TFICAssignment:
    """TFIC assignment on a fixed vanilla grid.

    The current EigenFlip TFIC implementation assumes non-negative integer code
    updates. This wrapper shifts signed symmetric grids to an equivalent
    non-negative internal representation before calling TFIC.
    """

    name = "tfic"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        eta: float = 1.0,
        gamma_th: float = 0.5,
        kappa: float = 2.0,
        gmax: int = 6,
        n_stages: int = 2,
        sweeps: int = 3,
        c_cand: float = 8.0,
        top_m: int = 32,
        chunk_cols: int = 256,
    ):
        self.encoder = TFICEncoder(
            alpha=alpha,
            beta=beta,
            eta=eta,
            gamma_th=gamma_th,
            kappa=kappa,
            gmax=gmax,
            n_stages=n_stages,
            sweeps=sweeps,
            c_cand=c_cand,
            top_m=top_m,
            chunk_cols=chunk_cols,
        )

    @torch.no_grad()
    def apply_to_grid(
        self,
        grid: VanillaQuantizationGrid,
        stats: LayerStats,
    ) -> tuple[torch.Tensor, dict]:
        state = state_from_vanilla_grid(grid, non_negative_codes=True)
        dequantized, info = self.encoder.apply(state, stats)
        info = dict(info)
        info["assignment"] = self.name
        info["grid_scheme"] = grid.scheme
        return dequantized, info
