"""GPTQ assignment baseline wrapper."""

from __future__ import annotations

import torch

from eigenflip.encoders.dense_reference import DenseGPTQ
from eigenflip.statistics.trust_region import LayerStats

from .state_adapter import state_from_grid


class GPTQAssignment:
    """GPTQ assignment on a fixed quantization grid."""

    name = "gptq"

    def __init__(self, damp: float = 0.01, order: str = "diag"):
        self.encoder = DenseGPTQ(damp=damp, order=order)

    @torch.no_grad()
    def apply_to_grid(
        self,
        grid,
        stats: LayerStats,
    ) -> tuple[torch.Tensor, dict]:
        state = state_from_grid(grid)
        dequantized, info = self.encoder.apply(state, stats)
        info = dict(info)
        info["assignment"] = self.name
        info["grid_scheme"] = grid.scheme
        return dequantized, info
