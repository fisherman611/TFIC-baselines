"""RTN assignment baseline."""

from __future__ import annotations

import torch

from grid_baselines import VanillaQuantizationGrid


class RTNAssignment:
    """Round-to-nearest assignment on a fixed vanilla grid."""

    name = "rtn"

    @torch.no_grad()
    def apply_to_grid(self, grid: VanillaQuantizationGrid) -> tuple[torch.Tensor, dict]:
        integer_weights, dequantized = grid.round_to_nearest()
        return dequantized, {
            "assignment": self.name,
            "codes": integer_weights,
            "grid_scheme": grid.scheme,
        }
