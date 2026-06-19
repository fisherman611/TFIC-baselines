"""Adapters from grid-baseline objects to EigenFlip encoder state."""

from __future__ import annotations

import torch

from eigenflip.quantization.state import IntegerQuantizedTensorState
from grid_baselines import VanillaQuantizationGrid


@torch.no_grad()
def state_from_vanilla_grid(
    grid: VanillaQuantizationGrid,
    *,
    non_negative_codes: bool = False,
) -> IntegerQuantizedTensorState:
    """Convert a vanilla grid to the state consumed by assignment encoders.

    Some legacy encoders assume code updates live in ``[0, max_int]``. For a
    signed symmetric grid, ``non_negative_codes=True`` shifts both integer codes
    and zero-points by ``-qmin``. The represented dequantized values are
    unchanged because ``(q + shift) - (z + shift) == q - z``.
    """
    integer_weights = grid.quantize()
    pre_round = grid.float_weights / grid.scale + grid.zero_point
    zero_point = grid.zero_point
    min_int = grid.qmin
    max_int = grid.qmax
    if non_negative_codes and grid.qmin < 0:
        shift = -grid.qmin
        integer_weights = integer_weights + shift
        pre_round = pre_round + shift
        zero_point = zero_point + shift
        min_int = 0
        max_int = grid.qmax + shift
    return IntegerQuantizedTensorState(
        float_weights=grid.float_weights,
        pre_round=pre_round,
        integer_weights=integer_weights,
        scale=grid.scale,
        zero_point=zero_point,
        max_int=max_int,
        min_int=min_int,
        in_features=grid.in_features,
        padded_in_features=grid.padded_in_features,
        original_dtype=grid.original_dtype,
        group_size=grid.group_size,
    )
