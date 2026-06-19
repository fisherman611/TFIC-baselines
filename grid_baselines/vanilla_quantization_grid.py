"""Vanilla quantization grid baseline.

The vanilla grid is a uniform quantization grid built directly from the original
weight tensor. This module supports both common variants:

* symmetric absmax: zero-point is fixed at 0 and signed integer codes are used.
* asymmetric min-max: scale and zero-point are computed from each group range.

Both variants fix the grid before assignment methods choose integer codes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VanillaQuantizationGrid:
    """Group-wise uniform grid for a weight tensor."""

    float_weights: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    bits: int
    group_size: int
    in_features: int
    padded_in_features: int
    original_dtype: torch.dtype
    scheme: str

    @torch.no_grad()
    def quantize(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Assign weights to nearest integer codes on this fixed grid."""
        source = self.float_weights if weights is None else weights
        if source.shape[-1] != self.padded_in_features:
            source = self._pad_like_grid(source)
        codes = torch.round(source / self.scale + self.zero_point)
        return codes.clamp(self.qmin, self.qmax)

    @torch.no_grad()
    def dequantize(self, integer_weights: torch.Tensor) -> torch.Tensor:
        """Reconstruct floating-point weights from integer grid codes."""
        weights = (integer_weights - self.zero_point) * self.scale
        if self.padded_in_features > self.in_features:
            weights = weights[:, : self.in_features]
        return weights.to(self.original_dtype)

    @torch.no_grad()
    def round_to_nearest(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RTN integer codes and their dequantized weights."""
        integer_weights = self.quantize()
        return integer_weights, self.dequantize(integer_weights)

    @torch.no_grad()
    def _pad_like_grid(self, weights: torch.Tensor) -> torch.Tensor:
        rows, in_features = weights.shape
        if in_features != self.in_features:
            raise ValueError(
                f"expected {self.in_features} input features, got {in_features}"
            )
        if self.padded_in_features == self.in_features:
            return weights
        padded = torch.zeros(
            rows,
            self.padded_in_features,
            device=weights.device,
            dtype=weights.dtype,
        )
        padded[:, : self.in_features] = weights
        return padded


@torch.no_grad()
def _expand_group(group_values: torch.Tensor, group_size: int, padded_in: int) -> torch.Tensor:
    """Expand [rows, groups, 1] group parameters to [rows, padded_in]."""
    rows, _n_groups, _ = group_values.shape
    return group_values.repeat(1, 1, group_size).reshape(rows, padded_in)


@torch.no_grad()
def build_vanilla_quantization_grid(
    weights: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    scheme: str = "symmetric",
    eps: float = 1e-8,
) -> VanillaQuantizationGrid:
    """Build a group-wise vanilla grid for ``weights``.

    ``scheme="symmetric"`` uses absmax scaling with zero-point 0:
        scale = max(abs(W)) / (2 ** (bits - 1) - 1)
        q in [-2 ** (bits - 1), 2 ** (bits - 1) - 1]

    ``scheme="asymmetric"`` mirrors ``IntegerQuantizedTensorState.from_rtn``:
        scale = (wmax - wmin) / (2 ** bits - 1)
        zero_point = round(-wmin / scale)
        q in [0, 2 ** bits - 1]
    """
    if weights.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(weights.shape)}")
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if scheme not in {"symmetric", "asymmetric"}:
        raise ValueError(f"scheme must be 'symmetric' or 'asymmetric', got {scheme!r}")

    rows, in_features = weights.shape
    device, dtype = weights.device, weights.dtype
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size

    if padded_in > in_features:
        padded = torch.zeros(rows, padded_in, device=device, dtype=dtype)
        padded[:, :in_features] = weights
    else:
        padded = weights

    grouped = padded.reshape(rows, n_groups, group_size)
    if scheme == "symmetric":
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        if qmax <= 0:
            raise ValueError("symmetric quantization requires bits >= 2")
        absmax = grouped.abs().amax(dim=2, keepdim=True)
        scale_group = (absmax / qmax).clamp_min(eps)
        zero_point_group = torch.zeros_like(scale_group)
    else:
        wmin = grouped.min(dim=2, keepdim=True)[0]
        wmax = grouped.max(dim=2, keepdim=True)[0]
        qmin = 0
        qmax = 2**bits - 1
        scale_group = ((wmax - wmin) / qmax).clamp_min(eps)
        zero_point_group = torch.round(-wmin / scale_group).clamp(qmin, qmax)

    return VanillaQuantizationGrid(
        float_weights=padded,
        scale=_expand_group(scale_group, group_size, padded_in),
        zero_point=_expand_group(zero_point_group, group_size, padded_in),
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        padded_in_features=padded_in,
        original_dtype=dtype,
        scheme=scheme,
    )


@torch.no_grad()
def build_symmetric_vanilla_quantization_grid(
    weights: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    eps: float = 1e-8,
) -> VanillaQuantizationGrid:
    """Build a symmetric absmax vanilla grid."""
    return build_vanilla_quantization_grid(
        weights,
        bits,
        group_size,
        scheme="symmetric",
        eps=eps,
    )


@torch.no_grad()
def build_asymmetric_vanilla_quantization_grid(
    weights: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    eps: float = 1e-8,
) -> VanillaQuantizationGrid:
    """Build an asymmetric min-max vanilla grid."""
    return build_vanilla_quantization_grid(
        weights,
        bits,
        group_size,
        scheme="asymmetric",
        eps=eps,
    )
