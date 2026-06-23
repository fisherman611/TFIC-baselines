"""FlatQuant diagonal-scale quantization grid baseline.

FlatQuant's full method learns affine activation/weight transformations and
then runs low-bit matmuls in the transformed coordinate.  The non-diagonal
Kronecker affine transform requires an online activation transform in the model
forward path, so it is not representable as the per-coordinate fixed grid used
by this repository's assignment methods.

This module implements only the diagonal-scale part of FlatQuant that is
compatible with this repository's fixed-grid interface: the learned pair-wise
per-channel scaling vector and the learned weight clipping threshold.  It is
not a full FlatQuant implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlatQuantDiagQuantizationGrid:
    """FlatQuant diagonal-scale group-wise uniform grid for a weight tensor."""

    float_weights: torch.Tensor
    scaled_weights: torch.Tensor
    flatquant_scales: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    bits: int
    group_size: int
    in_features: int
    padded_in_features: int
    original_dtype: torch.dtype
    scheme: str = "asymmetric"
    weight_clip: float = 1.0

    @torch.no_grad()
    def quantize(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Assign weights to nearest integer codes on this fixed grid."""

        if weights is None:
            scaled = self.scaled_weights
        else:
            source = weights
            if source.shape[-1] != self.padded_in_features:
                source = self._pad_like_grid(source)
            scaled = source * self.flatquant_scales
        scale_q = self.scale * self.flatquant_scales
        codes = torch.round(scaled / scale_q + self.zero_point)
        return codes.clamp(self.qmin, self.qmax)

    def pre_round_values(self) -> torch.Tensor:
        scale_q = self.scale * self.flatquant_scales
        return self.scaled_weights / scale_q + self.zero_point

    @torch.no_grad()
    def dequantize(self, integer_weights: torch.Tensor) -> torch.Tensor:
        """Reconstruct unscaled weights from integer grid codes."""

        weights = (integer_weights - self.zero_point) * self.scale
        if self.padded_in_features > self.in_features:
            weights = weights[:, : self.in_features]
        return weights.to(self.original_dtype)

    @torch.no_grad()
    def round_to_nearest(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RTN integer codes and their unscaled dequantized weights."""

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
    rows, _n_groups, _ = group_values.shape
    return group_values.repeat(1, 1, group_size).reshape(rows, padded_in)


@torch.no_grad()
def _clip_group_range(
    grouped: torch.Tensor,
    *,
    scheme: str,
    weight_clip: float,
    weight_clip_max: torch.Tensor | None = None,
    weight_clip_min: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    if not (0 < weight_clip <= 1):
        raise ValueError(
            f"weight_clip must be in the interval (0, 1], got {weight_clip}"
        )

    if scheme == "symmetric":
        return grouped.abs().amax(dim=2, keepdim=True) * weight_clip

    wmin = grouped.min(dim=2, keepdim=True)[0]
    wmax = grouped.max(dim=2, keepdim=True)[0]
    if weight_clip_max is not None or weight_clip_min is not None:
        if weight_clip_max is None or weight_clip_min is None:
            raise ValueError("FlatQuant clipping requires both max and min factors")
        max_factor = weight_clip_max.to(grouped).reshape(-1, 1, 1)
        min_factor = weight_clip_min.to(grouped).reshape(-1, 1, 1)
        if max_factor.shape[0] != grouped.shape[0]:
            raise ValueError("FlatQuant weight clip factors must be per output channel")
        return wmin * min_factor, wmax * max_factor
    if weight_clip == 1:
        return wmin, wmax
    midpoint = 0.5 * (wmin + wmax)
    half_range = 0.5 * (wmax - wmin) * weight_clip
    return midpoint - half_range, midpoint + half_range


@torch.no_grad()
def build_flatquant_diag_quantization_grid(
    weights: torch.Tensor,
    flatquant_scales: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    scheme: str = "asymmetric",
    weight_clip: float | torch.Tensor = 1.0,
    weight_clip_max: torch.Tensor | None = None,
    weight_clip_min: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> FlatQuantDiagQuantizationGrid:
    """Build a FlatQuant diagonal-scale fixed grid for ``weights``.

    ``flatquant_scales`` is the learned per-input-channel scaling vector
    corresponding to FlatQuant's pair-wise merged ``diag(c)``.  The grid is
    built from ``W * c`` and stores the effective dequantization scale
    ``scale_q / c`` so assignment methods still return ordinary module weights.

    ``weight_clip`` represents FlatQuant's learned weight clipping threshold
    after sigmoid, i.e. a ratio in ``(0, 1]``.  For symmetric quantization it
    shrinks the absmax range.  For asymmetric quantization it shrinks the
    min-max range around its midpoint.
    """

    if weights.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(weights.shape)}")
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if scheme not in {"asymmetric", "symmetric"}:
        raise ValueError(f"scheme must be 'asymmetric' or 'symmetric', got {scheme!r}")
    if scheme == "symmetric" and bits < 2:
        raise ValueError("symmetric FlatQuant quantization requires bits >= 2")

    rows, in_features = weights.shape
    device, dtype = weights.device, weights.dtype
    scales = flatquant_scales.to(device=device, dtype=dtype).reshape(1, -1)
    if scales.shape[1] != in_features:
        raise ValueError(
            "expected flatquant_scales with "
            f"{in_features} values, got {scales.shape[1]}"
        )
    if torch.any(scales == 0):
        raise ValueError("flatquant_scales must be non-zero")

    clip_value = float(torch.as_tensor(weight_clip).detach().cpu().reshape(-1)[0].item())
    scaled_weights = weights * scales
    if weight_clip_max is not None or weight_clip_min is not None:
        if weight_clip_max is None or weight_clip_min is None:
            raise ValueError("FlatQuant clipping requires both max and min factors")
        max_factor = weight_clip_max.to(device=device, dtype=dtype).reshape(-1, 1)
        min_factor = weight_clip_min.to(device=device, dtype=dtype).reshape(-1, 1)
        if max_factor.shape[0] != rows or min_factor.shape[0] != rows:
            raise ValueError(
                "FlatQuant weight clipping factors must have one value per "
                "output channel"
            )
        row_min = scaled_weights.amin(dim=1, keepdim=True) * min_factor
        row_max = scaled_weights.amax(dim=1, keepdim=True) * max_factor
        scaled_weights = torch.clamp(scaled_weights, min=row_min, max=row_max)
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size
    if padded_in > in_features:
        padded_scaled = torch.zeros(rows, padded_in, device=device, dtype=dtype)
        padded_scaled[:, :in_features] = scaled_weights
        padded_weights = torch.zeros(rows, padded_in, device=device, dtype=dtype)
        padded_weights[:, :in_features] = weights
        padded_scales = torch.ones(1, padded_in, device=device, dtype=dtype)
        padded_scales[:, :in_features] = scales
    else:
        padded_scaled = scaled_weights
        padded_weights = weights
        padded_scales = scales

    grouped = padded_scaled.reshape(rows, n_groups, group_size)
    if scheme == "symmetric":
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        absmax = _clip_group_range(
            grouped,
            scheme=scheme,
            weight_clip=clip_value,
        )
        scale_q_group = (absmax / qmax).clamp_min(eps)
        zero_point_group = torch.zeros_like(scale_q_group)
    else:
        qmin = 0
        qmax = 2**bits - 1
        wmin, wmax = _clip_group_range(
            grouped,
            scheme=scheme,
            weight_clip=clip_value,
        )
        scale_q_group = ((wmax - wmin) / qmax).clamp_min(eps)
        zero_point_group = torch.round(-wmin / scale_q_group).clamp(qmin, qmax)

    scale_q = _expand_group(scale_q_group, group_size, padded_in)
    effective_scale = scale_q / padded_scales

    return FlatQuantDiagQuantizationGrid(
        float_weights=padded_weights,
        scaled_weights=padded_scaled,
        flatquant_scales=padded_scales,
        scale=effective_scale,
        zero_point=_expand_group(zero_point_group, group_size, padded_in),
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        padded_in_features=padded_in,
        original_dtype=dtype,
        scheme=scheme,
        weight_clip=clip_value,
    )


@torch.no_grad()
def build_symmetric_flatquant_diag_quantization_grid(
    weights: torch.Tensor,
    flatquant_scales: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    weight_clip: float | torch.Tensor = 1.0,
    eps: float = 1e-8,
) -> FlatQuantDiagQuantizationGrid:
    """Build a symmetric FlatQuant diagonal-scale fixed grid."""

    return build_flatquant_diag_quantization_grid(
        weights,
        flatquant_scales,
        bits,
        group_size,
        scheme="symmetric",
        weight_clip=weight_clip,
        eps=eps,
    )


@torch.no_grad()
def build_asymmetric_flatquant_diag_quantization_grid(
    weights: torch.Tensor,
    flatquant_scales: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    weight_clip: float | torch.Tensor = 1.0,
    eps: float = 1e-8,
) -> FlatQuantDiagQuantizationGrid:
    """Build an asymmetric FlatQuant diagonal-scale fixed grid."""

    return build_flatquant_diag_quantization_grid(
        weights,
        flatquant_scales,
        bits,
        group_size,
        scheme="asymmetric",
        weight_clip=weight_clip,
        eps=eps,
    )
