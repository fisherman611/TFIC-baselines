"""AWQ quantization grid baseline.

AWQ rescales input channels before quantization, applies group-wise uniform
quantization to the scaled weights, then folds the inverse AWQ scale into the
effective dequantization scale. This module supports symmetric and asymmetric
uniform quantization on top of the AWQ-scaled weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AWQQuantizationGrid:
    """AWQ-scaled group-wise uniform grid for a weight tensor."""

    float_weights: torch.Tensor
    scaled_weights: torch.Tensor
    awq_scales: torch.Tensor
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
    clip_max: torch.Tensor | None = None

    @torch.no_grad()
    def quantize(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Assign weights to nearest integer codes on the AWQ-scaled grid."""
        source = self.float_weights if weights is None else weights
        if source.dim() != 2:
            raise ValueError(f"expected 2D weights, got shape {tuple(source.shape)}")
        if source.shape[0] != self.float_weights.shape[0]:
            raise ValueError(
                f"expected {self.float_weights.shape[0]} rows, got {source.shape[0]}"
            )
        if source.shape[-1] != self.padded_in_features:
            source = self._pad_like_grid(source)
        scaled = source * self.awq_scales
        if self.clip_max is not None:
            grouped = scaled.reshape(
                scaled.shape[0], -1, self.group_size
            ).clamp(-self.clip_max, self.clip_max)
            scaled = grouped.reshape_as(scaled)
        scale_q = self.scale * self.awq_scales
        codes = torch.round(scaled / scale_q + self.zero_point)
        return codes.clamp(self.qmin, self.qmax)

    @torch.no_grad()
    def dequantize(self, integer_weights: torch.Tensor) -> torch.Tensor:
        """Reconstruct unscaled weights from integer AWQ grid codes."""
        expected = (self.float_weights.shape[0], self.padded_in_features)
        if tuple(integer_weights.shape) != expected:
            raise ValueError(
                f"expected integer weights with shape {expected}, "
                f"got {tuple(integer_weights.shape)}"
            )
        weights = (integer_weights - self.zero_point) * self.scale
        if self.padded_in_features > self.in_features:
            weights = weights[:, : self.in_features]
        return weights.to(self.original_dtype)

    @torch.no_grad()
    def round_to_nearest(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return AWQ RTN integer codes and their unscaled dequantized weights."""
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


def _minimum_range(reference: torch.Tensor, eps: float) -> torch.Tensor:
    floor = torch.as_tensor(eps, device=reference.device, dtype=reference.dtype)
    if floor == 0:
        floor = torch.nextafter(
            torch.zeros((), device=reference.device, dtype=reference.dtype),
            torch.ones((), device=reference.device, dtype=reference.dtype),
        )
    return floor


@torch.no_grad()
def build_awq_quantization_grid(
    weights: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    scheme: str = "asymmetric",
    eps: float = 1e-5,
    clip_max: torch.Tensor | None = None,
) -> AWQQuantizationGrid:
    """Build an AWQ-scaled grid for ``weights``.

    ``scheme="asymmetric"`` mirrors ``IntegerQuantizedTensorState.from_awq``:

    ```text
    W_scaled = W * awq_scales
    q = round(W_scaled / scale_q + zero_point)
    W_hat = (q - zero_point) * scale_q / awq_scales
    ```

    ``scheme="symmetric"`` uses absmax scaling on ``W_scaled`` and zero-point 0.
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
        raise ValueError("symmetric AWQ quantization requires bits >= 2")

    rows, in_features = weights.shape
    device, dtype = weights.device, weights.dtype
    scales = awq_scales.to(device=device, dtype=dtype).reshape(1, -1)
    if scales.shape[1] != in_features:
        raise ValueError(
            f"expected awq_scales with {in_features} values, got {scales.shape[1]}"
        )
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise ValueError("awq_scales values must be finite and positive")
    if not torch.isfinite(weights).all():
        raise ValueError("weights must contain only finite values")

    scaled_weights = weights * scales
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

    grouped_clip = None
    if clip_max is not None:
        grouped_clip = torch.as_tensor(clip_max, device=device, dtype=dtype)
        if grouped_clip.shape == (rows, n_groups):
            grouped_clip = grouped_clip.unsqueeze(-1)
        if grouped_clip.shape != (rows, n_groups, 1):
            raise ValueError(
                "clip_max must have shape "
                f"({rows}, {n_groups}, 1), got {tuple(grouped_clip.shape)}"
            )
        if not torch.isfinite(grouped_clip).all() or torch.any(grouped_clip <= 0):
            raise ValueError("clip_max values must be finite and positive")

    grouped = padded_scaled.reshape(rows, n_groups, group_size)
    if grouped_clip is not None:
        grouped = grouped.clamp(-grouped_clip, grouped_clip)
        padded_scaled = grouped.reshape(rows, padded_in)

    floor = _minimum_range(grouped, eps)
    if scheme == "symmetric":
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        absmax = grouped.abs().amax(dim=2, keepdim=True)
        scale_q_group = absmax.clamp_min(floor) / qmax
        zero_point_group = torch.zeros_like(scale_q_group)
    else:
        wmin = grouped.min(dim=2, keepdim=True)[0]
        wmax = grouped.max(dim=2, keepdim=True)[0]
        qmin = 0
        qmax = 2**bits - 1
        scale_q_group = (wmax - wmin).clamp_min(floor) / qmax
        zero_point_group = torch.round(-wmin / scale_q_group).clamp(qmin, qmax)

    scale_q = _expand_group(scale_q_group, group_size, padded_in)
    effective_scale = scale_q / padded_scales

    return AWQQuantizationGrid(
        float_weights=padded_weights,
        scaled_weights=padded_scaled,
        awq_scales=padded_scales,
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
        clip_max=grouped_clip,
    )


@torch.no_grad()
def build_symmetric_awq_quantization_grid(
    weights: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    eps: float = 1e-5,
    clip_max: torch.Tensor | None = None,
) -> AWQQuantizationGrid:
    """Build a symmetric AWQ-scaled grid."""
    return build_awq_quantization_grid(
        weights,
        awq_scales,
        bits,
        group_size,
        scheme="symmetric",
        eps=eps,
        clip_max=clip_max,
    )


@torch.no_grad()
def build_asymmetric_awq_quantization_grid(
    weights: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    eps: float = 1e-5,
    clip_max: torch.Tensor | None = None,
) -> AWQQuantizationGrid:
    """Build an asymmetric AWQ-scaled grid."""
    return build_awq_quantization_grid(
        weights,
        awq_scales,
        bits,
        group_size,
        scheme="asymmetric",
        eps=eps,
        clip_max=clip_max,
    )
