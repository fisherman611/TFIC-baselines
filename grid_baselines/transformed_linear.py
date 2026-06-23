"""Transform-aware linear layers used by model-level grid baselines.

The stored weight lives in transformed activation coordinates.  The matching
input transform is applied online, and ``assignment_input`` exposes that exact
coordinate system to the shared statistics collector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quantize_activation(
    values: torch.Tensor,
    *,
    bits: int,
    symmetric: bool,
    group_size: int = -1,
    clip_ratio: float = 1.0,
    clip_factor_max: torch.Tensor | None = None,
    clip_factor_min: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-token fake quantization used by FlatQuant and SpinQuant."""

    if bits >= 16:
        return values
    if not (0 < clip_ratio <= 1):
        raise ValueError("clip_ratio must be in (0, 1]")
    width = values.shape[-1]
    actual_group = width if group_size <= 0 else group_size
    padded_width = ((width + actual_group - 1) // actual_group) * actual_group
    if padded_width != width:
        raise ValueError(
            "activation width must be divisible by group_size, matching the "
            "official FlatQuant/SpinQuant quantizers"
        )
    padded = values
    grouped = padded.reshape(*padded.shape[:-1], -1, actual_group)
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        if clip_factor_max is not None or clip_factor_min is not None:
            if clip_factor_max is None or clip_factor_min is None:
                raise ValueError("activation clipping requires max and min factors")
            zeros = torch.zeros_like(grouped[..., :1])
            lower = torch.minimum(grouped.amin(dim=-1, keepdim=True), zeros)
            upper = torch.maximum(grouped.amax(dim=-1, keepdim=True), zeros)
            lower = lower * clip_factor_min.to(lower)
            upper = upper * clip_factor_max.to(upper)
            bound = torch.maximum(lower.abs(), upper)
        else:
            bound = grouped.abs().amax(dim=-1, keepdim=True) * clip_ratio
        scale = (bound / qmax).clamp_min(eps)
        quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
    else:
        qmin = 0
        qmax = 2**bits - 1
        zeros = torch.zeros_like(grouped[..., :1])
        lower = torch.minimum(grouped.amin(dim=-1, keepdim=True), zeros)
        upper = torch.maximum(grouped.amax(dim=-1, keepdim=True), zeros)
        if clip_factor_max is not None or clip_factor_min is not None:
            if clip_factor_max is None or clip_factor_min is None:
                raise ValueError("activation clipping requires max and min factors")
            upper = upper * clip_factor_max.to(upper)
            lower = lower * clip_factor_min.to(lower)
        else:
            lower = lower * clip_ratio
            upper = upper * clip_ratio
        scale = ((upper - lower) / qmax).clamp_min(eps)
        zero = torch.round(-lower / scale).clamp(qmin, qmax)
        codes = torch.round(grouped / scale + zero).clamp(qmin, qmax)
        quantized = (codes - zero) * scale
    return quantized.reshape(*padded.shape).to(values.dtype)


def _apply_kronecker(
    values: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    initial_shape = values.shape
    if initial_shape[-1] != left.shape[0] * right.shape[0]:
        raise ValueError(
            "Kronecker transform dimension mismatch: "
            f"{initial_shape[-1]} != {left.shape[0]} * {right.shape[0]}"
        )
    work = values.reshape(-1, left.shape[0], right.shape[0])
    work = torch.matmul(work, right)
    work = torch.matmul(left.t(), work)
    return work.reshape(initial_shape)


@dataclass
class KroneckerTransform:
    """FlatQuant transform ``P = P_left ⊗ P_right`` plus optional ``diag(c)``."""

    left: torch.Tensor
    right: torch.Tensor
    diagonal: torch.Tensor | None = None

    def validate(self, width: int) -> None:
        if self.left.dim() != 2 or self.left.shape[0] != self.left.shape[1]:
            raise ValueError("left FlatQuant factor must be square")
        if self.right.dim() != 2 or self.right.shape[0] != self.right.shape[1]:
            raise ValueError("right FlatQuant factor must be square")
        if self.left.shape[0] * self.right.shape[0] != width:
            raise ValueError(
                f"FlatQuant factors represent width "
                f"{self.left.shape[0] * self.right.shape[0]}, expected {width}"
            )
        if self.diagonal is not None and self.diagonal.numel() != width:
            raise ValueError(
                f"FlatQuant diagonal has {self.diagonal.numel()} values, "
                f"expected {width}"
            )

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        self.validate(values.shape[-1])
        diagonal = self.diagonal
        if diagonal is not None:
            values = values * diagonal.to(device=values.device, dtype=values.dtype)
        return _apply_kronecker(
            values,
            self.left.to(device=values.device, dtype=values.dtype),
            self.right.to(device=values.device, dtype=values.dtype),
        )

    def inverse_transpose_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Return ``W P^{-T}`` for use with the online activation ``X P``."""

        self.validate(weight.shape[-1])
        work_dtype = torch.float64
        left = self.left.to(device=weight.device, dtype=work_dtype)
        right = self.right.to(device=weight.device, dtype=work_dtype)
        left_inv_t = torch.linalg.inv(left).t()
        right_inv_t = torch.linalg.inv(right).t()
        transformed = weight.to(work_dtype)
        if self.diagonal is not None:
            transformed = transformed / self.diagonal.to(
                device=weight.device,
                dtype=work_dtype,
            )
        transformed = _apply_kronecker(
            transformed,
            left_inv_t,
            right_inv_t,
        )
        return transformed.to(weight.dtype)

    def state_dict(self) -> dict[str, torch.Tensor]:
        result = {
            "left": self.left.detach().cpu(),
            "right": self.right.detach().cpu(),
        }
        if self.diagonal is not None:
            result["diagonal"] = self.diagonal.detach().cpu()
        return result

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "KroneckerTransform":
        return cls(
            left=torch.as_tensor(state["left"]),
            right=torch.as_tensor(state["right"]),
            diagonal=(
                torch.as_tensor(state["diagonal"])
                if "diagonal" in state
                else None
            ),
        )


class TransformAwareLinear(nn.Linear):
    """Linear layer whose weight is expressed in transformed input coordinates."""

    def __init__(
        self,
        source: nn.Linear,
        transform: KroneckerTransform,
        *,
        weight_transform: KroneckerTransform | None = None,
        output_head_transform: torch.Tensor | None = None,
        weight_is_transformed: bool = False,
        activation_bits: int = 16,
        activation_symmetric: bool = False,
        activation_group_size: int = -1,
        activation_clip_ratio: float = 1.0,
        activation_clip_max: torch.Tensor | None = None,
        activation_clip_min: torch.Tensor | None = None,
    ):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.input_transform = transform
        self.weight_transform = weight_transform or transform
        self.output_head_transform = output_head_transform
        self.activation_bits = activation_bits
        self.activation_symmetric = activation_symmetric
        self.activation_group_size = activation_group_size
        self.activation_clip_ratio = activation_clip_ratio
        self.activation_clip_max = activation_clip_max
        self.activation_clip_min = activation_clip_min
        transformed_weight = (
            source.weight.detach()
            if weight_is_transformed
            else self.weight_transform.inverse_transpose_weight(source.weight.detach())
        )
        if output_head_transform is not None and not weight_is_transformed:
            matrix = output_head_transform.to(
                device=transformed_weight.device,
                dtype=torch.float64,
            )
            head_dim = matrix.shape[0]
            grouped = transformed_weight.to(torch.float64).reshape(
                -1,
                head_dim,
                transformed_weight.shape[1],
            )
            transformed_weight = (
                matrix.t().unsqueeze(0).matmul(grouped)
            ).reshape_as(transformed_weight).to(source.weight.dtype)
        self.weight.data.copy_(transformed_weight)
        if source.bias is not None:
            bias = source.bias.detach()
            if output_head_transform is not None and not weight_is_transformed:
                matrix = output_head_transform.to(
                    device=bias.device,
                    dtype=torch.float64,
                )
                grouped = bias.to(torch.float64).reshape(-1, matrix.shape[0], 1)
                bias = (
                    matrix.t().unsqueeze(0).matmul(grouped)
                ).reshape_as(bias).to(source.bias.dtype)
            self.bias.data.copy_(bias)

    def transformed_input(self, values: torch.Tensor) -> torch.Tensor:
        return self.input_transform.apply(values)

    def assignment_input(self, values: torch.Tensor) -> torch.Tensor:
        transformed = self.transformed_input(values)
        return fake_quantize_activation(
            transformed,
            bits=self.activation_bits,
            symmetric=self.activation_symmetric,
            group_size=self.activation_group_size,
            clip_ratio=self.activation_clip_ratio,
            clip_factor_max=self.activation_clip_max,
            clip_factor_min=self.activation_clip_min,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = self.assignment_input(values)
        return F.linear(transformed, self.weight, self.bias)


def apply_factorized_hadamard(
    values: torch.Tensor,
    *,
    had_k: torch.Tensor | None,
    k: int,
) -> torch.Tensor:
    """Apply SpinQuant/QuaRot's normalized factorized Hadamard transform."""

    width = values.shape[-1]
    if k <= 0 or width % k != 0 or (width // k) & (width // k - 1):
        raise ValueError("Hadamard width / K must be a positive power of two")
    if k > 1:
        if had_k is None or tuple(had_k.shape) != (k, k):
            raise ValueError(f"had_k must have shape ({k}, {k})")
    work = values.reshape(-1, width, 1)
    while work.shape[1] > k:
        work = work.reshape(work.shape[0], work.shape[1] // 2, 2, -1)
        first = work[:, :, 0, :]
        second = work[:, :, 1, :]
        work = torch.stack((first + second, first - second), dim=2)
        work = work.reshape(work.shape[0], work.shape[1], -1)
    if k > 1:
        work = torch.matmul(
            had_k.to(device=work.device, dtype=work.dtype).unsqueeze(0),
            work,
        )
    return (work.reshape_as(values) / (width**0.5)).to(values.dtype)


class SpinQuantHadamardLinear(nn.Linear):
    """SpinQuant R4 down projection with its matching online Hadamard."""

    def __init__(
        self,
        source: nn.Linear,
        *,
        had_k: torch.Tensor | None,
        k: int,
        weight_is_transformed: bool = False,
    ):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.k = int(k)
        self.register_buffer(
            "had_k",
            None if had_k is None else torch.as_tensor(had_k).detach().clone(),
            persistent=True,
        )
        self.activation_bits = 16
        self.activation_symmetric = False
        self.activation_group_size = -1
        self.activation_clip_ratio = 1.0
        weight = source.weight.detach()
        if not weight_is_transformed:
            weight = apply_factorized_hadamard(
                weight,
                had_k=self.had_k,
                k=self.k,
            )
        self.weight.data.copy_(weight)
        if source.bias is not None:
            self.bias.data.copy_(source.bias.detach())

    def transformed_input(self, values: torch.Tensor) -> torch.Tensor:
        return apply_factorized_hadamard(values, had_k=self.had_k, k=self.k)

    def assignment_input(self, values: torch.Tensor) -> torch.Tensor:
        values = self.transformed_input(values)
        return fake_quantize_activation(
            values,
            bits=self.activation_bits,
            symmetric=self.activation_symmetric,
            group_size=self.activation_group_size,
            clip_ratio=self.activation_clip_ratio,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.assignment_input(values)
        return F.linear(values, self.weight, self.bias)


class ActivationQuantizedLinear(nn.Linear):
    """Drop-in linear with per-token fake activation quantization."""

    def __init__(
        self,
        source: nn.Linear,
        *,
        bits: int,
        symmetric: bool,
        group_size: int = -1,
        clip_ratio: float = 1.0,
        output_bits: int = 16,
        output_symmetric: bool = False,
        output_group_size: int = -1,
        output_clip_ratio: float = 1.0,
    ):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.weight.data.copy_(source.weight.detach())
        if source.bias is not None:
            self.bias.data.copy_(source.bias.detach())
        self.activation_bits = bits
        self.activation_symmetric = symmetric
        self.activation_group_size = group_size
        self.activation_clip_ratio = clip_ratio
        self.output_bits = output_bits
        self.output_symmetric = output_symmetric
        self.output_group_size = output_group_size
        self.output_clip_ratio = output_clip_ratio

    def assignment_input(self, values: torch.Tensor) -> torch.Tensor:
        return self.quantized_assignment_input(values)

    def quantized_assignment_input(self, values: torch.Tensor) -> torch.Tensor:
        """Actual tensor consumed by the weight during forward."""

        return fake_quantize_activation(
            values,
            bits=self.activation_bits,
            symmetric=self.activation_symmetric,
            group_size=self.activation_group_size,
            clip_ratio=self.activation_clip_ratio,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        quantized = self.quantized_assignment_input(values)
        output = F.linear(quantized, self.weight, self.bias)
        return fake_quantize_activation(
            output,
            bits=self.output_bits,
            symmetric=self.output_symmetric,
            group_size=self.output_group_size,
            clip_ratio=self.output_clip_ratio,
        )
