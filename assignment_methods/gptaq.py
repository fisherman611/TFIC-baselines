"""GPTAQ asymmetric-calibration assignment on a fixed quantization grid.

This is a grid-preserving port of Algorithm 1 from GPTAQ (ICML 2025). In
addition to the quantized-path second moment used by GPTQ, GPTAQ consumes the
paired cross moment

    dXXT = E[(X_full_precision - X_quantized)^T X_quantized]

and incorporates its correction matrix ``P`` into GPTQ's lazy block updates.
The grid scale and zero-point are never re-estimated, so the method remains an
assignment baseline that can be applied consistently to Vanilla and AWQ grids.
"""

from __future__ import annotations

import torch

from eigenflip.statistics.trust_region import LayerStats

def stats_from_paired_inputs(
    quantized_inputs: torch.Tensor,
    full_precision_inputs: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> LayerStats:
    """Build exact GPTAQ statistics from paired layer inputs."""

    if quantized_inputs.shape != full_precision_inputs.shape:
        raise ValueError(
            "paired GPTAQ inputs must have identical shapes, got "
            f"{tuple(quantized_inputs.shape)} and "
            f"{tuple(full_precision_inputs.shape)}"
        )
    if quantized_inputs.dim() < 2:
        raise ValueError("GPTAQ inputs must have at least two dimensions")

    quantized = quantized_inputs.reshape(-1, quantized_inputs.shape[-1]).double()
    reference = full_precision_inputs.reshape(
        -1, full_precision_inputs.shape[-1]
    ).double()
    if quantized.shape[0] == 0:
        raise ValueError("GPTAQ inputs must contain at least one sample")

    count = quantized.shape[0]
    mean = quantized.mean(dim=0)
    second_moment = quantized.t().matmul(quantized) / count
    covariance = second_moment - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.t())
    diagonal = torch.diagonal(second_moment).clone()
    diagonal_covariance = torch.diagonal(covariance).clamp_min(0).clone()
    delta_cross = (reference - quantized).t().matmul(quantized) / count

    return LayerStats(
        d=quantized.shape[1],
        mu_hat=mean,
        diag_H=diagonal,
        diag_Sigma=diagonal_covariance,
        eps=eps,
        Sigma=covariance,
        delta_cross=delta_cross,
        backend="gptaq_paired_inputs",
    ).build()


class GPTAQAssignment:
    """GPTAQ lazy block assignment using paired asymmetric statistics."""

    name = "gptaq"

    def __init__(
        self,
        *,
        damp: float = 0.01,
        block_size: int = 128,
        alpha: float = 1.0,
        act_order: bool = False,
        work_dtype: torch.dtype = torch.float64,
    ):
        if damp < 0:
            raise ValueError(f"damp must be non-negative, got {damp}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if work_dtype not in {torch.float32, torch.float64}:
            raise ValueError("work_dtype must be torch.float32 or torch.float64")
        self.damp = damp
        self.block_size = block_size
        self.alpha = alpha
        self.act_order = act_order
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply_to_grid(
        self,
        grid,
        stats: LayerStats,
    ) -> tuple[torch.Tensor, dict]:
        if stats.Sigma is None:
            raise ValueError("GPTAQ requires a materialized Sigma")
        if stats.delta_cross is None:
            raise ValueError(
                "GPTAQ requires paired delta_cross statistics from "
                "full-precision and quantized layer inputs"
            )
        if stats.d != grid.in_features:
            raise ValueError(
                f"stats dimension {stats.d} does not match grid input "
                f"dimension {grid.in_features}"
            )

        device = grid.scale.device
        dtype = self.work_dtype
        d = stats.d
        padded_in = grid.padded_in_features
        weights = grid.float_weights.to(device=device, dtype=dtype).clone()
        scale = grid.scale.to(device=device, dtype=dtype)
        zero_point = grid.zero_point.to(device=device, dtype=dtype)

        mean = stats.mu_hat.to(device=device, dtype=dtype)
        hessian = stats.Sigma.to(device=device, dtype=dtype) + torch.outer(
            mean, mean
        )
        delta_cross = stats.delta_cross.to(device=device, dtype=dtype).clone()
        if delta_cross.shape != (d, d):
            raise ValueError(
                "delta_cross must have shape "
                f"({d}, {d}), got {tuple(delta_cross.shape)}"
            )

        if padded_in > d:
            padded_hessian = torch.zeros(
                padded_in, padded_in, device=device, dtype=dtype
            )
            padded_hessian[:d, :d] = hessian
            padded_idx = torch.arange(d, padded_in, device=device)
            padded_hessian[padded_idx, padded_idx] = torch.diagonal(
                hessian
            ).mean()
            hessian = padded_hessian
            padded_cross = torch.zeros_like(hessian)
            padded_cross[:d, :d] = delta_cross
            delta_cross = padded_cross

        dead = torch.diagonal(hessian) == 0
        hessian[dead, dead] = 1
        weights[:, dead] = 0
        delta_cross[:, dead] = 0

        inverse_permutation = None
        if self.act_order:
            permutation = torch.argsort(torch.diagonal(hessian), descending=True)
            inverse_permutation = torch.argsort(permutation)
            weights = weights[:, permutation]
            scale = scale[:, permutation]
            zero_point = zero_point[:, permutation]
            hessian = hessian[permutation][:, permutation]
            delta_cross = delta_cross[permutation][:, permutation]

        diagonal_mean = torch.diagonal(hessian).mean()
        diagonal_idx = torch.arange(padded_in, device=device)
        hessian[diagonal_idx, diagonal_idx] += self.damp * diagonal_mean

        # Official GPTAQ construction: U is the upper Cholesky factor of H^-1.
        hessian_cholesky = torch.linalg.cholesky(hessian)
        inverse_hessian = torch.cholesky_inverse(hessian_cholesky)
        inverse_factor = torch.linalg.cholesky(inverse_hessian, upper=True)
        correction = self.alpha * torch.triu(
            delta_cross @ inverse_factor.t(), diagonal=1
        ) @ inverse_factor

        codes = torch.zeros_like(weights)
        for start in range(0, padded_in, self.block_size):
            end = min(start + self.block_size, padded_in)
            count = end - start
            block_weights = weights[:, start:end].clone()
            block_codes = torch.zeros_like(block_weights)
            block_errors = torch.zeros_like(block_weights)
            block_factor = inverse_factor[start:end, start:end]
            block_correction = correction[start:end, start:end]

            for offset in range(count):
                column = block_weights[:, offset].clone()
                factor_diagonal = block_factor[offset, offset]
                column_scale = scale[:, start + offset]
                column_zero = zero_point[:, start + offset]
                column_codes = torch.round(
                    column / column_scale + column_zero
                ).clamp(grid.qmin, grid.qmax)
                quantized_column = (column_codes - column_zero) * column_scale
                block_codes[:, offset] = column_codes

                error = (column - quantized_column) / factor_diagonal
                block_weights[:, offset:] -= error.unsqueeze(1) * block_factor[
                    offset, offset:
                ].unsqueeze(0)
                block_weights[:, offset:] += column.unsqueeze(1) * block_correction[
                    offset, offset:
                ].unsqueeze(0)
                block_errors[:, offset] = error

            codes[:, start:end] = block_codes
            if end < padded_in:
                weights[:, end:] -= block_errors @ inverse_factor[start:end, end:]
                weights[:, end:] += block_weights @ correction[start:end, end:]

        if inverse_permutation is not None:
            codes = codes[:, inverse_permutation]

        output = grid.dequantize(codes).to(grid.original_dtype)
        rtn_codes = grid.quantize().to(codes)
        changed_from_rtn = int(
            (
                codes[:, : grid.in_features]
                != rtn_codes[:, : grid.in_features]
            )
            .sum()
            .item()
        )
        info = {
            "assignment": self.name,
            "grid_scheme": grid.scheme,
            "codes": codes,
            "damp": self.damp,
            "block_size": self.block_size,
            "alpha": self.alpha,
            "act_order": self.act_order,
            "changed_codes": changed_from_rtn,
        }
        return output, info
