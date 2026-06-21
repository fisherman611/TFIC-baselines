"""GPTAQ plus compensation-aware residual correction on a fixed grid."""

from __future__ import annotations

import torch

from eigenflip.statistics.trust_region import LayerStats

class GPTAQResCompAssignment:
    """GPTAQ lazy block assignment with ResComp's compensation-aware error."""

    name = "gptaq_rescomp"

    def __init__(
        self,
        *,
        damp: float = 0.01,
        block_size: int = 128,
        alpha: float = 0.25,
        rescomp_alpha: float = 0.25,
        act_order: bool = False,
        work_dtype: torch.dtype = torch.float64,
    ):
        if damp < 0:
            raise ValueError(f"damp must be non-negative, got {damp}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if rescomp_alpha < 0:
            raise ValueError(
                f"rescomp_alpha must be non-negative, got {rescomp_alpha}"
            )
        if work_dtype not in {torch.float32, torch.float64}:
            raise ValueError("work_dtype must be torch.float32 or torch.float64")
        self.damp = damp
        self.block_size = block_size
        self.alpha = alpha
        self.rescomp_alpha = rescomp_alpha
        self.act_order = act_order
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply_to_grid(
        self,
        grid,
        stats: LayerStats,
    ) -> tuple[torch.Tensor, dict]:
        if stats.Sigma is None:
            raise ValueError("GPTAQ+ResComp requires a materialized Sigma")
        if stats.delta_cross is None:
            raise ValueError(
                "GPTAQ+ResComp requires paired delta_cross statistics from "
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
        original_weights = weights.clone()
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
        original_weights[:, dead] = 0
        delta_cross[:, dead] = 0

        inverse_permutation = None
        if self.act_order:
            permutation = torch.argsort(torch.diagonal(hessian), descending=True)
            inverse_permutation = torch.argsort(permutation)
            weights = weights[:, permutation]
            original_weights = original_weights[:, permutation]
            scale = scale[:, permutation]
            zero_point = zero_point[:, permutation]
            hessian = hessian[permutation][:, permutation]
            delta_cross = delta_cross[permutation][:, permutation]

        diagonal_mean = torch.diagonal(hessian).mean()
        diagonal_idx = torch.arange(padded_in, device=device)
        hessian[diagonal_idx, diagonal_idx] += self.damp * diagonal_mean

        # U is the upper Cholesky factor of H^-1. P is GPTAQ's input-asymmetry
        # correction; R is ResComp's compensation-aware residual correction.
        hessian_cholesky = torch.linalg.cholesky(hessian)
        inverse_hessian = torch.cholesky_inverse(hessian_cholesky)
        inverse_factor = torch.linalg.cholesky(inverse_hessian, upper=True)
        correction = self.alpha * torch.triu(
            delta_cross @ inverse_factor.t(), diagonal=1
        ) @ inverse_factor
        rescomp_correction = self.rescomp_alpha * torch.triu(
            (hessian + delta_cross) @ inverse_factor.t(), diagonal=1
        ) @ inverse_factor

        mode = "org" if grid.bits == 2 else "allw"
        codes = torch.zeros_like(weights)
        for start in range(0, padded_in, self.block_size):
            end = min(start + self.block_size, padded_in)
            count = end - start
            block_weights = weights[:, start:end].clone()
            block_original_weights = original_weights[:, start:end]
            block_scale = scale[:, start:end]
            block_zero_point = zero_point[:, start:end]
            block_lower_bound = (grid.qmin - block_zero_point) * block_scale
            block_upper_bound = (grid.qmax - block_zero_point) * block_scale
            block_codes = torch.zeros_like(block_weights)
            block_errors = torch.zeros_like(block_weights)
            block_factor = inverse_factor[start:end, start:end]
            block_correction = correction[start:end, start:end]
            block_rescomp_correction = rescomp_correction[start:end, start:end]

            for offset in range(count):
                column = block_weights[:, offset].clone()
                original_column = block_original_weights[:, offset]
                factor_diagonal = block_factor[offset, offset]
                column_scale = scale[:, start + offset]
                column_zero = zero_point[:, start + offset]
                column_codes = torch.round(
                    column / column_scale + column_zero
                ).clamp(grid.qmin, grid.qmax)
                quantized_column = (column_codes - column_zero) * column_scale
                block_codes[:, offset] = column_codes

                error = (column - quantized_column) / factor_diagonal
                if mode == "org":
                    update_slice = slice(offset, None)
                else:
                    update_slice = slice(offset + 1, None)
                block_weights[:, update_slice] -= (
                    error.unsqueeze(1)
                    * block_factor[offset, update_slice].unsqueeze(0)
                )
                block_weights[:, update_slice] += (
                    column.unsqueeze(1)
                    * block_correction[offset, update_slice].unsqueeze(0)
                )
                compensation_gap = original_column - column
                block_weights[:, update_slice] += (
                    compensation_gap.unsqueeze(1)
                    * block_rescomp_correction[offset, update_slice].unsqueeze(0)
                )
                block_errors[:, offset] = error

            codes[:, start:end] = block_codes
            if end < padded_in:
                weights[:, end:] -= block_errors @ inverse_factor[start:end, end:]
                if mode == "org":
                    compensated_weights = block_weights
                else:
                    compensated_weights = torch.minimum(
                        torch.maximum(block_weights, block_lower_bound),
                        block_upper_bound,
                    )
                weights[:, end:] += compensated_weights @ correction[start:end, end:]
                weights[:, end:] += (
                    original_weights[:, start:end] - compensated_weights
                ) @ rescomp_correction[start:end, end:]

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
            "rescomp_alpha": self.rescomp_alpha,
            "rescomp_mode": mode,
            "act_order": self.act_order,
            "changed_codes": changed_from_rtn,
        }
        return output, info
