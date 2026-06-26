"""Qronus assignment on a fixed quantization grid.

This implements the efficient Qronus layer-wise assignment rule from the
paper's Section 3.1 using only square calibration moments:

    H = X_tilde^T X_tilde
    G = X_tilde^T X

The repository's paired activation accumulator already stores
``delta_cross = (X - X_tilde)^T X_tilde``, so ``G`` is recovered as
``H + delta_cross.T``.  The grid scale and zero-point are fixed; Qronus only
chooses integer codes.
"""

from __future__ import annotations

import torch

from eigenflip.statistics.trust_region import LayerStats


class QronusAssignment:
    """Qronus mismatched-input assignment on a fixed quantization grid."""

    name = "qronus"

    def __init__(
        self,
        *,
        alpha: float = 1e-6,
        act_order: bool = True,
        work_dtype: torch.dtype = torch.float64,
        damp: float | None = None,
    ):
        if damp is not None:
            alpha = damp
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if work_dtype not in {torch.float32, torch.float64}:
            raise ValueError("work_dtype must be torch.float32 or torch.float64")
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
            raise ValueError("Qronus requires a materialized Sigma")
        if stats.delta_cross is None:
            raise ValueError(
                "Qronus requires paired delta_cross statistics from "
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

        diagonal = torch.diagonal(hessian).clone()
        dead = diagonal == 0
        hessian[dead, dead] = 1
        weights[:, dead] = 0
        delta_cross[dead, :] = 0
        delta_cross[:, dead] = 0

        # G = X_tilde^T X. The paired accumulator stores
        # (X - X_tilde)^T X_tilde, so transpose it before adding to H.
        cross_hessian = hessian + delta_cross.t()

        diagonal_idx = torch.arange(padded_in, device=device)
        if self.alpha > 0:
            spectral_norm = torch.linalg.eigvalsh(hessian).amax().clamp_min(0)
            hessian[diagonal_idx, diagonal_idx] += self.alpha * spectral_norm

        inverse_permutation = None
        if self.act_order:
            permutation = torch.argsort(torch.diagonal(hessian), descending=True)
            inverse_permutation = torch.argsort(permutation)
            weights = weights[:, permutation]
            scale = scale[:, permutation]
            zero_point = zero_point[:, permutation]
            hessian = hessian[permutation][:, permutation]
            cross_hessian = cross_hessian[permutation][:, permutation]

        codes = torch.empty_like(weights)
        qmin = float(grid.qmin)
        qmax = float(grid.qmax)

        if padded_in == 0:
            output = grid.dequantize(codes).to(grid.original_dtype)
            return output, self._info(grid, codes)

        if padded_in == 1:
            first_values = (weights @ cross_hessian[0]) / hessian[0, 0]
            codes[:, 0] = self._quantize_column(
                first_values,
                scale[:, 0],
                zero_point[:, 0],
                qmin,
                qmax,
            )
        else:
            first_values = (
                weights @ cross_hessian[0]
                - weights[:, 1:] @ hessian[0, 1:]
            ) / hessian[0, 0]
            codes[:, 0] = self._quantize_column(
                first_values,
                scale[:, 0],
                zero_point[:, 0],
                qmin,
                qmax,
            )
            first_dequantized = (codes[:, 0] - zero_point[:, 0]) * scale[:, 0]
            rhs = weights @ cross_hessian[1:].t()
            rhs -= first_dequantized.unsqueeze(1) * hessian[1:, 0].unsqueeze(0)
            weights[:, 1:] = torch.linalg.solve(hessian[1:, 1:], rhs.t()).t()

        if padded_in > 1:
            inverse_hessian = torch.linalg.inv(hessian)
            lower_factor = torch.linalg.cholesky(inverse_hessian, upper=False)
            for index in range(1, padded_in):
                column = weights[:, index].clone()
                codes[:, index] = self._quantize_column(
                    column,
                    scale[:, index],
                    zero_point[:, index],
                    qmin,
                    qmax,
                )
                if index + 1 < padded_in:
                    dequantized = (codes[:, index] - zero_point[:, index]) * scale[
                        :, index
                    ]
                    error = column - dequantized
                    factor = (
                        lower_factor[index + 1 :, index] / lower_factor[index, index]
                    )
                    weights[:, index + 1 :] -= error.unsqueeze(1) * factor.unsqueeze(0)

        if inverse_permutation is not None:
            codes = codes[:, inverse_permutation]

        output = grid.dequantize(codes).to(grid.original_dtype)
        info = self._info(grid, codes)
        return output, info

    @staticmethod
    def _quantize_column(
        values: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qmin: float,
        qmax: float,
    ) -> torch.Tensor:
        return torch.round(values / scale + zero_point).clamp(qmin, qmax)

    def _info(self, grid, codes: torch.Tensor) -> dict:
        rtn_codes = grid.quantize().to(codes)
        changed_from_rtn = int(
            (
                codes[:, : grid.in_features]
                != rtn_codes[:, : grid.in_features]
            )
            .sum()
            .item()
        )
        return {
            "assignment": self.name,
            "grid_scheme": grid.scheme,
            "codes": codes,
            "alpha": self.alpha,
            "act_order": self.act_order,
            "changed_codes": changed_from_rtn,
        }
