"""Fixed-grid FlexRound assignment baseline.

This module adapts FlexRound's element-wise division rule to this repository's
``grid x assignment`` experiments.  The grid scale and zero-point are kept
fixed so Vanilla, AWQ, and future grid baselines remain directly comparable.
Only the integer-code assignment is learned.

NOTE: This is a fixed-grid surrogate assignment method, NOT the full official 
FlexRound paper implementation.

For a fixed base grid scale ``delta1 = log(scale)`` and a positive learned
FlexRound divisor ``S`` the fake-quantized weight follows the official
``delta1 + delta2 + delta3`` parameterization:

    q_signed = clip(round(W / exp(delta1 + log(S))),
                    qmin - zero_point, qmax - zero_point)
    W_hat    = q_signed * exp(delta1)

``S`` is factorized into an element-wise term and, optionally, a shared output
channel term as in FlexRound's linear-layer formulation.  The parameters are
optimized against the calibration reconstruction surrogate already carried by
``LayerStats``:

    H_tilde = diag(D) + V V^T
    loss    = sum_rows (W_hat - W) H_tilde (W_hat - W)^T

Rounding uses a straight-through estimator.  The learned divisors are only an
optimization device; the returned checkpoint contains ordinary integer codes
on the original fixed grid and has no per-weight inference overhead.
"""

from __future__ import annotations

import torch

from eigenflip.statistics.trust_region import LayerStats

def _ste_round(values: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass and use the identity gradient backward."""

    return values + (torch.round(values) - values).detach()


class FlexRoundAssignment:
    """Learn FlexRound integer assignments on an existing fixed grid."""

    name = "flexround"
    variant = "fixed_grid_surrogate"

    def __init__(
        self,
        *,
        steps: int = 5000,
        lr: float = 2e-4,
        log_divisor_bound: float = 6.0,
        learn_layer_scale: bool = False,
        learn_row_scale: bool = True,
        work_dtype: torch.dtype = torch.float32,
    ):
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if log_divisor_bound <= 0:
            raise ValueError(
                "log_divisor_bound must be positive, "
                f"got {log_divisor_bound}"
            )
        if work_dtype not in {torch.float32, torch.float64}:
            raise ValueError("work_dtype must be torch.float32 or torch.float64")

        self.steps = steps
        self.lr = lr
        self.log_divisor_bound = log_divisor_bound
        self.learn_layer_scale = learn_layer_scale
        self.learn_row_scale = learn_row_scale
        self.work_dtype = work_dtype

    def _fake_quantize(
        self,
        weights: torch.Tensor,
        log_base_scale: torch.Tensor,
        zero_point: torch.Tensor,
        log_element_divisor: torch.Tensor,
        log_layer_scale: torch.Tensor | None,
        log_row_scale: torch.Tensor | None,
        qmin: int,
        qmax: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dequant_scale = log_base_scale
        if log_layer_scale is not None:
            dequant_scale = dequant_scale + log_layer_scale

        log_divisor = log_element_divisor
        if log_row_scale is not None:
            log_divisor = log_divisor + log_row_scale

        log_quant_scale = dequant_scale + log_divisor.clamp(
            min=-self.log_divisor_bound,
            max=self.log_divisor_bound,
        )
        signed_codes = _ste_round(weights / torch.exp(log_quant_scale)).clamp(
            qmin - zero_point,
            qmax - zero_point,
        )
        return signed_codes + zero_point, signed_codes * torch.exp(dequant_scale)

    @staticmethod
    def _reconstruction_loss(
        target: torch.Tensor,
        dequantized: torch.Tensor,
        diagonal: torch.Tensor,
        low_rank: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean output-channel energy under ``diag(D) + V V^T``."""

        residual = dequantized - target
        diagonal_energy = (residual.square() * diagonal.unsqueeze(0)).sum()
        low_rank_energy = (residual @ low_rank).square().sum()
        return (diagonal_energy + low_rank_energy) / max(1, target.shape[0])

    def apply_to_grid(
        self,
        grid,
        stats: LayerStats,
    ) -> tuple[torch.Tensor, dict]:
        """Optimize assignments and return dequantized weights plus metadata."""

        if stats.D is None or stats.V is None:
            raise ValueError("FlexRound requires built LayerStats with D and V")
        if stats.d != grid.in_features:
            raise ValueError(
                f"stats dimension {stats.d} does not match grid input "
                f"dimension {grid.in_features}"
            )

        device = grid.float_weights.device
        dtype = self.work_dtype
        padded_weights = grid.float_weights.detach().to(device=device, dtype=dtype)
        scale = grid.scale.detach().to(device=device, dtype=dtype)
        log_base_scale = scale.log()
        zero_point = grid.zero_point.detach().to(device=device, dtype=dtype)
        diagonal = stats.D.detach().to(device=device, dtype=dtype)
        low_rank = stats.V.detach().to(device=device, dtype=dtype)
        target = padded_weights[:, : grid.in_features]

        with torch.no_grad():
            rtn_codes = grid.quantize().to(device=device, dtype=dtype)
            rtn_dequantized = (rtn_codes - zero_point) * scale
            initial_loss_tensor = self._reconstruction_loss(
                target,
                rtn_dequantized[:, : grid.in_features],
                diagonal,
                low_rank,
            )
            initial_loss = float(initial_loss_tensor.item())

        if self.steps == 0:
            output = grid.dequantize(rtn_codes).to(grid.original_dtype)
            return output, self._info(
                grid,
                rtn_codes,
                rtn_codes,
                initial_loss,
                initial_loss,
            )

        log_element_scale = torch.nn.Parameter(torch.zeros_like(padded_weights))
        parameters: list[torch.nn.Parameter] = [log_element_scale]
        log_layer_scale = None
        if self.learn_layer_scale:
            log_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, device=device, dtype=dtype)
            )
            parameters.append(log_layer_scale)
        log_row_scale = None
        if self.learn_row_scale:
            log_row_scale = torch.nn.Parameter(
                torch.zeros(
                    padded_weights.shape[0],
                    1,
                    device=device,
                    dtype=dtype,
                )
            )
            parameters.append(log_row_scale)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        # The outer collector runs under no_grad; explicitly re-enable
        # autograd only for the local FlexRound optimization.
        with torch.enable_grad():
            for step in range(1, self.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                codes, dequantized = self._fake_quantize(
                    padded_weights,
                    log_base_scale,
                    zero_point,
                    log_element_scale,
                    log_layer_scale,
                    log_row_scale,
                    grid.qmin,
                    grid.qmax,
                )
                loss = self._reconstruction_loss(
                    target,
                    dequantized[:, : grid.in_features],
                    diagonal,
                    low_rank,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"FlexRound loss became non-finite at step {step}"
                    )
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                final_codes, final_dequantized = self._fake_quantize(
                    padded_weights,
                    log_base_scale,
                    zero_point,
                    log_element_scale,
                    log_layer_scale,
                    log_row_scale,
                    grid.qmin,
                    grid.qmax,
                )
                final_loss = float(
                    self._reconstruction_loss(
                        target,
                        final_dequantized[:, : grid.in_features],
                        diagonal,
                        low_rank,
                    ).item()
                )
                if not torch.isfinite(final_dequantized).all():
                    raise RuntimeError("FlexRound produced non-finite final weights")

        final_codes = final_codes.detach()
        if self.learn_layer_scale and log_layer_scale is not None:
            grid.scale.data = grid.scale.data * torch.exp(log_layer_scale.detach()).to(grid.scale.dtype)

        output = grid.dequantize(final_codes).to(grid.original_dtype)
        info = self._info(
            grid,
            final_codes,
            rtn_codes,
            initial_loss,
            final_loss,
        )

        del optimizer, parameters, log_element_scale, log_layer_scale, log_row_scale
        return output, info

    def _info(
        self,
        grid,
        final_codes: torch.Tensor,
        rtn_codes: torch.Tensor,
        initial_loss: float,
        final_loss: float,
    ) -> dict:
        actual_codes = final_codes[:, : grid.in_features]
        actual_rtn = rtn_codes[:, : grid.in_features]
        changed = int((actual_codes != actual_rtn).sum().item())
        total = actual_codes.numel()
        return {
            "assignment": self.name,
            "variant": self.variant,
            "grid_scheme": grid.scheme,
            "codes": final_codes.detach(),
            "steps": self.steps,
            "lr": self.lr,
            "learn_layer_scale": self.learn_layer_scale,
            "learn_row_scale": self.learn_row_scale,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "changed_codes": changed,
            "changed_fraction": changed / max(1, total),
        }
