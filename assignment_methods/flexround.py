"""FlexRound assignment baseline.

This module adapts FlexRound's official weight quantizer to this repository's
``grid x assignment`` runner.  It learns the same ``delta1 + delta2 + delta3``
parameters as FlexRound_LRQ's ``UniformAffineQuantizer`` while optimizing the
calibration reconstruction surrogate already carried by ``LayerStats``:

    H_tilde = diag(D) + V V^T
    loss    = sum_rows (W_hat - W) H_tilde (W_hat - W)^T

It is still not the full FlexRound_LRQ block-reconstruction pipeline, because
the local assignment API does not receive cached block input/output tensors.

The fake-quantized weight follows the official parameterization:

    q_signed = clip(round(W / exp(delta1 + log(S))),
                    qmin - zero_point, qmax - zero_point)
    W_hat    = q_signed * exp(delta1)

``delta1`` is initialized from the selected grid scale and is learnable by
default, matching the original code.  Set ``learn_layer_scale=False`` to recover
the older fixed-grid assignment-only behavior.
"""

from __future__ import annotations

import torch

from eigenflip.statistics.trust_region import LayerStats

def _ste_round(values: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass and use the identity gradient backward."""

    return values + (torch.round(values) - values).detach()


class FlexRoundAssignment:
    """Learn FlexRound integer assignments on an existing grid."""

    name = "flexround"
    variant = "official_quantizer_surrogate"

    def __init__(
        self,
        *,
        steps: int = 5000,
        lr: float = 3e-3,
        log_divisor_bound: float = float("inf"),
        learn_layer_scale: bool = True,
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
        log_delta1: torch.Tensor,
        zero_point: torch.Tensor,
        log_element_divisor: torch.Tensor,
        log_row_scale: torch.Tensor | None,
        qmin: int,
        qmax: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_divisor = log_element_divisor
        if log_row_scale is not None:
            log_divisor = log_divisor + log_row_scale

        if self.log_divisor_bound != float("inf"):
            log_divisor = log_divisor.clamp(
                min=-self.log_divisor_bound,
                max=self.log_divisor_bound,
            )
        log_quant_scale = log_delta1 + log_divisor
        signed_codes = _ste_round(weights / torch.exp(log_quant_scale)).clamp(
            qmin - zero_point,
            qmax - zero_point,
        )
        return signed_codes + zero_point, signed_codes * torch.exp(log_delta1)

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

        log_delta1 = scale.log().clone()
        parameters: list[torch.nn.Parameter] = []
        if self.learn_layer_scale:
            log_delta1 = torch.nn.Parameter(log_delta1)
            parameters.append(log_delta1)
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
        log_element_scale = torch.nn.Parameter(torch.zeros_like(padded_weights))
        parameters.append(log_element_scale)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.steps),
            eta_min=0.0,
        )

        # The outer collector runs under no_grad; explicitly re-enable
        # autograd only for the local FlexRound optimization.
        with torch.enable_grad():
            for step in range(1, self.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                codes, dequantized = self._fake_quantize(
                    padded_weights,
                    log_delta1,
                    zero_point,
                    log_element_scale,
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
                scheduler.step()

            with torch.no_grad():
                final_codes, final_dequantized = self._fake_quantize(
                    padded_weights,
                    log_delta1,
                    zero_point,
                    log_element_scale,
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
        if self.learn_layer_scale:
            grid.scale.data = torch.exp(log_delta1.detach()).to(grid.scale.dtype)

        output = grid.dequantize(final_codes).to(grid.original_dtype)
        info = self._info(
            grid,
            final_codes,
            rtn_codes,
            initial_loss,
            final_loss,
        )

        del optimizer, scheduler, parameters, log_delta1, log_element_scale, log_row_scale
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
