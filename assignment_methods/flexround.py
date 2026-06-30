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

``delta1`` is initialized from per-output-channel qparams and is learnable by
default, matching the LLM setting in the original paper.  Set
``learn_layer_scale=False`` to keep that per-channel scale fixed while learning
only the FlexRound divisors.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from eigenflip.statistics.trust_region import LayerStats
from grid_baselines.flatquant_calibration import (
    _block_output,
    _move_tree,
    _remove_dispatch_hooks,
)

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

    @staticmethod
    def _store_channel_qparams(
        grid,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        dequant_column_scale: torch.Tensor,
    ) -> None:
        effective_scale = scale / dequant_column_scale
        grid.scale.data.copy_(
            effective_scale.expand_as(grid.scale).to(grid.scale.dtype)
        )
        grid.zero_point.data.copy_(
            zero_point.expand_as(grid.zero_point).to(grid.zero_point.dtype)
        )

    def _fake_quantize(
        self,
        weights: torch.Tensor,
        log_delta1: torch.Tensor,
        zero_point: torch.Tensor,
        log_element_divisor: torch.Tensor,
        log_row_scale: torch.Tensor | None,
        dequant_column_scale: torch.Tensor,
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
        dequantized = signed_codes * torch.exp(log_delta1) / dequant_column_scale
        return signed_codes + zero_point, dequantized

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
        awq_scales = getattr(grid, "awq_scales", None)
        if awq_scales is None:
            quantized_source = padded_weights
            dequant_column_scale = torch.ones(
                1,
                padded_weights.shape[1],
                device=device,
                dtype=dtype,
            )
        else:
            dequant_column_scale = awq_scales.detach().to(device=device, dtype=dtype)
            quantized_source = grid.scaled_weights.detach().to(
                device=device,
                dtype=dtype,
            )
        if grid.scheme not in {"symmetric", "asymmetric"}:
            raise ValueError(
                f"FlexRound requires symmetric/asymmetric grids, got {grid.scheme!r}"
            )
        if self.learn_layer_scale:
            scale, zero_point = _channelwise_qparams(
                quantized_source,
                bits=grid.bits,
                symmetric=grid.scheme == "symmetric",
            )
        else:
            # Keep the selected grid exactly fixed. AWQ grids store their
            # effective dequantization scale after undoing the activation-aware
            # column scaling, so recover the source-domain scale here.
            scale = grid.scale.detach() * dequant_column_scale
            zero_point = grid.zero_point.detach()
        scale = scale.to(device=device, dtype=dtype)
        zero_point = zero_point.to(device=device, dtype=dtype)
        diagonal = stats.D.detach().to(device=device, dtype=dtype)
        low_rank = stats.V.detach().to(device=device, dtype=dtype)
        target = padded_weights[:, : grid.in_features]

        with torch.no_grad():
            zero_element_scale = torch.zeros_like(quantized_source)
            rtn_codes, rtn_dequantized = self._fake_quantize(
                quantized_source,
                scale.log(),
                zero_point,
                zero_element_scale,
                None,
                dequant_column_scale,
                grid.qmin,
                grid.qmax,
            )
            initial_loss_tensor = self._reconstruction_loss(
                target,
                rtn_dequantized[:, : grid.in_features],
                diagonal,
                low_rank,
            )
            initial_loss = float(initial_loss_tensor.item())

        if self.steps == 0:
            if self.learn_layer_scale:
                self._store_channel_qparams(
                    grid,
                    scale,
                    zero_point,
                    dequant_column_scale,
                )
            output = rtn_dequantized[:, : grid.in_features].to(grid.original_dtype)
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
        log_element_scale = torch.nn.Parameter(torch.zeros_like(quantized_source))
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
                    quantized_source,
                    log_delta1,
                    zero_point,
                    log_element_scale,
                    log_row_scale,
                    dequant_column_scale,
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
                    quantized_source,
                    log_delta1,
                    zero_point,
                    log_element_scale,
                    log_row_scale,
                    dequant_column_scale,
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
        final_scale = torch.exp(log_delta1.detach())
        if self.learn_layer_scale:
            self._store_channel_qparams(
                grid,
                final_scale,
                zero_point,
                dequant_column_scale,
            )

        output = final_dequantized[:, : grid.in_features].detach().to(
            grid.original_dtype
        )
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
            "variant": (
                self.variant
                if self.learn_layer_scale
                else "fixed_grid_surrogate"
            ),
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


def _channelwise_qparams(
    weight: torch.Tensor,
    *,
    bits: int,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    row_min = weight.amin(dim=1, keepdim=True)
    row_max = weight.amax(dim=1, keepdim=True)
    zeros = torch.zeros_like(row_min)
    min_val = torch.minimum(row_min, zeros)
    max_val = torch.maximum(row_max, zeros)
    if symmetric:
        scale = (
            2 * torch.maximum(max_val, min_val.abs()) / (2**bits - 1)
        ).clamp_min(1e-8)
        zero_point = torch.zeros_like(scale)
    else:
        qmax = 2**bits - 1
        scale = ((max_val - min_val) / qmax).clamp_min(1e-8)
        zero_point = torch.round(-min_val / scale).clamp(0, qmax)
    return scale, zero_point


class FlexRoundQuantizer(nn.Module):
    """Official FlexRound delta1/delta2/delta3 weight quantizer."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        bits: int,
        symmetric: bool,
    ):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        scale, zero_point = _channelwise_qparams(
            weight.detach().float(),
            bits=bits,
            symmetric=symmetric,
        )
        self.delta1 = nn.Parameter(scale.log().to(weight.dtype))
        self.delta2 = nn.Parameter(torch.zeros_like(weight))
        self.delta3 = nn.Parameter(torch.zeros_like(weight[:, :1]))
        self.register_buffer("zero_point", zero_point.to(weight.dtype))

    def signed_codes(self, weight: torch.Tensor) -> torch.Tensor:
        divisor = (self.delta1 + self.delta2 + self.delta3).exp().to(weight)
        codes = _ste_round(weight / divisor)
        if self.symmetric:
            return codes.clamp(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        return codes.clamp(
            -self.zero_point.to(weight),
            2**self.bits - 1 - self.zero_point.to(weight),
        )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self.signed_codes(weight) * self.delta1.exp().to(weight)

    @torch.no_grad()
    def export(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        signed = self.signed_codes(weight)
        codes = signed if self.symmetric else signed + self.zero_point.to(signed)
        return {
            "weight": (signed * self.delta1.exp().to(signed)).detach().cpu(),
            "codes": codes.detach().cpu(),
            "delta1": self.delta1.detach().cpu(),
            "delta2": self.delta2.detach().cpu(),
            "delta3": self.delta3.detach().cpu(),
            "zero_point": self.zero_point.detach().cpu(),
        }


class FlexRoundLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        *,
        bits: int,
        symmetric: bool,
    ):
        super().__init__()
        self.source = source
        self.quantizer = FlexRoundQuantizer(
            source.weight.detach(),
            bits=bits,
            symmetric=symmetric,
        )
        for parameter in self.source.parameters():
            parameter.requires_grad = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weight = self.quantizer(self.source.weight)
        return F.linear(values, weight, self.source.bias)

    def export(self) -> dict[str, torch.Tensor]:
        return self.quantizer.export(self.source.weight)


@dataclass
class FlexRoundCalibrationConfig:
    weight_bits: int = 3
    weight_symmetric: bool = False
    iters: int = 5000
    batch_size: int = 1
    learning_rate: float = 3e-3
    propagate_quantized_inputs: bool = True


@torch.no_grad()
def apply_flexround_artifact(
    block: nn.Module,
    artifact: dict[str, dict[str, torch.Tensor]],
    *,
    strict: bool = True,
) -> None:
    """Apply exported FlexRound weights to their source linear modules."""

    expected = {
        name
        for name, module in block.named_modules()
        if isinstance(module, nn.Linear)
        and name.endswith(
            ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj")
        )
    }
    provided = set(artifact)
    if strict and provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(
            "FlexRound artifact layer mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(expected & provided):
        module = block.get_submodule(name)
        weight = artifact[name].get("weight")
        if weight is None:
            raise ValueError(f"FlexRound artifact for {name!r} has no weight")
        if tuple(weight.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"FlexRound weight shape mismatch for {name!r}: "
                f"artifact={tuple(weight.shape)}, model={tuple(module.weight.shape)}"
            )
        module.weight.data.copy_(
            weight.to(device=module.weight.device, dtype=module.weight.dtype)
        )


def prepare_flexround_block(
    block: nn.Module,
    config: FlexRoundCalibrationConfig,
) -> tuple[nn.Module, dict[str, FlexRoundLinear]]:
    trainable = _remove_dispatch_hooks(copy.deepcopy(block))
    for parameter in trainable.parameters():
        parameter.requires_grad = False
    selected = {
        name: module
        for name, module in trainable.named_modules()
        if isinstance(module, nn.Linear)
        and name.endswith(
            ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj")
        )
    }
    wrappers: dict[str, FlexRoundLinear] = {}
    for name, module in selected.items():
        wrapper = FlexRoundLinear(
            module,
            bits=config.weight_bits,
            symmetric=config.weight_symmetric,
        )
        parent_name, _, child = name.rpartition(".")
        parent = trainable.get_submodule(parent_name) if parent_name else trainable
        setattr(parent, child, wrapper)
        wrappers[name] = wrapper
    if len(wrappers) != 7:
        raise ValueError(f"expected seven FlexRound linears in block, found {len(wrappers)}")
    return trainable, wrappers


def calibrate_flexround_block(
    block: nn.Module,
    inputs: list[torch.Tensor],
    block_kwargs: list[dict],
    *,
    config: FlexRoundCalibrationConfig,
    device: torch.device | str,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[torch.Tensor], list[float]]:
    """Optimize one decoder block with cached block reconstruction loss.

    The returned inputs are produced by the optimized quantized block by
    default, so sequential calibration propagates the quantized path. Set
    ``propagate_quantized_inputs=False`` to retain teacher-output propagation.
    """
    if len(inputs) != len(block_kwargs) or not inputs:
        raise ValueError("inputs and block_kwargs must be non-empty and aligned")
    if config.iters <= 0 or config.batch_size <= 0:
        raise ValueError("iters and batch_size must be positive")

    device = torch.device(device)
    reference = _remove_dispatch_hooks(copy.deepcopy(block)).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    quantized, wrappers = prepare_flexround_block(block, config)
    quantized = quantized.to(device).train()

    targets = []
    with torch.no_grad():
        for hidden, kwargs in zip(inputs, block_kwargs):
            targets.append(
                _block_output(reference, hidden.to(device), _move_tree(kwargs, device))
                .detach()
                .cpu()
            )

    parameters = [
        parameter
        for name, parameter in quantized.named_parameters()
        if parameter.requires_grad and ".quantizer." in name
    ]
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.iters,
        eta_min=0.0,
    )

    history = []
    steps_done = 0
    epochs = math.ceil(config.iters / math.ceil(len(inputs) / config.batch_size))
    for _epoch in range(epochs):
        permutation = torch.randperm(len(inputs)).tolist()
        for start in range(0, len(inputs), config.batch_size):
            if steps_done >= config.iters:
                break
            indices = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in indices:
                prediction = _block_output(
                    quantized,
                    inputs[index].to(device),
                    _move_tree(block_kwargs[index], device),
                )
                losses.append(F.mse_loss(prediction.float(), targets[index].to(device).float()))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            history.append(float(loss.detach()))
            steps_done += 1
        if steps_done >= config.iters:
            break

    artifacts = {name: wrapper.export() for name, wrapper in wrappers.items()}
    if config.propagate_quantized_inputs:
        next_inputs = []
        quantized.eval()
        with torch.no_grad():
            for hidden, kwargs in zip(inputs, block_kwargs):
                next_inputs.append(
                    _block_output(
                        quantized,
                        hidden.to(device),
                        _move_tree(kwargs, device),
                    )
                    .detach()
                    .cpu()
                )
    else:
        next_inputs = targets
    return artifacts, next_inputs, history
