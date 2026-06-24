"""Block-wise FlexRound calibration optimization.

This follows FlexRound_LRQ's calibration shape: replace each linear with a
trainable FlexRound quantizer, cache full-precision block outputs, and optimize
the quantizer parameters with reconstruction loss.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from grid_baselines.flatquant_calibration import (
    _block_output,
    _move_tree,
    _remove_dispatch_hooks,
)


def _round_ste(values: torch.Tensor) -> torch.Tensor:
    return values + (values.round() - values).detach()


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
        qmax = 2 ** (bits - 1) - 1
        scale = (2 * torch.maximum(max_val, min_val.abs()) / (2**bits - 1)).clamp_min(1e-8)
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
        codes = _round_ste(weight / divisor)
        if self.symmetric:
            return codes.clamp(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        return codes.clamp(-self.zero_point.to(weight), 2**self.bits - 1 - self.zero_point.to(weight))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self.signed_codes(weight) * self.delta1.exp().to(weight)

    @torch.no_grad()
    def export(self, weight: torch.Tensor) -> dict[str, torch.Tensor]:
        signed = self.signed_codes(weight)
        if self.symmetric:
            codes = signed
        else:
            codes = signed + self.zero_point.to(signed)
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
    weight_bits: int = 4
    weight_symmetric: bool = False
    iters: int = 5000
    batch_size: int = 1
    learning_rate: float = 3e-3


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
    """Optimize one decoder block with cached block reconstruction loss."""
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
    return artifacts, targets, history


# Backward-compatible aliases for callers that imported the initial names.
FlexRoundTrainingConfig = FlexRoundCalibrationConfig
train_flexround_block = calibrate_flexround_block
