"""Block-wise calibration optimization for repository-native FlatQuant artifacts.

Base-model parameters stay frozen.  Each decoder block learns Kronecker input
transforms, diagonal scales, and weight/activation clipping by minimizing the
MSE between the floating-point block and its fake-quantized counterpart.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformed_linear import _apply_kronecker


def _remove_dispatch_hooks(module: nn.Module) -> nn.Module:
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        return module
    remove_hook_from_module(module, recurse=True)
    return module


def factor_dimensions(width: int) -> tuple[int, int]:
    """Return the closest factor pair used by FlatQuant's Kronecker transform."""
    if width <= 0:
        raise ValueError("width must be positive")
    left = int(math.sqrt(width))
    while left > 1 and width % left:
        left -= 1
    return left, width // left


def _round_ste(values: torch.Tensor) -> torch.Tensor:
    return values + (values.round() - values).detach()


def _safe_floor(values: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.as_tensor(eps, device=values.device, dtype=values.dtype)


def _fake_quantize(
    values: torch.Tensor,
    *,
    bits: int,
    symmetric: bool,
    group_size: int,
    clip_max: torch.Tensor | None = None,
    clip_min: torch.Tensor | None = None,
) -> torch.Tensor:
    if bits >= 16:
        return values
    width = values.shape[-1]
    actual_group = width if group_size <= 0 else group_size
    if width % actual_group:
        raise ValueError(
            f"quantized width {width} must be divisible by group size {actual_group}"
        )
    grouped = values.reshape(*values.shape[:-1], -1, actual_group)
    zeros = torch.zeros_like(grouped[..., :1])
    lower = torch.minimum(grouped.amin(-1, keepdim=True), zeros)
    upper = torch.maximum(grouped.amax(-1, keepdim=True), zeros)
    if clip_max is not None:
        factor = clip_max.to(upper)
        if values.dim() == 2 and factor.numel() == values.shape[0]:
            factor = factor.reshape(values.shape[0], 1, 1)
        upper = upper * factor
    if clip_min is not None:
        factor = clip_min.to(lower)
        if values.dim() == 2 and factor.numel() == values.shape[0]:
            factor = factor.reshape(values.shape[0], 1, 1)
        lower = lower * factor

    if symmetric:
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        bound = torch.maximum(lower.abs(), upper)
        scale = torch.where(
            bound == 0,
            torch.ones_like(bound),
            bound.clamp_min(_safe_floor(bound)) / qmax,
        )
        quantized = _round_ste(grouped / scale).clamp(qmin, qmax) * scale
    else:
        qmin, qmax = 0, 2**bits - 1
        value_range = upper - lower
        scale = torch.where(
            value_range == 0,
            torch.ones_like(value_range),
            value_range.clamp_min(_safe_floor(value_range)) / qmax,
        )
        zero = _round_ste(-lower / scale).clamp(qmin, qmax)
        codes = _round_ste(grouped / scale + zero).clamp(qmin, qmax)
        quantized = (codes - zero) * scale
    return quantized.reshape_as(values)


class TrainableKroneckerTransform(nn.Module):
    def __init__(self, width: int, *, add_diagonal: bool = True):
        super().__init__()
        left_dim, right_dim = factor_dimensions(width)
        self.left = nn.Parameter(torch.eye(left_dim))
        self.right = nn.Parameter(torch.eye(right_dim))
        self.diagonal_log = nn.Parameter(torch.zeros(width)) if add_diagonal else None

    @property
    def diagonal(self) -> torch.Tensor | None:
        if self.diagonal_log is None:
            return None
        return self.diagonal_log.exp()

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        diagonal = self.diagonal
        if diagonal is not None:
            values = values * diagonal.to(values)
        return _apply_kronecker(
            values,
            self.left.to(values),
            self.right.to(values),
        )

    def inverse_transpose_weight(self, weight: torch.Tensor) -> torch.Tensor:
        work = weight.float()
        diagonal = self.diagonal
        if diagonal is not None:
            work = work / diagonal.float()
        left_inv_t = torch.linalg.pinv(self.left.float()).t()
        right_inv_t = torch.linalg.pinv(self.right.float()).t()
        return _apply_kronecker(work, left_inv_t, right_inv_t).to(weight.dtype)

    def export(self) -> dict[str, torch.Tensor]:
        result = {
            "matrix_left": self.left.detach().cpu(),
            "matrix_right": self.right.detach().cpu(),
        }
        if self.diagonal is not None:
            result["diag_scale"] = self.diagonal.detach().cpu()
        return result


class CalibrationFlatQuantLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        transform: TrainableKroneckerTransform,
        *,
        weight_bits: int,
        activation_bits: int,
        weight_symmetric: bool,
        activation_symmetric: bool,
        weight_group_size: int,
        activation_group_size: int,
        learn_weight_clipping: bool,
        learn_activation_clipping: bool,
    ):
        super().__init__()
        self.source = source
        self.transform = transform
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_symmetric = weight_symmetric
        self.activation_symmetric = activation_symmetric
        self.weight_group_size = weight_group_size
        self.activation_group_size = activation_group_size
        self.weight_clip_max_logit = nn.Parameter(
            torch.full((source.out_features, 1), 4.0),
            requires_grad=learn_weight_clipping,
        )
        self.weight_clip_min_logit = nn.Parameter(
            torch.full((source.out_features, 1), 4.0),
            requires_grad=learn_weight_clipping,
        )
        self.activation_clip_max_logit = nn.Parameter(
            torch.tensor(4.0), requires_grad=learn_activation_clipping
        )
        self.activation_clip_min_logit = nn.Parameter(
            torch.tensor(4.0), requires_grad=learn_activation_clipping
        )
        for parameter in self.source.parameters():
            parameter.requires_grad = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed_input = self.transform.apply(values)
        weight = self.transform.inverse_transpose_weight(self.source.weight)
        weight = _fake_quantize(
            weight,
            bits=self.weight_bits,
            symmetric=self.weight_symmetric,
            group_size=self.weight_group_size,
            clip_max=self.weight_clip_max_logit.sigmoid(),
            clip_min=self.weight_clip_min_logit.sigmoid(),
        )
        transformed_input = _fake_quantize(
            transformed_input,
            bits=self.activation_bits,
            symmetric=self.activation_symmetric,
            group_size=self.activation_group_size,
            clip_max=self.activation_clip_max_logit.sigmoid(),
            clip_min=self.activation_clip_min_logit.sigmoid(),
        )
        return F.linear(transformed_input, weight, self.source.bias)

    def export_clips(self) -> dict[str, torch.Tensor]:
        return {
            "weight_clip_max": self.weight_clip_max_logit.sigmoid().detach().cpu(),
            "weight_clip_min": self.weight_clip_min_logit.sigmoid().detach().cpu(),
            "activation_clip_max": self.activation_clip_max_logit.sigmoid().detach().cpu(),
            "activation_clip_min": self.activation_clip_min_logit.sigmoid().detach().cpu(),
        }


@dataclass
class FlatQuantCalibrationConfig:
    weight_bits: int = 4
    activation_bits: int = 4
    weight_symmetric: bool = True
    activation_symmetric: bool = True
    weight_group_size: int = 128
    activation_group_size: int = -1
    epochs: int = 15
    batch_size: int = 4
    learning_rate: float = 5e-3
    add_diagonal: bool = True
    learn_weight_clipping: bool = True
    learn_activation_clipping: bool = True
    clipping_learning_rate: float = 1e-4


def _group_key(name: str) -> str:
    if name.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")):
        return "self_attn.qkv"
    if name.endswith(("mlp.up_proj", "mlp.gate_proj")):
        return "mlp.up_gate"
    return name


class TrainableHeadTransform(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.skew = nn.Parameter(torch.zeros(head_dim, head_dim))

    @property
    def matrix(self) -> torch.Tensor:
        S = self.skew - self.skew.t()
        I = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        return torch.linalg.solve(I + S, I - S)


class _FlatQuantTrainableAttention(nn.Module):
    def __init__(self, source: nn.Module, config: FlatQuantCalibrationConfig):
        super().__init__()
        self.source = source
        self.config = config
        head_dim = getattr(source, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(source, "k_proj").source.out_features // getattr(source, "num_key_value_heads", getattr(source, "num_heads", 1))
        self.head_dim = head_dim
        self.p_h = TrainableHeadTransform(self.head_dim)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        from grid_baselines.flatquant_model import resolve_attention_runtime
        runtime = resolve_attention_runtime(self.source)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.source.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.source.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.source.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = runtime.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        matrix = self.p_h.matrix.to(query_states)
        inverse_transpose = torch.linalg.inv(matrix.float()).t().to(query_states)
        query_states = query_states.matmul(inverse_transpose)
        key_states = key_states.matmul(matrix)

        query_states = _fake_quantize(
            query_states,
            bits=self.config.activation_bits,
            symmetric=self.config.activation_symmetric,
            group_size=-1,
            clip_max=self.source.q_proj.activation_clip_max_logit.sigmoid(),
            clip_min=self.source.q_proj.activation_clip_min_logit.sigmoid(),
        )
        key_states = _fake_quantize(
            key_states,
            bits=self.config.activation_bits,
            symmetric=self.config.activation_symmetric,
            group_size=-1,
            clip_max=self.source.k_proj.activation_clip_max_logit.sigmoid(),
            clip_min=self.source.k_proj.activation_clip_min_logit.sigmoid(),
        )
        value_states = _fake_quantize(
            value_states,
            bits=self.config.activation_bits,
            symmetric=self.config.activation_symmetric,
            group_size=-1,
            clip_max=self.source.v_proj.activation_clip_max_logit.sigmoid(),
            clip_min=self.source.v_proj.activation_clip_min_logit.sigmoid(),
        )

        attention_interface = runtime.attention_interfaces.get_interface(
            self.source.config._attn_implementation, runtime.eager_attention_forward
        )
        attention_kwargs = runtime.extra_attention_kwargs(self.source)
        attention_kwargs.update(kwargs)
        attn_output, attn_weights = attention_interface(
            self.source,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.source.attention_dropout,
            scaling=self.source.scaling,
            **attention_kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.source.o_proj(attn_output), attn_weights

    def export(self) -> dict[str, torch.Tensor]:
        return {
            "matrix_h": self.p_h.matrix.detach().cpu(),
            "q_clip_max": self.source.q_proj.activation_clip_max_logit.sigmoid().detach().cpu(),
            "q_clip_min": self.source.q_proj.activation_clip_min_logit.sigmoid().detach().cpu(),
            "k_clip_max": self.source.k_proj.activation_clip_max_logit.sigmoid().detach().cpu(),
            "k_clip_min": self.source.k_proj.activation_clip_min_logit.sigmoid().detach().cpu(),
            "v_clip_max": self.source.v_proj.activation_clip_max_logit.sigmoid().detach().cpu(),
            "v_clip_min": self.source.v_proj.activation_clip_min_logit.sigmoid().detach().cpu(),
        }


def prepare_trainable_block(
    block: nn.Module,
    config: FlatQuantCalibrationConfig,
) -> tuple[nn.Module, dict[str, CalibrationFlatQuantLinear]]:
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
    transforms: dict[str, TrainableKroneckerTransform] = {}
    wrappers: dict[str, CalibrationFlatQuantLinear] = {}
    for name, module in selected.items():
        key = _group_key(name)
        transform = transforms.setdefault(
            key,
            TrainableKroneckerTransform(
                module.in_features, add_diagonal=config.add_diagonal
            ),
        )
        wrapper = CalibrationFlatQuantLinear(
            module,
            transform,
            weight_bits=config.weight_bits,
            activation_bits=config.activation_bits,
            weight_symmetric=config.weight_symmetric,
            activation_symmetric=config.activation_symmetric,
            weight_group_size=config.weight_group_size,
            activation_group_size=config.activation_group_size,
            learn_weight_clipping=config.learn_weight_clipping,
            learn_activation_clipping=config.learn_activation_clipping,
        )
        parent_name, _, child = name.rpartition(".")
        parent = trainable.get_submodule(parent_name) if parent_name else trainable
        setattr(parent, child, wrapper)
        wrappers[name] = wrapper
    if len(wrappers) != 7:
        raise ValueError(f"expected seven FlatQuant linears in block, found {len(wrappers)}")
    if hasattr(trainable, "self_attn"):
        try:
            from grid_baselines.flatquant_model import resolve_attention_runtime
            resolve_attention_runtime(trainable.self_attn)
            trainable.self_attn = _FlatQuantTrainableAttention(trainable.self_attn, config)
        except TypeError:
            pass
    return trainable, wrappers


def _move_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def _block_output(block: nn.Module, hidden: torch.Tensor, kwargs: dict) -> torch.Tensor:
    output = block(hidden, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def calibrate_flatquant_block(
    block: nn.Module,
    inputs: list[torch.Tensor],
    block_kwargs: list[dict],
    *,
    config: FlatQuantCalibrationConfig,
    device: torch.device | str,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[torch.Tensor], list[float]]:
    """Optimize one decoder block and return normalized per-linear artifacts."""
    if len(inputs) != len(block_kwargs) or not inputs:
        raise ValueError("inputs and block_kwargs must be non-empty and aligned")
    device = torch.device(device)
    reference = _remove_dispatch_hooks(copy.deepcopy(block)).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    quantized, wrappers = prepare_trainable_block(block, config)
    quantized = quantized.to(device).train()

    targets = []
    with torch.no_grad():
        for hidden, kwargs in zip(inputs, block_kwargs):
            targets.append(
                _block_output(reference, hidden.to(device), _move_tree(kwargs, device))
                .detach()
                .cpu()
            )

    transform_parameters = []
    clipping_parameters = []
    for name, parameter in quantized.named_parameters():
        if not parameter.requires_grad:
            continue
        if "clip" in name:
            clipping_parameters.append(parameter)
        else:
            transform_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": transform_parameters},
            {"params": clipping_parameters, "lr": config.clipping_learning_rate},
        ],
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs * math.ceil(len(inputs) / config.batch_size)),
        eta_min=config.learning_rate * 1e-3,
    )
    history = []
    for _epoch in range(config.epochs):
        permutation = torch.randperm(len(inputs)).tolist()
        epoch_loss = 0.0
        for start in range(0, len(inputs), config.batch_size):
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
            torch.nn.utils.clip_grad_norm_(transform_parameters + clipping_parameters, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(indices)
        history.append(epoch_loss / len(inputs))

    artifacts = {}
    for name, wrapper in wrappers.items():
        artifacts[name] = {
            **wrapper.transform.export(),
            **wrapper.export_clips(),
        }
    if hasattr(quantized, "self_attn") and isinstance(quantized.self_attn, _FlatQuantTrainableAttention):
        attn_export = quantized.self_attn.export()
        artifacts["self_attn.kcache_trans"] = {"matrix": attn_export.pop("matrix_h")}
        # the clips should be recorded inside q_proj, k_proj, v_proj
        artifacts["self_attn.q_proj"].update({
            "act_quantizer.clip_factor_a_max": attn_export.pop("q_clip_max"),
            "act_quantizer.clip_factor_a_min": attn_export.pop("q_clip_min"),
        })
        artifacts["self_attn.k_proj"].update({
            "act_quantizer.clip_factor_a_max": attn_export.pop("k_clip_max"),
            "act_quantizer.clip_factor_a_min": attn_export.pop("k_clip_min"),
        })
        artifacts["self_attn.v_proj"].update({
            "act_quantizer.clip_factor_a_max": attn_export.pop("v_clip_max"),
            "act_quantizer.clip_factor_a_min": attn_export.pop("v_clip_min"),
        })

    return artifacts, targets, history

