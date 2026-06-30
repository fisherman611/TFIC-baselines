"""Calibration optimization for SpinQuant R1/R2 rotation artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spinquant_quantization_grid import (
    SpinQuantRotations,
    cayley_update,
    fuse_spinquant_norms,
)
from .transformed_linear import apply_factorized_hadamard, fake_quantize_activation


@dataclass
class SpinQuantCalibrationConfig:
    weight_bits: int = 3
    weight_group_size: int = 128
    weight_scheme: str = "asymmetric"
    activation_bits: int = 16
    activation_symmetric: bool = False
    activation_group_size: int = -1
    activation_clip_ratio: float = 1.0
    v_bits: int = 16
    v_symmetric: bool = False
    v_clip_ratio: float = 1.0
    r1_steps: int = 100
    r2_steps: int | None = None
    batch_size: int = 4
    learning_rate: float = 1e-3
    objective: str = "cross_entropy"


def identity_spinquant_rotations(
    *,
    num_layers: int,
    hidden_size: int,
    head_dim: int,
) -> SpinQuantRotations:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    return SpinQuantRotations(
        R1=torch.eye(hidden_size, dtype=torch.float64),
        R2={idx: torch.eye(head_dim, dtype=torch.float64) for idx in range(num_layers)},
    )


def _random_signed_hadamard(size: int, generator: torch.Generator) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(
            "Hadamard SpinQuant initialization requires power-of-two dimensions; "
            "use rotation_init='random' or 'identity' for this model"
        )

    matrix = apply_factorized_hadamard(
        torch.eye(size, dtype=torch.float64),
        had_k=None,
        k=1,
    )
    signs = torch.randint(
        0,
        2,
        (size,),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float64)
    signs = signs.mul_(2).sub_(1)
    return (matrix * signs.unsqueeze(0)).contiguous()


def hadamard_spinquant_rotations(
    *,
    num_layers: int,
    hidden_size: int,
    head_dim: int,
    seed: int = 0,
) -> SpinQuantRotations:
    """Generate random-signed Hadamard R1/R2 rotations for SpinQuant init."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    generator = torch.Generator().manual_seed(seed)
    return SpinQuantRotations(
        R1=_random_signed_hadamard(hidden_size, generator),
        R2={
            idx: _random_signed_hadamard(head_dim, generator)
            for idx in range(num_layers)
        },
    )


def _round_ste(values: torch.Tensor) -> torch.Tensor:
    return values + (values.round() - values).detach()


def _fake_quantize_weight(
    values: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    scheme: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    if bits >= 16:
        return values
    if scheme not in {"symmetric", "asymmetric"}:
        raise ValueError("weight_scheme must be symmetric or asymmetric")
    width = values.shape[-1]
    actual_group = width if group_size <= 0 else group_size
    if width % actual_group:
        raise ValueError("weight width must be divisible by weight_group_size")
    grouped = values.reshape(*values.shape[:-1], -1, actual_group)
    zeros = torch.zeros_like(grouped[..., :1])
    lower = torch.minimum(grouped.amin(dim=-1, keepdim=True), zeros)
    upper = torch.maximum(grouped.amax(dim=-1, keepdim=True), zeros)
    if scheme == "symmetric":
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        bound = torch.maximum(lower.abs(), upper)
        scale = torch.where(
            bound == 0,
            torch.ones_like(bound),
            bound.clamp_min(eps) / qmax,
        )
        quantized = _round_ste(grouped / scale).clamp(qmin, qmax) * scale
    else:
        qmin, qmax = 0, 2**bits - 1
        value_range = upper - lower
        scale = torch.where(
            value_range == 0,
            torch.ones_like(value_range),
            value_range.clamp_min(eps) / qmax,
        )
        zero = _round_ste(-lower / scale).clamp(qmin, qmax)
        codes = _round_ste(grouped / scale + zero).clamp(qmin, qmax)
        quantized = (codes - zero) * scale
    return quantized.reshape_as(values)


def _quantize_activation(values: torch.Tensor, config: SpinQuantCalibrationConfig):
    return fake_quantize_activation(
        values,
        bits=config.activation_bits,
        symmetric=config.activation_symmetric,
        group_size=config.activation_group_size,
        clip_ratio=config.activation_clip_ratio,
    )


def _head_output_rotate(values: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    head_dim = rotation.shape[0]
    grouped = values.reshape(*values.shape[:-1], -1, head_dim)
    return grouped.matmul(rotation).reshape_as(values)


def _head_output_rotate_transpose(
    values: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    return _head_output_rotate(values, rotation.t())


def _rotate_v_weight(
    weight: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
) -> torch.Tensor:
    rotated = weight.matmul(r1)
    grouped = rotated.reshape(-1, r2.shape[0], rotated.shape[1])
    return r2.t().unsqueeze(0).matmul(grouped).reshape_as(rotated)


def _rotate_o_weight(
    weight: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
) -> torch.Tensor:
    rotated = r1.t().matmul(weight)
    grouped = rotated.reshape(rotated.shape[0], -1, r2.shape[0])
    return grouped.matmul(r2).reshape_as(rotated)


def _rotate_bias_left(
    bias: torch.Tensor | None,
    rotation: torch.Tensor,
) -> torch.Tensor | None:
    if bias is None:
        return None
    return rotation.t().matmul(bias)


def _rotate_bias_head_transpose(
    bias: torch.Tensor | None,
    rotation: torch.Tensor,
) -> torch.Tensor | None:
    if bias is None:
        return None
    return _head_output_rotate_transpose(bias, rotation)


def _linear_weight_loss(
    values: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: SpinQuantCalibrationConfig,
) -> torch.Tensor:
    quantized_weight = _fake_quantize_weight(
        weight,
        bits=config.weight_bits,
        group_size=config.weight_group_size,
        scheme=config.weight_scheme,
    )
    return F.mse_loss(F.linear(values, quantized_weight, bias).float(), target.float())


def _sample_batch(
    tensors: list[torch.Tensor],
    indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.cat([tensors[idx].to(device=device, dtype=dtype) for idx in indices])


class _RotationBank:
    def __init__(self, rotations: SpinQuantRotations):
        self.r1 = rotations.R1
        self.r2 = dict(rotations.R2)


class _SpinQuantEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding, bank: _RotationBank):
        super().__init__()
        self.bank = bank
        self.padding_idx = source.padding_idx
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.register_buffer("weight", source.weight.detach().clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=self.bank.r1.dtype).matmul(self.bank.r1)
        return F.embedding(input_ids, weight, padding_idx=self.padding_idx)


class _SpinQuantLmHead(nn.Module):
    def __init__(self, source: nn.Linear, bank: _RotationBank):
        super().__init__()
        self.bank = bank
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer("weight", source.weight.detach().clone())
        if source.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=self.bank.r1.dtype).matmul(self.bank.r1)
        bias = None if self.bias is None else self.bias.to(dtype=self.bank.r1.dtype)
        return F.linear(values.to(dtype=self.bank.r1.dtype), weight, bias)


class _SpinQuantTrainableLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        *,
        bank: _RotationBank,
        layer_idx: int,
        role: str,
        config: SpinQuantCalibrationConfig,
        head_dim: int,
    ):
        super().__init__()
        self.bank = bank
        self.layer_idx = layer_idx
        self.role = role
        self.config = config
        self.head_dim = head_dim
        self.in_features = source.in_features
        self.out_features = source.out_features

        from .transformed_linear import SpinQuantHadamardLinear
        self.has_hadamard = isinstance(source, SpinQuantHadamardLinear)
        if self.has_hadamard:
            self.register_buffer("had_k", source.had_k.detach().clone())
            self.k = source.k

        self.register_buffer("weight", source.weight.detach().clone())
        if source.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def _weight_and_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        r1 = self.bank.r1
        r2 = self.bank.r2[self.layer_idx]
        weight = self.weight.to(dtype=r1.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=r1.dtype)
        if self.role in {"q", "k", "up", "gate"}:
            weight = weight.matmul(r1)
        elif self.role == "v":
            weight = _rotate_v_weight(weight, r1, r2)
            bias = _rotate_bias_head_transpose(bias, r2)
        elif self.role == "o":
            weight = _rotate_o_weight(weight, r1, r2)
            bias = _rotate_bias_left(bias, r1)
        elif self.role == "down":
            weight = r1.t().matmul(weight)
            bias = _rotate_bias_left(bias, r1)
        else:
            raise ValueError(f"unknown SpinQuant calibration role: {self.role}")
        return (
            _fake_quantize_weight(
                weight,
                bits=self.config.weight_bits,
                group_size=self.config.weight_group_size,
                scheme=self.config.weight_scheme,
            ),
            bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if getattr(self, "has_hadamard", False) and self.role == "down":
            from .transformed_linear import apply_factorized_hadamard
            values = apply_factorized_hadamard(values, had_k=self.had_k, k=self.k)

        group_size = self.config.activation_group_size
        if self.role == "o":
            group_size = self.head_dim
        values = fake_quantize_activation(
            values.to(dtype=self.bank.r1.dtype),
            bits=self.config.activation_bits,
            symmetric=self.config.activation_symmetric,
            group_size=group_size,
            clip_ratio=self.config.activation_clip_ratio,
        )
        weight, bias = self._weight_and_bias()
        output = F.linear(values, weight, bias)

        if self.role == "v":
            output = fake_quantize_activation(
                output,
                bits=self.config.v_bits,
                symmetric=self.config.v_symmetric,
                group_size=self.head_dim,
                clip_ratio=self.config.v_clip_ratio,
            )

        return output


def install_spinquant_calibration_wrappers(
    model,
    rotations: SpinQuantRotations,
    config: SpinQuantCalibrationConfig,
) -> _RotationBank:
    """Install differentiable no-had SpinQuant fake-quant wrappers in place."""

    fuse_spinquant_norms(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    bank = _RotationBank(rotations)
    model.model.embed_tokens = _SpinQuantEmbedding(model.model.embed_tokens, bank)
    model.lm_head = _SpinQuantLmHead(model.lm_head, bank)
    for layer_idx, layer in enumerate(model.model.layers):
        head_dim = int(layer.self_attn.head_dim)
        replacements = {
            "q_proj": "q",
            "k_proj": "k",
            "v_proj": "v",
            "o_proj": "o",
        }
        for child, role in replacements.items():
            setattr(
                layer.self_attn,
                child,
                _SpinQuantTrainableLinear(
                    getattr(layer.self_attn, child),
                    bank=bank,
                    layer_idx=layer_idx,
                    role=role,
                    config=config,
                    head_dim=head_dim,
                ),
            )
        for child, role in {"up_proj": "up", "gate_proj": "gate", "down_proj": "down"}.items():
            setattr(
                layer.mlp,
                child,
                _SpinQuantTrainableLinear(
                    getattr(layer.mlp, child),
                    bank=bank,
                    layer_idx=layer_idx,
                    role=role,
                    config=config,
                    head_dim=head_dim,
                ),
            )
    import types
    from grid_baselines.spinquant_quantization_grid import _spinquant_attention_forward, _model_layers
    from grid_baselines.flatquant_model import resolve_attention_runtime
    for layer in _model_layers(model):
        attention = layer.self_attn
        resolve_attention_runtime(attention)
        attention._spinquant_k_bits = config.activation_bits
        attention._spinquant_k_symmetric = config.activation_symmetric
        attention._spinquant_k_group_size = -1
        attention._spinquant_k_clip_ratio = config.activation_clip_ratio
        if not getattr(attention, "_spinquant_qk_patched", False):
            attention.forward = types.MethodType(
                _spinquant_attention_forward, attention
            )
            attention._spinquant_qk_patched = True

    return bank


def calibrate_spinquant_cross_entropy(
    model,
    input_ids: list[torch.Tensor],
    rotations: SpinQuantRotations,
    *,
    config: SpinQuantCalibrationConfig,
    device: torch.device | str,
) -> tuple[SpinQuantRotations, list[float]]:
    """Optimize R1/R2 with the paper-style quantized-network CE objective."""

    if not input_ids:
        raise ValueError("input_ids must be non-empty")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    actual_r2_steps = config.r2_steps if config.r2_steps is not None else config.r1_steps
    total_steps = max(config.r1_steps, actual_r2_steps)
    if total_steps <= 0:
        return rotations, []

    device = torch.device(device)
    model = model.to(device).eval()
    bank = install_spinquant_calibration_wrappers(model, rotations, config)
    r1_work = rotations.R1.to(device=device, dtype=torch.float32)
    r2_work = {
        idx: value.to(device=device, dtype=torch.float32)
        for idx, value in rotations.R2.items()
    }
    history: list[float] = []
    sample_count = len(input_ids)
    for step in range(total_steps):
        generator = torch.Generator().manual_seed(step)
        count = min(config.batch_size, sample_count)
        indices = torch.randperm(sample_count, generator=generator)[:count].tolist()
        batch = torch.cat([input_ids[idx].to(device) for idx in indices], dim=0)

        train_r1 = step < config.r1_steps
        train_r2 = step < actual_r2_steps

        r1_var = r1_work.detach().requires_grad_(train_r1)
        r2_vars = {
            idx: value.detach().requires_grad_(train_r2)
            for idx, value in r2_work.items()
        }
        bank.r1 = r1_var
        bank.r2 = r2_vars
        outputs = model(input_ids=batch, use_cache=False)
        logits = outputs.logits.float()
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
        )
        loss.backward()
        with torch.no_grad():
            if train_r1 and r1_var.grad is not None:
                r1_work = cayley_update(
                    r1_var.detach(),
                    r1_var.grad.detach(),
                    step_size=config.learning_rate,
                )
            if train_r2:
                for idx, r2_var in r2_vars.items():
                    if r2_var.grad is not None:
                        r2_work[idx] = cayley_update(
                            r2_var.detach(),
                            r2_var.grad.detach(),
                            step_size=config.learning_rate,
                        )
        history.append(float(loss.detach().cpu()))

    return (
        SpinQuantRotations(
            R1=r1_work.detach().cpu().double(),
            R2={idx: value.detach().cpu().double() for idx, value in r2_work.items()},
        ),
        history,
    )


@torch.no_grad()
def capture_spinquant_layer_inputs(
    layer: torch.nn.Module,
    inputs: list[torch.Tensor],
    block_kwargs: list[dict],
    *,
    device: torch.device | str,
) -> tuple[dict[str, list[torch.Tensor]], list[torch.Tensor]]:
    """Capture per-linear calibration inputs and next-layer hidden states."""

    if len(inputs) != len(block_kwargs) or not inputs:
        raise ValueError("inputs and block_kwargs must be non-empty and aligned")
    device = torch.device(device)
    layer = layer.to(device).eval()
    names = {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "up_proj": layer.mlp.up_proj,
        "gate_proj": layer.mlp.gate_proj,
        "down_proj": layer.mlp.down_proj,
    }
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    handles = []
    for name, module in names.items():
        handles.append(
            module.register_forward_pre_hook(
                lambda _module, args, name=name: captured[name].append(
                    args[0].detach().cpu()
                )
            )
        )
    outputs: list[torch.Tensor] = []
    try:
        for hidden, kwargs in zip(inputs, block_kwargs):
            local_kwargs = {
                key: _move_tree(value, device) for key, value in kwargs.items()
            }
            output = layer(hidden.to(device), **local_kwargs)
            outputs.append((output[0] if isinstance(output, tuple) else output).cpu())
    finally:
        for handle in handles:
            handle.remove()
    return captured, outputs


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


def calibrate_spinquant_layer_rotations(
    layer: torch.nn.Module,
    captured: dict[str, list[torch.Tensor]],
    *,
    r1: torch.Tensor,
    r2: torch.Tensor,
    config: SpinQuantCalibrationConfig,
    device: torch.device | str,
    train_r1: bool,
    train_r2: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Optimize SpinQuant rotations with a local linear reconstruction loss."""

    if not train_r1 and not train_r2:
        return r1, r2, []
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    steps = config.r1_steps if train_r1 and not train_r2 else config.r2_steps
    if steps <= 0:
        return r1, r2, []

    device = torch.device(device)
    dtype = torch.float32
    layer = layer.to(device).eval()
    for parameter in layer.parameters():
        parameter.requires_grad = False
    r1_work = r1.to(device=device, dtype=dtype).detach()
    r2_work = r2.to(device=device, dtype=dtype).detach()
    history: list[float] = []
    sample_count = len(next(iter(captured.values())))

    for step in range(steps):
        generator = torch.Generator().manual_seed(step)
        count = min(config.batch_size, sample_count)
        indices = torch.randperm(sample_count, generator=generator)[:count].tolist()
        r1_var = r1_work.detach().requires_grad_(train_r1)
        r2_var = r2_work.detach().requires_grad_(train_r2)
        loss = _spinquant_layer_loss(layer, captured, indices, r1_var, r2_var, config, device)
        loss.backward()
        with torch.no_grad():
            if train_r1 and r1_var.grad is not None:
                r1_work = cayley_update(
                    r1_var.detach(),
                    r1_var.grad.detach(),
                    step_size=config.learning_rate,
                )
            if train_r2 and r2_var.grad is not None:
                r2_work = cayley_update(
                    r2_var.detach(),
                    r2_var.grad.detach(),
                    step_size=config.learning_rate,
                )
        history.append(float(loss.detach().cpu()))
    return r1_work.cpu().double(), r2_work.cpu().double(), history


def _spinquant_layer_loss(
    layer: torch.nn.Module,
    captured: dict[str, list[torch.Tensor]],
    indices: list[int],
    r1: torch.Tensor,
    r2: torch.Tensor,
    config: SpinQuantCalibrationConfig,
    device: torch.device,
) -> torch.Tensor:
    losses = []
    for name in ("q_proj", "k_proj", "up_proj", "gate_proj"):
        module = _named_linear(layer, name)
        values = _sample_batch(captured[name], indices, device, r1.dtype)
        target = F.linear(values, module.weight.to(r1.dtype), _bias(module, r1.dtype))
        rotated_values = _quantize_activation(values.matmul(r1), config)
        rotated_weight = module.weight.to(r1.dtype).matmul(r1)
        losses.append(
            _linear_weight_loss(
                rotated_values,
                target,
                rotated_weight,
                _bias(module, r1.dtype),
                config,
            )
        )

    v_proj = layer.self_attn.v_proj
    v_values = _sample_batch(captured["v_proj"], indices, device, r1.dtype)
    v_target = _head_output_rotate_transpose(
        F.linear(v_values, v_proj.weight.to(r1.dtype), _bias(v_proj, r1.dtype)),
        r2,
    )
    v_weight = _rotate_v_weight(v_proj.weight.to(r1.dtype), r1, r2)
    losses.append(
        _linear_weight_loss(
            _quantize_activation(v_values.matmul(r1), config),
            v_target,
            v_weight,
            _rotate_bias_head_transpose(_bias(v_proj, r1.dtype), r2),
            config,
        )
    )

    o_proj = layer.self_attn.o_proj
    o_values = _sample_batch(captured["o_proj"], indices, device, r1.dtype)
    o_target = F.linear(o_values, o_proj.weight.to(r1.dtype), _bias(o_proj, r1.dtype))
    o_target = o_target.matmul(r1)
    o_values = _quantize_activation(_head_output_rotate_transpose(o_values, r2), config)
    losses.append(
        _linear_weight_loss(
            o_values,
            o_target,
            _rotate_o_weight(o_proj.weight.to(r1.dtype), r1, r2),
            _rotate_bias_left(_bias(o_proj, r1.dtype), r1),
            config,
        )
    )

    down_proj = layer.mlp.down_proj
    down_values = _sample_batch(captured["down_proj"], indices, device, r1.dtype)
    down_target = F.linear(
        down_values,
        down_proj.weight.to(r1.dtype),
        _bias(down_proj, r1.dtype),
    ).matmul(r1)
    losses.append(
        _linear_weight_loss(
            _quantize_activation(down_values, config),
            down_target,
            r1.t().matmul(down_proj.weight.to(r1.dtype)),
            _rotate_bias_left(_bias(down_proj, r1.dtype), r1),
            config,
        )
    )
    return torch.stack(losses).mean()


def _named_linear(layer: torch.nn.Module, name: str) -> torch.nn.Linear:
    parent = layer.self_attn if name.endswith("_proj") and name[0] in "qkvo" else layer.mlp
    return getattr(parent, name)


def _bias(module: torch.nn.Linear, dtype: torch.dtype) -> torch.Tensor | None:
    if module.bias is None:
        return None
    return module.bias.to(dtype=dtype)


def summarize_history(history: list[float]) -> dict[str, float | int | None]:
    if not history:
        return {"steps": 0, "initial_loss": None, "final_loss": None}
    return {
        "steps": len(history),
        "initial_loss": history[0],
        "final_loss": history[-1],
        "min_loss": min(history),
        "mean_loss": math.fsum(history) / len(history),
    }
