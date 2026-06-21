'''SpinQuant no-had model reparameterization and weight grid.

SpinQuant does not rotate each linear layer independently. It learns one
residual-stream rotation R1 and one head-dimension rotation R2 per attention
layer. These rotations are absorbed into paired Transformer weights before
weight quantization, preserving the floating-point network without adding an
online rotation to the no-had inference path.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .vanilla_quantization_grid import (
    VanillaQuantizationGrid,
    build_vanilla_quantization_grid,
)


@dataclass
class SpinQuantRotations:
    '''Learned SpinQuant no-had rotations in the official checkpoint layout.'''

    R1: torch.Tensor
    R2: dict[int, torch.Tensor]


@dataclass
class SpinQuantQuantizationGrid(VanillaQuantizationGrid):
    '''Uniform weight grid applied after model-level SpinQuant rotation.'''


def _orthogonal_matrix(size: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(size, size, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return (q * signs).contiguous()


def random_spinquant_rotations(
    *,
    num_layers: int,
    hidden_size: int,
    head_dim: int,
    seed: int = 0,
) -> SpinQuantRotations:
    '''Generate random orthogonal R1/R2 rotations for SpinQuant smoke runs.'''

    if num_layers <= 0:
        raise ValueError('num_layers must be positive')
    if hidden_size <= 0:
        raise ValueError('hidden_size must be positive')
    if head_dim <= 0:
        raise ValueError('head_dim must be positive')

    generator = torch.Generator().manual_seed(seed)
    return SpinQuantRotations(
        R1=_orthogonal_matrix(hidden_size, generator),
        R2={
            layer_idx: _orthogonal_matrix(head_dim, generator)
            for layer_idx in range(num_layers)
        },
    )


def _as_matrix(value, name: str) -> torch.Tensor:
    matrix = torch.as_tensor(value).detach().cpu()
    if matrix.dim() != 2:
        raise ValueError(f'{name} must be a matrix, got shape {tuple(matrix.shape)}')
    return matrix


def load_spinquant_rotations(
    path: str | Path,
    *,
    num_layers: int,
    hidden_size: int,
    head_dim: int,
) -> SpinQuantRotations:
    '''Load the official ``R1`` and per-layer ``self_attn.R2`` checkpoint.'''

    raw = torch.load(path, map_location='cpu')
    if isinstance(raw, dict) and 'state_dict' in raw:
        raw = raw['state_dict']
    if isinstance(raw, dict) and 'rotations' in raw:
        raw = raw['rotations']
    if not isinstance(raw, dict):
        raise ValueError('SpinQuant rotation checkpoint must contain a state dict')

    r1_value = raw.get('R1', raw.get('R1.weight'))
    if r1_value is None:
        raise ValueError('SpinQuant checkpoint is missing R1')
    r1 = _as_matrix(r1_value, 'R1')
    if r1.shape != (hidden_size, hidden_size):
        raise ValueError(
            f'R1 must have shape ({hidden_size}, {hidden_size}), got {tuple(r1.shape)}'
        )

    rotations: dict[int, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        base = f'model.layers.{layer_idx}.self_attn.R2'
        value = raw.get(base, raw.get(f'{base}.weight'))
        if value is None:
            raise ValueError(f'SpinQuant checkpoint is missing {base}')
        r2 = _as_matrix(value, base)
        if r2.shape != (head_dim, head_dim):
            raise ValueError(
                f'{base} must have shape ({head_dim}, {head_dim}), '
                f'got {tuple(r2.shape)}'
            )
        rotations[layer_idx] = r2
    return SpinQuantRotations(R1=r1, R2=rotations)


def _check_orthogonal(
    rotation: torch.Tensor,
    *,
    name: str,
    tolerance: float,
) -> None:
    check = rotation.float()
    identity = torch.eye(check.shape[0], device=check.device, dtype=check.dtype)
    error = check.t().matmul(check) - identity
    max_error = float(error.abs().max().item())
    if max_error > tolerance:
        raise ValueError(
            f'{name} must be orthogonal; max |R.T R - I| is {max_error:.6g}'
        )


@torch.no_grad()
def _fuse_rms_scale(norm: torch.nn.Module, linears: list[torch.nn.Linear]) -> None:
    if not hasattr(norm, 'weight'):
        raise TypeError(f'{type(norm).__name__} does not expose a weight parameter')
    scale = norm.weight.detach()
    for linear in linears:
        weight = linear.weight.detach()
        original = weight.to(dtype=torch.float64)
        fused = original * scale.to(device=weight.device, dtype=torch.float64)
        linear.weight.data = fused.to(dtype=weight.dtype)

        bias = getattr(norm, 'bias', None)
        if bias is not None:
            correction = original.matmul(
                bias.detach().to(device=weight.device, dtype=torch.float64)
            )
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(
                        linear.out_features,
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                )
            linear.bias.data = (
                linear.bias.detach().to(torch.float64) + correction
            ).to(dtype=weight.dtype)
    norm.weight.data = torch.ones_like(norm.weight)
    if getattr(norm, 'bias', None) is not None:
        norm.bias.data = torch.zeros_like(norm.bias)


def _model_layers(model) -> list[torch.nn.Module]:
    try:
        return list(model.model.layers)
    except AttributeError as exc:
        raise TypeError(
            'SpinQuant currently supports LLaMA/Mistral-style models exposing '
            'model.layers, self_attn, and mlp projections'
        ) from exc


@torch.no_grad()
def fuse_spinquant_norms(model) -> None:
    '''Fuse pre-norm scales into adjacent linears as required by SpinQuant.'''

    for layer in _model_layers(model):
        _fuse_rms_scale(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        _fuse_rms_scale(
            layer.post_attention_layernorm,
            [layer.mlp.up_proj, layer.mlp.gate_proj],
        )
    _fuse_rms_scale(model.model.norm, [model.lm_head])


def _work_rotation(
    rotation: torch.Tensor,
    reference: torch.Tensor,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    return rotation.to(device=reference.device, dtype=work_dtype)


@torch.no_grad()
def _right_rotate(linear, rotation, work_dtype) -> None:
    weight = linear.weight.detach()
    rotated = weight.to(work_dtype).matmul(
        _work_rotation(rotation, weight, work_dtype)
    )
    linear.weight.data = rotated.to(dtype=weight.dtype)


@torch.no_grad()
def _left_rotate(linear, rotation, work_dtype) -> None:
    weight = linear.weight.detach()
    matrix = _work_rotation(rotation, weight, work_dtype)
    rotated = matrix.t().matmul(weight.to(work_dtype))
    linear.weight.data = rotated.to(dtype=weight.dtype)
    if linear.bias is not None:
        bias = linear.bias.detach()
        linear.bias.data = matrix.t().matmul(bias.to(work_dtype)).to(bias.dtype)


@torch.no_grad()
def _rotate_output_headwise(linear, rotation, work_dtype) -> None:
    '''Apply R2.T independently to every output head of V projection.'''

    weight = linear.weight.detach()
    head_dim = rotation.shape[0]
    if linear.out_features % head_dim != 0:
        raise ValueError(
            f'{linear.out_features} output features are not divisible by '
            f'R2 head dimension {head_dim}'
        )
    matrix = _work_rotation(rotation, weight, work_dtype)
    grouped = weight.to(work_dtype).reshape(-1, head_dim, linear.in_features)
    rotated = torch.matmul(matrix.t().unsqueeze(0), grouped)
    linear.weight.data = rotated.reshape_as(weight).to(dtype=weight.dtype)
    if linear.bias is not None:
        bias = linear.bias.detach()
        grouped_bias = bias.to(work_dtype).reshape(-1, head_dim, 1)
        rotated_bias = torch.matmul(matrix.t().unsqueeze(0), grouped_bias)
        linear.bias.data = rotated_bias.reshape_as(bias).to(dtype=bias.dtype)


@torch.no_grad()
def _rotate_input_headwise(linear, rotation, work_dtype) -> None:
    '''Apply R2 independently to every input head of O projection.'''

    weight = linear.weight.detach()
    head_dim = rotation.shape[0]
    if linear.in_features % head_dim != 0:
        raise ValueError(
            f'{linear.in_features} input features are not divisible by '
            f'R2 head dimension {head_dim}'
        )
    matrix = _work_rotation(rotation, weight, work_dtype)
    grouped = weight.to(work_dtype).reshape(linear.out_features, -1, head_dim)
    rotated = grouped.matmul(matrix)
    linear.weight.data = rotated.reshape_as(weight).to(dtype=weight.dtype)


@torch.no_grad()
def apply_spinquant_no_had(
    model,
    rotations: SpinQuantRotations,
    *,
    work_dtype: torch.dtype = torch.float64,
    orthogonality_tolerance: float = 1e-4,
    fuse_norms: bool = True,
) -> None:
    '''Absorb learned R1/R2 into a LLaMA/Mistral-style model in place.'''

    if work_dtype not in {torch.float32, torch.float64}:
        raise ValueError('SpinQuant work_dtype must be float32 or float64')
    layers = _model_layers(model)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    if rotations.R1.shape != (hidden_size, hidden_size):
        raise ValueError('R1 shape does not match model hidden size')
    if set(rotations.R2) != set(range(len(layers))):
        raise ValueError('R2 rotations must contain exactly one entry per layer')

    first_weight = model.model.embed_tokens.weight
    r1_check = rotations.R1.to(device=first_weight.device)
    _check_orthogonal(
        r1_check,
        name='R1',
        tolerance=orthogonality_tolerance,
    )
    for layer_idx, r2 in rotations.R2.items():
        if r2.shape != (head_dim, head_dim):
            raise ValueError(f'R2 for layer {layer_idx} has the wrong shape')
        reference = layers[layer_idx].self_attn.v_proj.weight
        _check_orthogonal(
            r2.to(device=reference.device),
            name=f'R2 layer {layer_idx}',
            tolerance=orthogonality_tolerance,
        )

    if model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False

    if fuse_norms:
        fuse_spinquant_norms(model)

    embed = model.model.embed_tokens
    embed_weight = embed.weight.detach()
    embed.weight.data = embed_weight.to(work_dtype).matmul(
        _work_rotation(rotations.R1, embed_weight, work_dtype)
    ).to(embed_weight.dtype)
    _right_rotate(model.lm_head, rotations.R1, work_dtype)

    for layer_idx, layer in enumerate(layers):
        r2 = rotations.R2[layer_idx]
        for linear in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.mlp.up_proj,
            layer.mlp.gate_proj,
        ):
            _right_rotate(linear, rotations.R1, work_dtype)
        _left_rotate(layer.self_attn.o_proj, rotations.R1, work_dtype)
        _left_rotate(layer.mlp.down_proj, rotations.R1, work_dtype)
        _rotate_output_headwise(layer.self_attn.v_proj, r2, work_dtype)
        _rotate_input_headwise(layer.self_attn.o_proj, r2, work_dtype)


@torch.no_grad()
def build_spinquant_quantization_grid(
    weights: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    scheme: str = 'asymmetric',
    eps: float = 1e-8,
) -> SpinQuantQuantizationGrid:
    '''Build the ordinary uniform grid on an already rotated model weight.'''

    grid = build_vanilla_quantization_grid(
        weights,
        bits,
        group_size,
        scheme=scheme,
        eps=eps,
    )
    return SpinQuantQuantizationGrid(**vars(grid))


def build_symmetric_spinquant_quantization_grid(
    weights: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    eps: float = 1e-8,
) -> SpinQuantQuantizationGrid:
    return build_spinquant_quantization_grid(
        weights,
        bits,
        group_size,
        scheme='symmetric',
        eps=eps,
    )


def build_asymmetric_spinquant_quantization_grid(
    weights: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    eps: float = 1e-8,
) -> SpinQuantQuantizationGrid:
    return build_spinquant_quantization_grid(
        weights,
        bits,
        group_size,
        scheme='asymmetric',
        eps=eps,
    )
