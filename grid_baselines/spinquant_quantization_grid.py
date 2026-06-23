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
import types

import torch

from .attention_runtime import resolve_attention_runtime
from .transformed_linear import (
    ActivationQuantizedLinear,
    SpinQuantHadamardLinear,
    TransformAwareLinear,
    apply_factorized_hadamard,
    fake_quantize_activation,
)
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


@dataclass
class SpinQuantR4:
    """Factorized R4 transform returned by official ``get_hadK``."""

    had_k: torch.Tensor | None
    k: int


def load_spinquant_r4(path: str | Path, *, width: int) -> SpinQuantR4:
    """Load ``had_K``/``K`` exported from SpinQuant's Hadamard utility."""

    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "r4" in raw:
        raw = raw["r4"]
    if not isinstance(raw, dict):
        raise ValueError("SpinQuant R4 artifact must be a mapping")
    k = int(raw.get("K", raw.get("k", 0)))
    had_k = raw.get("had_K", raw.get("had_k"))
    if had_k is not None:
        had_k = torch.as_tensor(had_k).detach().cpu()
    r4 = SpinQuantR4(had_k=had_k, k=k)
    _validate_r4(r4, width=width)
    return r4


def _validate_r4(r4: SpinQuantR4, *, width: int) -> None:
    k = r4.k
    if k <= 0 or width % k or (width // k) & (width // k - 1):
        raise ValueError("R4 requires intermediate_size / K to be a power of two")
    if k == 1:
        if r4.had_k is not None and r4.had_k.numel() != 0:
            raise ValueError("power-of-two R4 must use K=1 without had_K")
        return
    if r4.had_k is None or tuple(r4.had_k.shape) != (k, k):
        raise ValueError(f"R4 had_K must have shape ({k}, {k})")
    matrix = r4.had_k.to(torch.float64)
    expected = torch.eye(k, dtype=torch.float64) * k
    if not torch.allclose(matrix.t() @ matrix, expected, atol=1e-5, rtol=1e-5):
        raise ValueError("R4 had_K must satisfy had_K.T @ had_K = K I")


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


def _model_head_dim(model) -> int:
    layers = _model_layers(model)
    if not layers:
        raise ValueError("SpinQuant requires at least one Transformer layer")
    head_dim = int(layers[0].self_attn.head_dim)
    for layer_idx, layer in enumerate(layers[1:], start=1):
        if int(layer.self_attn.head_dim) != head_dim:
            raise ValueError(f"attention head_dim changes at layer {layer_idx}")
    return head_dim


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
    head_dim = _model_head_dim(model)
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
def apply_spinquant_r4(
    model,
    r4: SpinQuantR4 | None = None,
    *,
    weight_is_transformed: bool = False,
) -> SpinQuantR4:
    """Apply paper R4 to MLP down projections and its online transform."""
    width = int(model.config.intermediate_size)
    if r4 is None:
        if width & (width - 1):
            raise ValueError(
                "non-power-of-two intermediate_size requires a SpinQuant "
                "R4 artifact containing had_K and K"
            )
        r4 = SpinQuantR4(had_k=None, k=1)
    _validate_r4(r4, width=width)
    for layer in _model_layers(model):
        source = layer.mlp.down_proj
        if not isinstance(source, SpinQuantHadamardLinear):
            layer.mlp.down_proj = SpinQuantHadamardLinear(
                source,
                had_k=r4.had_k,
                k=r4.k,
                weight_is_transformed=weight_is_transformed,
            )
    return r4


def _spinquant_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    runtime = resolve_attention_runtime(self)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = runtime.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    query_states = apply_factorized_hadamard(
        query_states, had_k=None, k=1
    )
    key_states = apply_factorized_hadamard(key_states, had_k=None, k=1)
    if self._spinquant_k_group_size == -1:
        batch, heads, sequence, head_dim = key_states.shape
        token_keys = key_states.transpose(1, 2).reshape(batch, sequence, -1)
        token_keys = fake_quantize_activation(
            token_keys,
            bits=self._spinquant_k_bits,
            symmetric=self._spinquant_k_symmetric,
            group_size=-1,
            clip_ratio=self._spinquant_k_clip_ratio,
        )
        key_states = token_keys.reshape(
            batch, sequence, heads, head_dim
        ).transpose(1, 2)
    else:
        key_states = fake_quantize_activation(
            key_states,
            bits=self._spinquant_k_bits,
            symmetric=self._spinquant_k_symmetric,
            group_size=self.head_dim,
            clip_ratio=self._spinquant_k_clip_ratio,
        )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )
    attention_interface = runtime.attention_interfaces.get_interface(
        self.config._attn_implementation, runtime.eager_attention_forward
    )
    attention_kwargs = runtime.extra_attention_kwargs(self)
    attention_kwargs.update(kwargs)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **attention_kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def add_spinquant_k_cache_quantization(
    model,
    *,
    bits: int,
    symmetric: bool = False,
    group_size: int = -1,
    clip_ratio: float = 1.0,
) -> None:
    """Install official post-RoPE R3 and K-cache fake quantization."""

    if bits >= 16:
        return
    head_dim = _model_head_dim(model)
    if head_dim & (head_dim - 1):
        raise ValueError("SpinQuant K-cache quantization requires power-of-two head_dim")
    if group_size not in {-1, head_dim}:
        raise ValueError("K-cache group_size must be -1 or head_dim")
    for layer in _model_layers(model):
        attention = layer.self_attn
        resolve_attention_runtime(attention)
        attention._spinquant_k_bits = bits
        attention._spinquant_k_symmetric = symmetric
        attention._spinquant_k_group_size = group_size
        attention._spinquant_k_clip_ratio = clip_ratio
        if not getattr(attention, "_spinquant_qk_patched", False):
            attention.forward = types.MethodType(
                _spinquant_attention_forward, attention
            )
            attention._spinquant_qk_patched = True


@torch.no_grad()
def add_spinquant_activation_quantization(
    model,
    *,
    bits: int,
    symmetric: bool = False,
    group_size: int = -1,
    clip_ratio: float = 1.0,
    v_bits: int = 16,
    v_symmetric: bool = False,
    v_clip_ratio: float = 1.0,
    int8_down_proj: bool = False,
) -> None:
    """Configure the activation path used by the official SpinQuant PTQ code.

    Inputs are quantized per token (or per group).  ``o_proj`` uses one group
    per attention head, ``down_proj`` preserves the number of groups used in
    the residual width, ``v_proj`` optionally quantizes its output per head,
    and ``lm_head`` remains in floating point.
    """

    if bits >= 16 and v_bits >= 16:
        return
    head_dim = _model_head_dim(model)
    down_group_size = group_size
    if group_size > 0 and model.config.intermediate_size % group_size != 0:
        group_count = model.config.hidden_size // group_size
        if group_count * group_size != model.config.hidden_size:
            raise ValueError(
                "activation group_size must divide the model hidden size"
            )
        if model.config.intermediate_size % group_count != 0:
            raise ValueError(
                "cannot preserve activation group count for down_proj"
            )
        down_group_size = model.config.intermediate_size // group_count
    replacements = []
    for name, module in model.named_modules():
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        if isinstance(module, SpinQuantHadamardLinear):
            module.activation_bits = 8 if int8_down_proj else bits
            module.activation_symmetric = symmetric
            module.activation_group_size = down_group_size
            module.activation_clip_ratio = clip_ratio
            continue
        if isinstance(module, TransformAwareLinear):
            module.activation_bits = bits
            module.activation_symmetric = symmetric
            module.activation_group_size = group_size
            module.activation_clip_ratio = clip_ratio
            continue
        if isinstance(module, torch.nn.Linear) and not isinstance(
            module, ActivationQuantizedLinear
        ):
            replacements.append((name, module))
    for name, module in replacements:
        input_bits = 8 if int8_down_proj and name.endswith("down_proj") else bits
        input_group_size = group_size
        if name.endswith("o_proj"):
            input_group_size = head_dim
        elif name.endswith("down_proj"):
            input_group_size = down_group_size
        output_bits = v_bits if name.endswith("v_proj") else 16
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            child_name,
            ActivationQuantizedLinear(
                module,
                bits=input_bits,
                symmetric=symmetric,
                group_size=input_group_size,
                clip_ratio=clip_ratio,
                output_bits=output_bits,
                output_symmetric=v_symmetric,
                output_group_size=head_dim,
                output_clip_ratio=v_clip_ratio,
            ),
        )


def cayley_update(
    rotation: torch.Tensor,
    gradient: torch.Tensor,
    *,
    step_size: float,
) -> torch.Tensor:
    """One Cayley-SGD update from Eq. 3-4 of the SpinQuant paper."""

    projected = gradient @ rotation.t()
    projected = projected - 0.5 * (
        rotation @ rotation.t() @ projected
    )
    skew = projected - projected.t()
    identity = torch.eye(
        rotation.shape[0],
        device=rotation.device,
        dtype=rotation.dtype,
    )
    left = identity - 0.5 * step_size * skew
    right = (identity + 0.5 * step_size * skew) @ rotation
    return torch.linalg.solve(left, right)


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
