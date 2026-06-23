"""Paper-compatible FlatQuant affine transforms for assignment-method runs.

This module implements the central FlatQuant relation

    ``Y = X W^T = (X P) (W P^{-T})^T``

with Kronecker factors and the optional pair-wise diagonal scale.  The model
stores transformed weights, applies the matching activation transform online,
and exposes transformed inputs to every assignment method through the shared
collector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types

import torch
import torch.nn as nn

from .attention_runtime import resolve_attention_runtime
from .transformed_linear import (
    KroneckerTransform,
    TransformAwareLinear,
    fake_quantize_activation,
)


@dataclass
class FlatQuantLinearSpec:
    input_transform: KroneckerTransform
    weight_transform: KroneckerTransform | None = None
    output_head_transform: torch.Tensor | None = None


def _get_parent_module(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


def _expected_flatquant_linears(model: nn.Module) -> set[str]:
    try:
        layers = list(model.model.layers)
    except AttributeError:
        return set()
    expected = set()
    for layer_idx, layer in enumerate(layers):
        base = f"model.layers.{layer_idx}"
        for owner_name, owner, projections in (
            (
                "self_attn",
                getattr(layer, "self_attn", None),
                ("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
            (
                "mlp",
                getattr(layer, "mlp", None),
                ("up_proj", "gate_proj", "down_proj"),
            ),
        ):
            if owner is None:
                continue
            for projection in projections:
                module = getattr(owner, projection, None)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"{base}.{owner_name}.{projection} must be nn.Linear"
                    )
                expected.add(f"{base}.{owner_name}.{projection}")
    return expected


def _first_present(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _parse_transform(value: dict, layer_name: str) -> KroneckerTransform:
    left = _first_present(
        value,
        ("left", "matrix_left", "p1", "P1", "transform_left"),
    )
    right = _first_present(
        value,
        ("right", "matrix_right", "p2", "P2", "transform_right"),
    )
    diagonal = _first_present(
        value,
        ("diagonal", "diag", "diag_scale", "scales", "c"),
    )
    dense = _first_present(value, ("matrix", "p", "P"))
    if dense is not None and (left is None or right is None):
        dense = torch.as_tensor(dense)
        left = torch.ones(1, 1, dtype=dense.dtype)
        right = dense
    if left is None or right is None:
        raise ValueError(
            f"FlatQuant transform for {layer_name!r} must provide "
            "matrix_left/matrix_right (or a dense matrix)"
        )
    return KroneckerTransform(
        left=torch.as_tensor(left),
        right=torch.as_tensor(right),
        diagonal=torch.as_tensor(diagonal) if diagonal is not None else None,
    )


def _parse_linear_spec(value: dict, layer_name: str) -> FlatQuantLinearSpec:
    input_transform = _parse_transform(value, layer_name)
    weight_transform = None
    if "weight_left" in value or "weight_right" in value:
        weight_transform = KroneckerTransform(
            left=torch.as_tensor(value["weight_left"]),
            right=torch.as_tensor(value["weight_right"]),
            diagonal=(
                torch.as_tensor(value["weight_diagonal"])
                if "weight_diagonal" in value
                else None
            ),
        )
    output_head_transform = (
        torch.as_tensor(value["output_head_transform"])
        if "output_head_transform" in value
        else None
    )
    return FlatQuantLinearSpec(
        input_transform=input_transform,
        weight_transform=weight_transform,
        output_head_transform=output_head_transform,
    )


def _official_transform(
    state: dict,
    prefix: str,
    *,
    diagonal: bool = True,
) -> KroneckerTransform | None:
    left = state.get(f"{prefix}.matrix_left")
    right = state.get(f"{prefix}.matrix_right")
    dense = state.get(f"{prefix}.matrix")
    if dense is not None and (left is None or right is None):
        left = torch.ones(1, 1, dtype=torch.as_tensor(dense).dtype)
        right = dense
    if left is None or right is None:
        return None
    diag_value = state.get(f"{prefix}.diag_scale") if diagonal else None
    return KroneckerTransform(
        left=torch.as_tensor(left),
        right=torch.as_tensor(right),
        diagonal=(
            torch.as_tensor(diag_value).reshape(-1)
            if diag_value is not None
            else None
        ),
    )


def _sigmoid_clip(state: dict, prefix: str) -> dict[str, torch.Tensor]:
    max_value = state.get(f"{prefix}.clip_factor_w_max")
    min_value = state.get(f"{prefix}.clip_factor_w_min")
    result = {}
    if max_value is not None and min_value is not None:
        result.update(
            weight_clip_max=torch.sigmoid(torch.as_tensor(max_value)),
            weight_clip_min=torch.sigmoid(torch.as_tensor(min_value)),
        )
    act_max = state.get(f"{prefix}.act_quantizer.clip_factor_a_max")
    act_min = state.get(f"{prefix}.act_quantizer.clip_factor_a_min")
    if act_max is not None and act_min is not None:
        result.update(
            activation_clip_max=torch.sigmoid(torch.as_tensor(act_max)),
            activation_clip_min=torch.sigmoid(torch.as_tensor(act_min)),
        )
    return result


def _parse_official_flat_matrices(
    raw: dict,
) -> tuple[dict[str, FlatQuantLinearSpec], dict[str, dict[str, torch.Tensor]]]:
    """Convert official per-block ``flat_matrices.pth`` states.

    The attention output transform is represented as one equivalent
    Kronecker input transform on ``o_proj``: ``o_trans ⊗ vcache_trans``.
    This is algebraically identical before K/V-cache quantization and keeps the
    linear visible to the repository's assignment methods.
    """

    transforms: dict[str, FlatQuantLinearSpec] = {}
    clips: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in raw.items():
        if not isinstance(state, dict):
            continue
        try:
            layer_idx = int(layer_key)
        except (TypeError, ValueError):
            continue
        base = f"model.layers.{layer_idx}"

        ln_transform = _official_transform(state, "self_attn.ln_trans")
        if ln_transform is not None:
            for projection in ("q_proj", "k_proj", "v_proj"):
                name = f"{base}.self_attn.{projection}"
                transforms[name] = FlatQuantLinearSpec(ln_transform)
                parsed_clip = _sigmoid_clip(state, f"self_attn.{projection}")
                if parsed_clip:
                    clips[name] = parsed_clip

        up_gate = _official_transform(state, "mlp.up_gate_trans")
        if up_gate is not None:
            for projection in ("up_proj", "gate_proj"):
                name = f"{base}.mlp.{projection}"
                transforms[name] = FlatQuantLinearSpec(up_gate)
                parsed_clip = _sigmoid_clip(state, f"mlp.{projection}")
                if parsed_clip:
                    clips[name] = parsed_clip

        down = _official_transform(state, "mlp.down_trans")
        if down is not None:
            name = f"{base}.mlp.down_proj"
            transforms[name] = FlatQuantLinearSpec(down)
            parsed_clip = _sigmoid_clip(state, "mlp.down_proj")
            if parsed_clip:
                clips[name] = parsed_clip

        o_matrix = state.get("self_attn.o_trans.matrix")
        v_matrix = state.get("self_attn.vcache_trans.matrix")
        if o_matrix is not None and v_matrix is not None:
            o_matrix = torch.as_tensor(o_matrix)
            v_matrix = torch.as_tensor(v_matrix)
            identity = torch.eye(
                v_matrix.shape[0],
                dtype=v_matrix.dtype,
            )
            transforms[f"{base}.self_attn.o_proj"] = FlatQuantLinearSpec(
                input_transform=KroneckerTransform(
                    left=o_matrix,
                    right=identity,
                ),
                weight_transform=KroneckerTransform(
                    left=o_matrix,
                    right=v_matrix,
                ),
            )
            v_name = f"{base}.self_attn.v_proj"
            if v_name in transforms:
                transforms[v_name].output_head_transform = v_matrix
            parsed_clip = _sigmoid_clip(
                state,
                "self_attn.o_proj",
            )
            if parsed_clip:
                clips[f"{base}.self_attn.o_proj"] = parsed_clip
    return transforms, clips


def load_flatquant_transforms(
    path: str | Path,
) -> tuple[
    dict[str, FlatQuantLinearSpec],
    dict[str, dict[str, torch.Tensor]],
]:
    """Load normalized per-linear FlatQuant transforms.

    Supported normalized format:

    ``{"layers": {linear_name: {"matrix_left": ..., "matrix_right": ...}}}``

    A top-level mapping without ``layers`` is accepted as well.  Weight clipping
    values are returned separately for grid construction.
    """

    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("FlatQuant parameter file must contain a dict")
    official, official_clips = _parse_official_flat_matrices(raw)
    if official:
        return official, official_clips

    entries = raw.get("layers", raw)
    if not isinstance(entries, dict):
        raise ValueError("FlatQuant 'layers' entry must be a dict")

    transforms: dict[str, FlatQuantLinearSpec] = {}
    clips: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in entries.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        transforms[name] = _parse_linear_spec(value, name)
        clip = _first_present(
            value,
            ("weight_clip", "alpha_w", "clip", "clip_ratio"),
        )
        if clip is not None:
            clip_tensor = torch.as_tensor(clip)
            clips[name] = {
                "weight_clip_max": clip_tensor,
                "weight_clip_min": clip_tensor,
            }
        for key in (
            "weight_clip_max",
            "weight_clip_min",
            "activation_clip_max",
            "activation_clip_min",
        ):
            if key in value:
                clips.setdefault(name, {})[key] = torch.as_tensor(value[key])
    if not transforms:
        raise ValueError(
            "No normalized per-linear FlatQuant transforms were found. "
            "Expected keys such as matrix_left/matrix_right."
        )
    return transforms, clips


def load_flatquant_attention_transforms(
    path: str | Path,
) -> dict[str, torch.Tensor]:
    """Load post-RoPE Q/K transforms from official or normalized artifacts."""

    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("FlatQuant parameter file must contain a dict")
    normalized = raw.get("attention", {})
    if isinstance(normalized, dict) and normalized:
        return {
            str(name): torch.as_tensor(matrix)
            for name, matrix in normalized.items()
        }
    result = {}
    for layer_key, state in raw.items():
        if not isinstance(state, dict):
            continue
        try:
            layer_idx = int(layer_key)
        except (TypeError, ValueError):
            continue
        matrix = state.get("self_attn.kcache_trans.matrix")
        if matrix is not None:
            result[f"model.layers.{layer_idx}.self_attn"] = torch.as_tensor(matrix)
    return result


def load_flatquant_attention_clips(
    path: str | Path,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load learned Q/K/V cache clipping factors from FlatQuant artifacts."""

    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("FlatQuant parameter file must contain a dict")
    normalized = raw.get("attention_clips", {})
    if isinstance(normalized, dict) and normalized:
        return {
            str(name): {
                str(key): torch.as_tensor(value)
                for key, value in values.items()
            }
            for name, values in normalized.items()
        }
    result: dict[str, dict[str, torch.Tensor]] = {}
    for layer_key, state in raw.items():
        if not isinstance(state, dict):
            continue
        try:
            layer_idx = int(layer_key)
        except (TypeError, ValueError):
            continue
        name = f"model.layers.{layer_idx}.self_attn"
        for short_name in ("q", "k", "v"):
            prefix = f"self_attn.{short_name}_cache_quantizer"
            max_value = state.get(f"{prefix}.clip_factor_a_max")
            min_value = state.get(f"{prefix}.clip_factor_a_min")
            if max_value is not None and min_value is not None:
                result.setdefault(name, {}).update(
                    {
                        f"{short_name}_clip_max": torch.sigmoid(
                            torch.as_tensor(max_value)
                        ),
                        f"{short_name}_clip_min": torch.sigmoid(
                            torch.as_tensor(min_value)
                        ),
                    }
                )
    return result


def _flatquant_attention_forward(
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

    matrix = self._flatquant_kcache_matrix
    if matrix is not None:
        matrix = matrix.to(query_states)
        inverse_transpose = torch.linalg.inv(matrix.float()).t().to(query_states)
        query_states = query_states.matmul(inverse_transpose)
        key_states = key_states.matmul(matrix)
    query_states = fake_quantize_activation(
        query_states,
        bits=self._flatquant_q_bits,
        symmetric=self._flatquant_q_symmetric,
        group_size=self._flatquant_q_group_size,
        clip_ratio=self._flatquant_q_clip_ratio,
        clip_factor_max=self._flatquant_q_clip_max,
        clip_factor_min=self._flatquant_q_clip_min,
    )
    key_states = fake_quantize_activation(
        key_states,
        bits=self._flatquant_k_bits,
        symmetric=self._flatquant_k_symmetric,
        group_size=self._flatquant_k_group_size,
        clip_ratio=self._flatquant_k_clip_ratio,
        clip_factor_max=self._flatquant_k_clip_max,
        clip_factor_min=self._flatquant_k_clip_min,
    )
    value_states = fake_quantize_activation(
        value_states,
        bits=self._flatquant_v_bits,
        symmetric=self._flatquant_v_symmetric,
        group_size=self._flatquant_v_group_size,
        clip_ratio=self._flatquant_v_clip_ratio,
        clip_factor_max=self._flatquant_v_clip_max,
        clip_factor_min=self._flatquant_v_clip_min,
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


def apply_flatquant_attention_transforms(
    model: nn.Module,
    transforms: dict[str, torch.Tensor],
    *,
    q_bits: int = 16,
    k_bits: int = 16,
    v_bits: int = 16,
    q_symmetric: bool = False,
    k_symmetric: bool = False,
    v_symmetric: bool = False,
    q_group_size: int = -1,
    k_group_size: int = -1,
    v_group_size: int = -1,
    q_clip_ratio: float = 1.0,
    k_clip_ratio: float = 1.0,
    v_clip_ratio: float = 1.0,
    clips: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    """Install FlatQuant's post-RoPE cache transform and Q/K/V quantizers."""

    if q_bits >= 16 and k_bits >= 16 and v_bits >= 16:
        return
    selected = dict(transforms)
    clips = clips or {}
    if not selected and v_bits < 16 and q_bits >= 16 and k_bits >= 16:
        selected = {
            f"model.layers.{idx}.self_attn": None
            for idx, _ in enumerate(model.model.layers)
        }
    if (q_bits < 16 or k_bits < 16) and not selected:
        raise ValueError("FlatQuant Q/K quantization requires kcache transforms")
    for name, raw_matrix in selected.items():
        attention = model.get_submodule(name)
        resolve_attention_runtime(attention)
        matrix = None if raw_matrix is None else torch.as_tensor(raw_matrix)
        if matrix is not None and tuple(matrix.shape) != (
            attention.head_dim,
            attention.head_dim,
        ):
            raise ValueError(f"{name} kcache transform has the wrong shape")
        attention.register_buffer(
            "_flatquant_kcache_matrix",
            None if matrix is None else matrix.detach().clone(),
            persistent=False,
        )
        for prefix, bits, symmetric, group_size, clip_ratio in (
            ("q", q_bits, q_symmetric, q_group_size, q_clip_ratio),
            ("k", k_bits, k_symmetric, k_group_size, k_clip_ratio),
            ("v", v_bits, v_symmetric, v_group_size, v_clip_ratio),
        ):
            setattr(attention, f"_flatquant_{prefix}_bits", bits)
            setattr(attention, f"_flatquant_{prefix}_symmetric", symmetric)
            setattr(attention, f"_flatquant_{prefix}_group_size", group_size)
            setattr(attention, f"_flatquant_{prefix}_clip_ratio", clip_ratio)
            setattr(
                attention,
                f"_flatquant_{prefix}_clip_max",
                clips.get(name, {}).get(f"{prefix}_clip_max"),
            )
            setattr(
                attention,
                f"_flatquant_{prefix}_clip_min",
                clips.get(name, {}).get(f"{prefix}_clip_min"),
            )
        if not getattr(attention, "_flatquant_cache_patched", False):
            attention.forward = types.MethodType(
                _flatquant_attention_forward, attention
            )
            attention._flatquant_cache_patched = True


@torch.no_grad()
def apply_flatquant_transforms(
    model: nn.Module,
    transforms: dict[str, FlatQuantLinearSpec],
    *,
    weight_is_transformed: bool = False,
    strict: bool = True,
    activation_bits: int = 16,
    activation_symmetric: bool = False,
    activation_group_size: int = -1,
    activation_clip_ratio: float = 1.0,
    clips: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    """Replace selected linears with transform-aware FlatQuant linears."""

    modules = dict(model.named_modules())
    clips = clips or {}
    missing = []
    if strict:
        absent = _expected_flatquant_linears(model) - set(transforms)
        if absent:
            raise KeyError(
                "FlatQuant artifact is missing required linears: "
                + ", ".join(sorted(absent)[:8])
            )
    for name, raw_spec in transforms.items():
        spec = (
            raw_spec
            if isinstance(raw_spec, FlatQuantLinearSpec)
            else FlatQuantLinearSpec(raw_spec)
        )
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            missing.append(name)
            continue
        spec.input_transform.validate(module.in_features)
        if spec.weight_transform is not None:
            spec.weight_transform.validate(module.in_features)
        layer_clips = clips.get(name, {})
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            TransformAwareLinear(
                module,
                spec.input_transform,
                weight_transform=spec.weight_transform,
                output_head_transform=spec.output_head_transform,
                weight_is_transformed=weight_is_transformed,
                activation_bits=activation_bits,
                activation_symmetric=activation_symmetric,
                activation_group_size=activation_group_size,
                activation_clip_ratio=activation_clip_ratio,
                activation_clip_max=layer_clips.get("activation_clip_max"),
                activation_clip_min=layer_clips.get("activation_clip_min"),
            ),
        )
    if strict and missing:
        raise KeyError(
            "FlatQuant transforms reference missing/non-linear modules: "
            + ", ".join(missing[:8])
        )


def serialize_flatquant_transforms(
    transforms: dict[str, FlatQuantLinearSpec],
    clips: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    clips = clips or {}
    result = {}
    for name, raw_spec in transforms.items():
        spec = (
            raw_spec
            if isinstance(raw_spec, FlatQuantLinearSpec)
            else FlatQuantLinearSpec(raw_spec)
        )
        state = spec.input_transform.state_dict()
        if spec.weight_transform is not None:
            weight_state = spec.weight_transform.state_dict()
            state.update(
                {
                    f"weight_{key}": value
                    for key, value in weight_state.items()
                }
            )
        if spec.output_head_transform is not None:
            state["output_head_transform"] = (
                spec.output_head_transform.detach().cpu()
            )
        state.update(
            {
                key: value.detach().cpu()
                for key, value in clips.get(name, {}).items()
            }
        )
        result[name] = state
    return result
