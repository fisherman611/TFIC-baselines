"""Load Hugging Face checkpoints with model-level quantization transforms."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from grid_baselines import (
    apply_flatquant_attention_transforms,
    add_spinquant_k_cache_quantization,
    add_spinquant_activation_quantization,
    apply_flatquant_transforms,
)
from grid_baselines.flatquant_model import load_flatquant_transforms
from grid_baselines.flatquant_model import (
    load_flatquant_attention_clips,
    load_flatquant_attention_transforms,
)
from grid_baselines.spinquant_quantization_grid import (
    apply_spinquant_r4,
    load_spinquant_r4,
)


TRANSFORM_MANIFEST = "tfic_transform_manifest.json"


def _validate_manifest_model(model, manifest: dict) -> None:
    expected = manifest.get("model")
    if not isinstance(expected, dict):
        return
    attention = model.model.layers[0].self_attn
    actual = {
        "model_type": str(model.config.model_type),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(attention.head_dim),
    }
    mismatches = [
        f"{key}: checkpoint={value!r}, model={actual.get(key)!r}"
        for key, value in expected.items()
        if key in actual and actual[key] != value
    ]
    if mismatches:
        raise ValueError(
            "transform checkpoint does not match the loaded model: "
            + "; ".join(mismatches)
        )


def load_transform_aware_model(
    model_path: str,
    *,
    torch_dtype: torch.dtype,
    device_map,
    trust_remote_code: bool = True,
):
    """Load a checkpoint and restore any required online transforms."""

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    manifest_path = Path(model_path) / TRANSFORM_MANIFEST
    if not manifest_path.exists():
        return model

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_model(model, manifest)
    method = manifest.get("method")
    if method == "flatquant":
        transform_path = Path(model_path) / manifest["transform_file"]
        transforms, clips = load_flatquant_transforms(transform_path)
        attention_transforms = load_flatquant_attention_transforms(transform_path)
        attention_clips = load_flatquant_attention_clips(transform_path)
        apply_flatquant_transforms(
            model,
            transforms,
            weight_is_transformed=True,
            strict=True,
            activation_bits=int(manifest.get("activation_bits", 16)),
            activation_symmetric=bool(
                manifest.get("activation_symmetric", False)
            ),
            activation_group_size=int(
                manifest.get("activation_group_size", -1)
            ),
            activation_clip_ratio=float(
                manifest.get("activation_clip_ratio", 1.0)
            ),
            clips=clips,
        )
        apply_flatquant_attention_transforms(
            model,
            attention_transforms,
            q_bits=int(manifest.get("q_bits", 16)),
            k_bits=int(manifest.get("k_bits", 16)),
            v_bits=int(manifest.get("v_bits", 16)),
            q_symmetric=bool(manifest.get("q_symmetric", False)),
            k_symmetric=bool(manifest.get("k_symmetric", False)),
            v_symmetric=bool(manifest.get("v_symmetric", False)),
            q_group_size=int(manifest.get("q_group_size", -1)),
            k_group_size=int(manifest.get("k_group_size", -1)),
            q_clip_ratio=float(manifest.get("q_clip_ratio", 1.0)),
            k_clip_ratio=float(manifest.get("k_clip_ratio", 1.0)),
            v_clip_ratio=float(manifest.get("v_clip_ratio", 1.0)),
            clips=attention_clips,
        )
        return model
    if method == "spinquant_no_had":
        if int(manifest.get("k_bits", 16)) < 16:
            raise ValueError(
                "spinquant_no_had checkpoint cannot contain K-cache "
                "quantization because it requires online R3"
            )
        add_spinquant_activation_quantization(
            model,
            bits=int(manifest.get("activation_bits", 16)),
            symmetric=bool(manifest.get("activation_symmetric", False)),
            group_size=int(manifest.get("activation_group_size", -1)),
            clip_ratio=float(manifest.get("activation_clip_ratio", 1.0)),
            v_bits=int(manifest.get("v_bits", 16)),
            v_symmetric=bool(manifest.get("v_symmetric", False)),
            v_clip_ratio=float(manifest.get("v_clip_ratio", 1.0)),
        )
        return model
    if method in {"spinquant", "spinquant_had"}:
        runtime_path = Path(model_path) / manifest["runtime_file"]
        r4 = load_spinquant_r4(
            runtime_path,
            width=model.config.intermediate_size,
        )
        apply_spinquant_r4(model, r4, weight_is_transformed=True)
        add_spinquant_activation_quantization(
            model,
            bits=int(manifest.get("activation_bits", 16)),
            symmetric=bool(manifest.get("activation_symmetric", False)),
            group_size=int(manifest.get("activation_group_size", -1)),
            clip_ratio=float(manifest.get("activation_clip_ratio", 1.0)),
            v_bits=int(manifest.get("v_bits", 16)),
            v_symmetric=bool(manifest.get("v_symmetric", False)),
            v_clip_ratio=float(manifest.get("v_clip_ratio", 1.0)),
        )
        add_spinquant_k_cache_quantization(
            model,
            bits=int(manifest.get("k_bits", 16)),
            symmetric=bool(manifest.get("k_symmetric", False)),
            group_size=int(manifest.get("k_group_size", -1)),
            clip_ratio=float(manifest.get("k_clip_ratio", 1.0)),
        )
        return model
    raise ValueError(f"unsupported transform-aware checkpoint method: {method!r}")
