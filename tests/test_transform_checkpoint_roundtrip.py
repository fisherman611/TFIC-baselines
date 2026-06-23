from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from baseline_utils.model_loading import (
    TRANSFORM_MANIFEST,
    load_transform_aware_model,
)
from grid_baselines import (
    SpinQuantRotations,
    add_spinquant_activation_quantization,
    add_spinquant_k_cache_quantization,
    apply_flatquant_attention_transforms,
    apply_flatquant_transforms,
    apply_spinquant_no_had,
    apply_spinquant_r4,
    serialize_flatquant_transforms,
)
from grid_baselines.transformed_linear import KroneckerTransform


def _tiny_model(family: str):
    common = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    if family == "llama":
        return LlamaForCausalLM(LlamaConfig(**common)).eval()
    if family == "qwen2":
        config = Qwen2Config(
            **common,
            use_sliding_window=True,
            sliding_window=4,
            max_window_layers=1,
        )
        return Qwen2ForCausalLM(config).eval()
    if family == "mistral":
        return MistralForCausalLM(
            MistralConfig(**common, sliding_window=4)
        ).eval()
    raise AssertionError(family)


def _metadata(model) -> dict[str, int | str]:
    attention = model.model.layers[0].self_attn
    return {
        "model_type": str(model.config.model_type),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(attention.head_dim),
    }


def _full_transform_map(model):
    transforms = {}
    projections = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(projections):
            matrix = torch.eye(module.in_features)
            matrix += 0.01 * torch.tril(torch.ones_like(matrix), diagonal=-1)
            transforms[name] = KroneckerTransform(torch.ones(1, 1), matrix)
    return transforms


def _orthogonal(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(size, size, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(value).Q


def _rotations(model) -> SpinQuantRotations:
    head_dim = int(model.model.layers[0].self_attn.head_dim)
    return SpinQuantRotations(
        R1=_orthogonal(model.config.hidden_size, 1),
        R2={0: _orthogonal(head_dim, 2)},
    )


def _write_manifest(path, manifest: dict) -> None:
    (path / TRANSFORM_MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
def test_flatquant_checkpoint_round_trip(family, tmp_path):
    torch.manual_seed(0)
    model = _tiny_model(family)
    transforms = _full_transform_map(model)
    apply_flatquant_transforms(model, transforms, activation_bits=4)
    attention_name = "model.layers.0.self_attn"
    attention_transforms = {attention_name: torch.tensor([[1.1, 0.2], [0.1, 0.9]])}
    apply_flatquant_attention_transforms(
        model,
        attention_transforms,
        q_bits=4,
        k_bits=4,
        v_bits=4,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids, use_cache=True).logits

    model.save_pretrained(tmp_path)
    torch.save(
        {
            "layers": serialize_flatquant_transforms(transforms),
            "attention": attention_transforms,
        },
        tmp_path / "flatquant_transforms.pt",
    )
    _write_manifest(
        tmp_path,
        {
            "version": 1,
            "method": "flatquant",
            "model": _metadata(model),
            "transform_file": "flatquant_transforms.pt",
            "activation_bits": 4,
            "q_bits": 4,
            "k_bits": 4,
            "v_bits": 4,
        },
    )
    loaded = load_transform_aware_model(
        str(tmp_path), torch_dtype=torch.float32, device_map=None
    ).eval()
    with torch.no_grad():
        actual = loaded(input_ids, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("family", ["llama", "qwen2", "mistral"])
def test_spinquant_had_checkpoint_round_trip(family, tmp_path):
    torch.manual_seed(0)
    model = _tiny_model(family)
    apply_spinquant_no_had(model, _rotations(model))
    r4 = apply_spinquant_r4(model)
    add_spinquant_activation_quantization(model, bits=4, v_bits=4)
    add_spinquant_k_cache_quantization(model, bits=4)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids, use_cache=True).logits

    model.save_pretrained(tmp_path)
    torch.save(
        {"r4": {"K": r4.k, "had_K": r4.had_k}},
        tmp_path / "spinquant_runtime.pt",
    )
    _write_manifest(
        tmp_path,
        {
            "version": 1,
            "method": "spinquant_had",
            "model": _metadata(model),
            "runtime_file": "spinquant_runtime.pt",
            "activation_bits": 4,
            "v_bits": 4,
            "k_bits": 4,
        },
    )
    loaded = load_transform_aware_model(
        str(tmp_path), torch_dtype=torch.float32, device_map=None
    ).eval()
    with torch.no_grad():
        actual = loaded(input_ids, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-5)
