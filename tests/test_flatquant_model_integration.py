from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from eigenflip.statistics.collect_fast import collect_and_encode_awq_style
from grid_baselines.flatquant_model import (
    apply_flatquant_transforms,
    apply_flatquant_attention_transforms,
    load_flatquant_attention_clips,
    load_flatquant_attention_transforms,
    load_flatquant_transforms,
    serialize_flatquant_transforms,
)
from grid_baselines.transformed_linear import KroneckerTransform
from grid_baselines.transformed_linear import fake_quantize_activation


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=True)

    def forward(self, input_ids, use_cache=False):
        return self.proj(input_ids.to(self.proj.weight.dtype))


class _ToyAttentionChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        attention = nn.Module()
        attention.q_proj = nn.Linear(4, 4, bias=False)
        attention.k_proj = nn.Linear(4, 4, bias=False)
        attention.v_proj = nn.Linear(4, 4, bias=False)
        attention.o_proj = nn.Linear(4, 4, bias=False)
        self.model.layers[0].self_attn = attention

    def forward(self, values):
        attention = self.model.layers[0].self_attn
        return attention.o_proj(attention.v_proj(values))


def _transform() -> KroneckerTransform:
    return KroneckerTransform(
        left=torch.tensor([[1.2, 0.1], [0.2, 0.9]], dtype=torch.float64),
        right=torch.tensor([[0.8, -0.1], [0.3, 1.1]], dtype=torch.float64),
        diagonal=torch.tensor([0.7, 1.3, 0.9, 1.1], dtype=torch.float64),
    )


def _tiny_llama():
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
    ).eval()


def _tiny_qwen2():
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            use_sliding_window=True,
            sliding_window=4,
            max_window_layers=1,
            tie_word_embeddings=False,
        )
    ).eval()


def _tiny_mistral():
    return MistralForCausalLM(
        MistralConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            sliding_window=4,
            tie_word_embeddings=False,
        )
    ).eval()


MODEL_FACTORIES = (_tiny_llama, _tiny_qwen2, _tiny_mistral)


def _full_transform_map(model):
    transforms = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            )
        ):
            continue
        width = module.in_features
        matrix = torch.eye(width, dtype=torch.float64)
        matrix += 0.01 * torch.tril(torch.ones_like(matrix), diagonal=-1)
        transforms[name] = KroneckerTransform(
            left=torch.ones(1, 1, dtype=torch.float64),
            right=matrix,
        )
    return transforms


def test_flatquant_affine_transform_preserves_full_precision_linear_output():
    torch.manual_seed(0)
    model = _ToyModel().double().eval()
    values = torch.randn(2, 5, 4, dtype=torch.float64)
    expected = model(values)

    apply_flatquant_transforms(model, {"proj": _transform()})
    actual = model(values)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("symmetric", [False, True])
def test_flatquant_activation_quantization_is_finite_for_zero_fp16(symmetric):
    output = fake_quantize_activation(
        torch.zeros(2, 3, 8, dtype=torch.float16),
        bits=4,
        symmetric=symmetric,
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_flatquant_multi_model_full_precision_parity(model_factory):
    torch.manual_seed(0)
    model = model_factory()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids).logits
    apply_flatquant_transforms(model, _full_transform_map(model))
    with torch.no_grad():
        actual = model(input_ids).logits
    assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-5)


def test_flatquant_collector_uses_transformed_activation_statistics():
    torch.manual_seed(0)
    model = _ToyModel().double().eval()
    transform = _transform()
    apply_flatquant_transforms(model, {"proj": transform})
    sample = torch.randn(1, 6, 4, dtype=torch.float64)
    captured = {}

    def callback(name, module, stats):
        captured[name] = stats.diag_H.clone()

    collect_and_encode_awq_style(
        model,
        tokenizer=None,
        calib=[sample],
        device="cpu",
        need_H=True,
        k=0,
        eps=1e-6,
        callback=callback,
        layer_batch_size=1,
        keep_sigma=True,
        skip_lm_head=False,
        stats_device="cpu",
    )

    transformed = transform.apply(sample).reshape(-1, 4)
    expected = transformed.square().mean(dim=0)
    assert torch.allclose(captured["proj"], expected, atol=1e-6, rtol=1e-6)


def test_flatquant_loader_accepts_official_flat_matrices_layout(tmp_path):
    state = {
        0: {
            "self_attn.ln_trans.matrix_left": torch.eye(2),
            "self_attn.ln_trans.matrix_right": torch.eye(2),
            "self_attn.o_trans.matrix": torch.eye(2),
            "self_attn.vcache_trans.matrix": torch.eye(2),
            "mlp.up_gate_trans.matrix_left": torch.eye(2),
            "mlp.up_gate_trans.matrix_right": torch.eye(2),
            "mlp.down_trans.matrix_left": torch.eye(2),
            "mlp.down_trans.matrix_right": torch.eye(2),
        }
    }
    path = tmp_path / "flat_matrices.pth"
    torch.save(state, path)
    transforms, clips = load_flatquant_transforms(path)
    assert clips == {}
    assert set(transforms) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
    }


def test_flatquant_strict_mode_rejects_incomplete_model_artifact():
    model = _tiny_llama()
    transforms = _full_transform_map(model)
    transforms.pop("model.layers.0.mlp.down_proj")
    with pytest.raises(KeyError, match="missing required linears"):
        apply_flatquant_transforms(model, transforms)


def test_flatquant_loader_accepts_official_cache_clipping(tmp_path):
    path = tmp_path / "flat_matrices.pth"
    torch.save(
        {
            0: {
                "self_attn.k_cache_quantizer.clip_factor_a_max": torch.tensor(2.0),
                "self_attn.k_cache_quantizer.clip_factor_a_min": torch.tensor(1.0),
            }
        },
        path,
    )
    clips = load_flatquant_attention_clips(path)
    layer = clips["model.layers.0.self_attn"]
    assert torch.allclose(layer["k_clip_max"], torch.sigmoid(torch.tensor(2.0)))
    assert torch.allclose(layer["k_clip_min"], torch.sigmoid(torch.tensor(1.0)))


def test_flatquant_normalized_artifact_round_trip(tmp_path):
    source = {"proj": _transform()}
    path = tmp_path / "normalized.pt"
    torch.save({"layers": serialize_flatquant_transforms(source)}, path)
    loaded, clips = load_flatquant_transforms(path)
    assert clips == {}
    restored = loaded["proj"].input_transform
    assert torch.equal(restored.left, source["proj"].left)
    assert torch.equal(restored.right, source["proj"].right)
    assert torch.equal(restored.diagonal, source["proj"].diagonal)


def test_official_vcache_and_o_transforms_preserve_attention_value_chain(tmp_path):
    torch.manual_seed(4)
    model = _ToyAttentionChain().double()
    values = torch.randn(2, 3, 4, dtype=torch.float64)
    expected = model(values)
    state = {
        0: {
            "self_attn.ln_trans.matrix_left": torch.eye(2),
            "self_attn.ln_trans.matrix_right": torch.eye(2),
            "self_attn.o_trans.matrix": torch.tensor(
                [[1.1, 0.2], [0.1, 0.9]], dtype=torch.float64
            ),
            "self_attn.vcache_trans.matrix": torch.tensor(
                [[0.8, -0.1], [0.3, 1.2]], dtype=torch.float64
            ),
        }
    }
    path = tmp_path / "flat_matrices.pth"
    torch.save(state, path)
    transforms, clips = load_flatquant_transforms(path)
    apply_flatquant_transforms(model, transforms, clips=clips)
    actual = model(values)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_flatquant_post_rope_cache_adapter_runs(tmp_path, model_factory):
    model = model_factory()
    matrix = torch.tensor([[1.1, 0.2], [0.1, 0.9]])
    path = tmp_path / "flat_matrices.pth"
    torch.save({0: {"self_attn.kcache_trans.matrix": matrix}}, path)
    transforms = load_flatquant_attention_transforms(path)
    apply_flatquant_attention_transforms(
        model,
        transforms,
        q_bits=4,
        k_bits=4,
        v_bits=4,
    )
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True).logits
    assert logits.shape == (1, 4, 32)
    attention = model.model.layers[0].self_attn
    assert attention._flatquant_cache_patched
