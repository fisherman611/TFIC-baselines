from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_baselines import (  # noqa: E402
    SpinQuantRotations,
    add_spinquant_k_cache_quantization,
    add_spinquant_activation_quantization,
    apply_spinquant_no_had,
    apply_spinquant_r4,
    build_asymmetric_spinquant_quantization_grid,
    build_spinquant_quantization_grid,
    build_symmetric_spinquant_quantization_grid,
    build_vanilla_quantization_grid,
    cayley_update,
    capture_spinquant_layer_inputs,
    identity_spinquant_rotations,
    load_spinquant_rotations,
    random_spinquant_rotations,
    SpinQuantTrainingConfig,
    train_spinquant_cross_entropy,
    train_spinquant_layer_rotations,
)
from grid_baselines.transformed_linear import ActivationQuantizedLinear  # noqa: E402
from scripts.train_flatquant import capture_first_layer_inputs  # noqa: E402
from tests.examples import assignment_toy_weights  # noqa: E402


def _orthogonal(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(size, size, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(matrix).Q


def _tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).eval()


def _tiny_qwen2() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_sliding_window=True,
        sliding_window=4,
        max_window_layers=2,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


def _tiny_mistral() -> MistralForCausalLM:
    config = MistralConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        sliding_window=4,
        tie_word_embeddings=False,
    )
    return MistralForCausalLM(config).eval()


MODEL_FACTORIES = (_tiny_llama, _tiny_qwen2, _tiny_mistral)


def _rotations_for(model) -> SpinQuantRotations:
    head_dim = int(model.model.layers[0].self_attn.head_dim)
    return SpinQuantRotations(
        R1=_orthogonal(model.config.hidden_size, 1),
        R2={
            layer_idx: _orthogonal(head_dim, layer_idx + 2)
            for layer_idx in range(model.config.num_hidden_layers)
        },
    )


def _tiny_rotations() -> SpinQuantRotations:
    return _rotations_for(_tiny_llama())


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_spinquant_no_had_multi_model_full_precision_parity(model_factory):
    torch.manual_seed(0)
    model = model_factory()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids).logits
    apply_spinquant_no_had(model, _rotations_for(model))
    with torch.no_grad():
        actual = model(input_ids).logits
    assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-5)


def test_spinquant_no_had_preserves_full_precision_model_output():
    torch.manual_seed(0)
    model = _tiny_llama()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        expected = model(input_ids).logits

    apply_spinquant_no_had(model, _tiny_rotations())

    with torch.no_grad():
        actual = model(input_ids).logits
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_spinquant_r4_preserves_full_precision_model_output():
    torch.manual_seed(0)
    model = _tiny_llama()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids).logits

    apply_spinquant_no_had(model, _tiny_rotations())
    apply_spinquant_r4(model)

    with torch.no_grad():
        actual = model(input_ids).logits
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
    down = model.model.layers[0].mlp.down_proj
    values = torch.randn(2, 3, model.config.intermediate_size)
    assert not torch.equal(down.assignment_input(values), values)


def test_spinquant_loader_accepts_official_rotation_keys(tmp_path):
    checkpoint = {
        'R1': _orthogonal(8, 1),
        'model.layers.0.self_attn.R2': _orthogonal(2, 2),
        'model.layers.1.self_attn.R2': _orthogonal(2, 3),
    }
    path = tmp_path / 'R.bin'
    torch.save(checkpoint, path)

    rotations = load_spinquant_rotations(
        path,
        num_layers=2,
        hidden_size=8,
        head_dim=2,
    )
    assert torch.equal(rotations.R1, checkpoint['R1'])
    assert torch.equal(
        rotations.R2[1],
        checkpoint['model.layers.1.self_attn.R2'],
    )


def test_spinquant_loader_rejects_missing_layer_rotation(tmp_path):
    path = tmp_path / 'R.bin'
    torch.save({'R1': _orthogonal(8, 1)}, path)
    with pytest.raises(ValueError, match='missing model.layers.0.self_attn.R2'):
        load_spinquant_rotations(
            path,
            num_layers=1,
            hidden_size=8,
            head_dim=2,
        )


def test_spinquant_rejects_non_orthogonal_rotation():
    model = _tiny_llama()
    rotations = _tiny_rotations()
    rotations.R1 = rotations.R1.clone()
    rotations.R1[0, 0] *= 2
    with pytest.raises(ValueError, match='R1 must be orthogonal'):
        apply_spinquant_no_had(model, rotations)


def test_random_spinquant_rotations_are_reproducible_and_orthogonal():
    first = random_spinquant_rotations(
        num_layers=2,
        hidden_size=8,
        head_dim=2,
        seed=123,
    )
    second = random_spinquant_rotations(
        num_layers=2,
        hidden_size=8,
        head_dim=2,
        seed=123,
    )
    assert torch.equal(first.R1, second.R1)
    assert torch.equal(first.R2[1], second.R2[1])
    assert torch.allclose(
        first.R1.t().matmul(first.R1),
        torch.eye(8, dtype=torch.float64),
    )
    assert torch.allclose(
        first.R2[0].t().matmul(first.R2[0]),
        torch.eye(2, dtype=torch.float64),
    )


def test_random_spinquant_no_had_preserves_full_precision_model_output():
    torch.manual_seed(0)
    model = _tiny_llama()
    rotations = random_spinquant_rotations(
        num_layers=2,
        hidden_size=8,
        head_dim=2,
        seed=123,
    )
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        expected = model(input_ids).logits

    apply_spinquant_no_had(model, rotations)

    with torch.no_grad():
        actual = model(input_ids).logits
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_spinquant_grid_matches_vanilla_on_rotated_weights():
    weights = assignment_toy_weights()
    for scheme in ('symmetric', 'asymmetric'):
        spinquant = build_spinquant_quantization_grid(
            weights,
            bits=3,
            group_size=5,
            scheme=scheme,
        )
        vanilla = build_vanilla_quantization_grid(
            weights,
            bits=3,
            group_size=5,
            scheme=scheme,
        )
        spin_codes, spin_dequantized = spinquant.round_to_nearest()
        vanilla_codes, vanilla_dequantized = vanilla.round_to_nearest()
        assert torch.equal(spin_codes, vanilla_codes)
        assert torch.allclose(spin_dequantized, vanilla_dequantized)


def test_spinquant_scheme_helpers_match_main_builder():
    weights = assignment_toy_weights()
    symmetric = build_symmetric_spinquant_quantization_grid(
        weights, bits=3, group_size=5
    )
    asymmetric = build_asymmetric_spinquant_quantization_grid(
        weights, bits=3, group_size=5
    )
    assert torch.equal(
        symmetric.quantize(),
        build_spinquant_quantization_grid(
            weights, bits=3, group_size=5, scheme='symmetric'
        ).quantize(),
    )
    assert torch.equal(
        asymmetric.quantize(),
        build_spinquant_quantization_grid(
            weights, bits=3, group_size=5, scheme='asymmetric'
        ).quantize(),
    )


def test_cayley_update_preserves_orthogonality():
    rotation = _orthogonal(8, 4)
    gradient = torch.randn_like(rotation)
    updated = cayley_update(rotation, gradient, step_size=0.05)
    assert torch.allclose(
        updated.t().matmul(updated),
        torch.eye(8, dtype=updated.dtype),
        atol=1e-10,
        rtol=1e-10,
    )


def test_spinquant_rotation_training_exports_loadable_artifact(tmp_path):
    torch.manual_seed(0)
    model = _tiny_llama().float()
    inputs, kwargs = capture_first_layer_inputs(
        model,
        tokenizer=None,
        calibration=[torch.tensor([[1, 2, 3, 4]])],
        input_device=torch.device("cpu"),
    )
    captured, _outputs = capture_spinquant_layer_inputs(
        model.model.layers[0],
        inputs,
        kwargs,
        device="cpu",
    )
    rotations = identity_spinquant_rotations(
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        head_dim=model.model.layers[0].self_attn.head_dim,
    )
    config = SpinQuantTrainingConfig(
        weight_group_size=4,
        r1_steps=1,
        r2_steps=1,
        batch_size=1,
        learning_rate=1e-3,
    )
    r1, r2, r1_history = train_spinquant_layer_rotations(
        model.model.layers[0],
        captured,
        r1=rotations.R1,
        r2=rotations.R2[0],
        config=config,
        device="cpu",
        train_r1=True,
        train_r2=False,
    )
    r1, r2, r2_history = train_spinquant_layer_rotations(
        model.model.layers[0],
        captured,
        r1=r1,
        r2=r2,
        config=config,
        device="cpu",
        train_r1=False,
        train_r2=True,
    )
    assert len(r1_history) == 1
    assert len(r2_history) == 1
    assert torch.allclose(r1.t() @ r1, torch.eye(8, dtype=r1.dtype), atol=1e-5)
    assert torch.allclose(r2.t() @ r2, torch.eye(2, dtype=r2.dtype), atol=1e-5)

    path = tmp_path / "spinquant_R.pt"
    torch.save(
        {
            "R1": r1,
            "model.layers.0.self_attn.R2": r2,
            "model.layers.1.self_attn.R2": rotations.R2[1],
        },
        path,
    )
    loaded = load_spinquant_rotations(
        path,
        num_layers=2,
        hidden_size=8,
        head_dim=2,
    )
    assert torch.equal(loaded.R1, r1)
    assert torch.equal(loaded.R2[0], r2)


def test_spinquant_cross_entropy_training_updates_orthogonal_rotations():
    torch.manual_seed(0)
    model = _tiny_llama().float()
    rotations = identity_spinquant_rotations(
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        head_dim=model.model.layers[0].self_attn.head_dim,
    )
    config = SpinQuantTrainingConfig(
        weight_group_size=4,
        r1_steps=1,
        batch_size=1,
        learning_rate=1e-4,
        objective="cross_entropy",
    )
    trained, history = train_spinquant_cross_entropy(
        model,
        [torch.tensor([[1, 2, 3, 4]])],
        rotations,
        config=config,
        device="cpu",
    )
    assert len(history) == 1
    assert trained.R1.shape == rotations.R1.shape
    assert trained.R2[0].shape == rotations.R2[0].shape
    assert torch.allclose(
        trained.R1.t() @ trained.R1,
        torch.eye(8, dtype=trained.R1.dtype),
        atol=1e-5,
    )
    assert torch.allclose(
        trained.R2[0].t() @ trained.R2[0],
        torch.eye(2, dtype=trained.R2[0].dtype),
        atol=1e-5,
    )


def test_spinquant_assignment_uses_actual_quantized_activation():
    model = _tiny_llama()
    apply_spinquant_no_had(model, _tiny_rotations())
    add_spinquant_activation_quantization(
        model,
        bits=8,
        symmetric=False,
        group_size=-1,
    )
    module = model.model.layers[0].self_attn.q_proj
    values = torch.randn(2, 3, 8)
    assert torch.equal(
        module.assignment_input(values),
        module.quantized_assignment_input(values),
    )
    assert not torch.equal(module.assignment_input(values), values)
    assert module(values).shape == (2, 3, 8)


def test_spinquant_activation_scope_matches_official_code():
    model = _tiny_llama()
    apply_spinquant_no_had(model, _tiny_rotations())
    add_spinquant_activation_quantization(
        model,
        bits=4,
        symmetric=False,
        group_size=4,
        v_bits=4,
        v_symmetric=True,
    )

    attention = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp
    assert isinstance(attention.q_proj, ActivationQuantizedLinear)
    assert attention.q_proj.activation_group_size == 4
    assert attention.o_proj.activation_group_size == 2
    assert attention.v_proj.output_bits == 4
    assert attention.v_proj.output_group_size == 2
    assert attention.v_proj.output_symmetric is True
    assert mlp.down_proj.activation_group_size == 4
    assert type(model.lm_head) is torch.nn.Linear


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
@pytest.mark.parametrize("group_size", [-1, 2])
def test_spinquant_post_rope_r3_k_cache_path_runs(model_factory, group_size):
    model = model_factory()
    apply_spinquant_no_had(model, _rotations_for(model))
    apply_spinquant_r4(model)
    add_spinquant_k_cache_quantization(
        model,
        bits=4,
        symmetric=False,
        group_size=group_size,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits = model(input_ids, use_cache=True).logits
    assert logits.shape == (1, 4, 32)
    assert model.model.layers[0].self_attn._spinquant_qk_patched
