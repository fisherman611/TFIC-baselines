from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_baselines import (  # noqa: E402
    SpinQuantRotations,
    apply_spinquant_no_had,
    build_asymmetric_spinquant_quantization_grid,
    build_spinquant_quantization_grid,
    build_symmetric_spinquant_quantization_grid,
    build_vanilla_quantization_grid,
    load_spinquant_rotations,
    random_spinquant_rotations,
)
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


def _tiny_rotations() -> SpinQuantRotations:
    return SpinQuantRotations(
        R1=_orthogonal(8, 1),
        R2={0: _orthogonal(2, 2), 1: _orthogonal(2, 3)},
    )


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
