from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_baselines import (
    build_asymmetric_vanilla_quantization_grid,
    build_symmetric_vanilla_quantization_grid,
    build_vanilla_quantization_grid,
)
from eigenflip.quantization.state import IntegerQuantizedTensorState


def _reconstruction_error(weights: torch.Tensor, dequantized: torch.Tensor) -> dict[str, float]:
    error = dequantized.float() - weights.float()
    return {
        "mse": float((error * error).mean().item()),
        "mae": float(error.abs().mean().item()),
        "max_abs": float(error.abs().max().item()),
    }


def test_round_to_nearest_returns_original_shape_after_padding():
    weights = torch.tensor([[0.0, 0.1, 0.26, -0.2, 0.4]], dtype=torch.float32)

    grid = build_vanilla_quantization_grid(weights, bits=3, group_size=4)
    integer_weights, dequantized = grid.round_to_nearest()

    assert integer_weights.shape == (1, 8)
    assert dequantized.shape == weights.shape
    assert grid.scheme == "symmetric"
    assert grid.qmin == -4
    assert grid.qmax == 3


def test_quantize_dequantize_matches_manual_groupwise_symmetric_rtn():
    weights = torch.tensor([[-0.2, 0.0, 0.1, 0.26]], dtype=torch.float32)

    grid = build_symmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    integer_weights, dequantized = grid.round_to_nearest()

    scale = 0.26 / 3
    expected_codes = torch.round(weights / scale).clamp(-4, 3)
    expected_dequantized = expected_codes * scale

    assert torch.equal(integer_weights[:, : weights.shape[1]], expected_codes)
    assert torch.allclose(dequantized, expected_dequantized, atol=1e-6)


def test_quantize_dequantize_matches_manual_groupwise_asymmetric_rtn():
    weights = torch.tensor([[-0.2, 0.0, 0.1, 0.26]], dtype=torch.float32)

    grid = build_asymmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    integer_weights, dequantized = grid.round_to_nearest()

    scale = (0.26 - (-0.2)) / 7
    zero_point = round(-(-0.2) / scale)
    expected_codes = torch.round(weights / scale + zero_point).clamp(0, 7)
    expected_dequantized = (expected_codes - zero_point) * scale

    assert grid.scheme == "asymmetric"
    assert torch.equal(integer_weights[:, : weights.shape[1]], expected_codes)
    assert torch.allclose(dequantized, expected_dequantized, atol=1e-6)


def test_quantize_can_assign_new_weights_to_fixed_grid():
    weights = torch.tensor([[-0.2, 0.0, 0.1, 0.3]], dtype=torch.float32)
    new_weights = torch.tensor([[-0.18, 0.04, 0.12, 0.27]], dtype=torch.float32)

    grid = build_symmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    new_codes = grid.quantize(new_weights)
    new_dequantized = grid.dequantize(new_codes)

    assert new_codes.shape == weights.shape
    assert new_dequantized.shape == weights.shape
    assert torch.all(new_codes >= grid.qmin)
    assert torch.all(new_codes <= grid.qmax)


def test_eigenflip_rtn_state_matches_vanilla_symmetric_grid():
    weights = torch.tensor([[-0.2, 0.0, 0.1, 0.26]], dtype=torch.float32)

    grid = build_symmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    state = IntegerQuantizedTensorState.from_rtn(
        weights,
        bits=3,
        group_size=4,
        scheme="symmetric",
    )
    integer_weights, dequantized = grid.round_to_nearest()

    assert torch.equal(state.integer_weights, integer_weights)
    assert torch.allclose(state.scale, grid.scale)
    assert torch.equal(state.zero_point, grid.zero_point)
    assert state.min_int == grid.qmin
    assert state.max_int == grid.qmax
    assert torch.allclose(state.dequantize(), dequantized)


def test_eigenflip_rtn_state_matches_vanilla_asymmetric_grid():
    weights = torch.tensor([[-0.2, 0.0, 0.1, 0.26]], dtype=torch.float32)

    grid = build_asymmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    state = IntegerQuantizedTensorState.from_rtn(
        weights,
        bits=3,
        group_size=4,
        scheme="asymmetric",
    )
    integer_weights, dequantized = grid.round_to_nearest()

    assert torch.equal(state.integer_weights, integer_weights)
    assert torch.allclose(state.scale, grid.scale)
    assert torch.equal(state.zero_point, grid.zero_point)
    assert state.min_int == grid.qmin
    assert state.max_int == grid.qmax
    assert torch.allclose(state.dequantize(), dequantized)


def _demo_vanilla_grid_logic():
    examples = [
        torch.tensor([[-0.2, 0.0, 0.1, 0.26]], dtype=torch.float32),
        torch.tensor([[0.0, 0.1, 0.26, -0.2, 0.4]], dtype=torch.float32),
    ]

    demo_cases = [
        (idx, weights, scheme)
        for idx, weights in enumerate(examples, start=1)
        for scheme in ("symmetric", "asymmetric")
    ]

    for idx, weights, scheme in tqdm(demo_cases, desc="vanilla grid demo"):
        grid = build_vanilla_quantization_grid(
            weights,
            bits=3,
            group_size=4,
            scheme=scheme,
        )
        integer_weights, dequantized = grid.round_to_nearest()
        print(f"\nexample {idx} ({scheme})")
        print("weights:")
        print(weights)
        print("q range:", grid.qmin, grid.qmax)
        print("scale:")
        print(grid.scale[:, : weights.shape[1]])
        print("zero_point:")
        print(grid.zero_point[:, : weights.shape[1]])
        print("integer codes:")
        print(integer_weights[:, : weights.shape[1]])
        print("dequantized:")
        print(dequantized)
        print("reconstruction error:")
        print(_reconstruction_error(weights, dequantized))


if __name__ == "__main__":
    tests = [
        test_round_to_nearest_returns_original_shape_after_padding,
        test_quantize_dequantize_matches_manual_groupwise_symmetric_rtn,
        test_quantize_dequantize_matches_manual_groupwise_asymmetric_rtn,
        test_quantize_can_assign_new_weights_to_fixed_grid,
        test_eigenflip_rtn_state_matches_vanilla_symmetric_grid,
        test_eigenflip_rtn_state_matches_vanilla_asymmetric_grid,
    ]
    for test in tqdm(tests, desc="vanilla grid tests"):
        test()
    print("\nall vanilla quantization grid tests passed")
    _demo_vanilla_grid_logic()
