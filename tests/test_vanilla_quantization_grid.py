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
from tests.examples import (  # noqa: E402
    reconstruction_error,
    vanilla_fixed_grid_weights,
    vanilla_grid_demo_examples,
    vanilla_manual_weights,
    vanilla_new_assignment_weights,
    vanilla_padding_weights,
)


def _pad_to_group(weights: torch.Tensor, group_size: int) -> torch.Tensor:
    rows, in_features = weights.shape
    padded_in = ((in_features + group_size - 1) // group_size) * group_size
    if padded_in == in_features:
        return weights
    padded = torch.zeros(rows, padded_in, dtype=weights.dtype, device=weights.device)
    padded[:, :in_features] = weights
    return padded


def _manual_symmetric_rtn(weights: torch.Tensor, bits: int, group_size: int):
    padded = _pad_to_group(weights, group_size)
    rows, padded_in = padded.shape
    groups = padded_in // group_size
    grouped = padded.reshape(rows, groups, group_size)
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale_group = (grouped.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8)
    scale = scale_group.repeat(1, 1, group_size).reshape(rows, padded_in)
    codes = torch.round(padded / scale).clamp(qmin, qmax)
    dequantized = (codes * scale)[:, : weights.shape[1]]
    return codes[:, : weights.shape[1]], dequantized


def _manual_asymmetric_rtn(weights: torch.Tensor, bits: int, group_size: int):
    padded = _pad_to_group(weights, group_size)
    rows, padded_in = padded.shape
    groups = padded_in // group_size
    grouped = padded.reshape(rows, groups, group_size)
    qmin = 0
    qmax = 2**bits - 1
    wmin = grouped.min(dim=2, keepdim=True)[0]
    wmax = grouped.max(dim=2, keepdim=True)[0]
    scale_group = ((wmax - wmin) / qmax).clamp_min(1e-8)
    zero_point_group = torch.round(-wmin / scale_group).clamp(qmin, qmax)
    scale = scale_group.repeat(1, 1, group_size).reshape(rows, padded_in)
    zero_point = zero_point_group.repeat(1, 1, group_size).reshape(rows, padded_in)
    codes = torch.round(padded / scale + zero_point).clamp(qmin, qmax)
    dequantized = ((codes - zero_point) * scale)[:, : weights.shape[1]]
    return codes[:, : weights.shape[1]], dequantized


def test_round_to_nearest_returns_original_shape_after_padding():
    weights = vanilla_padding_weights()

    grid = build_vanilla_quantization_grid(weights, bits=3, group_size=4)
    integer_weights, dequantized = grid.round_to_nearest()

    assert integer_weights.shape == (2, 8)
    assert dequantized.shape == weights.shape
    assert grid.scheme == "symmetric"
    assert grid.qmin == -4
    assert grid.qmax == 3


def test_quantize_dequantize_matches_manual_groupwise_symmetric_rtn():
    weights = vanilla_manual_weights()

    grid = build_symmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    integer_weights, dequantized = grid.round_to_nearest()

    expected_codes, expected_dequantized = _manual_symmetric_rtn(
        weights,
        bits=3,
        group_size=4,
    )

    assert torch.equal(integer_weights[:, : weights.shape[1]], expected_codes)
    assert torch.allclose(dequantized, expected_dequantized, atol=1e-6)


def test_quantize_dequantize_matches_manual_groupwise_asymmetric_rtn():
    weights = vanilla_manual_weights()

    grid = build_asymmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    integer_weights, dequantized = grid.round_to_nearest()

    expected_codes, expected_dequantized = _manual_asymmetric_rtn(
        weights,
        bits=3,
        group_size=4,
    )

    assert grid.scheme == "asymmetric"
    assert torch.equal(integer_weights[:, : weights.shape[1]], expected_codes)
    assert torch.allclose(dequantized, expected_dequantized, atol=1e-6)


def test_quantize_can_assign_new_weights_to_fixed_grid():
    weights = vanilla_fixed_grid_weights()
    new_weights = vanilla_new_assignment_weights()

    grid = build_symmetric_vanilla_quantization_grid(weights, bits=3, group_size=4)
    new_codes = grid.quantize(new_weights)
    new_dequantized = grid.dequantize(new_codes)

    assert new_codes.shape == (2, 8)
    assert new_dequantized.shape == weights.shape
    assert torch.all(new_codes >= grid.qmin)
    assert torch.all(new_codes <= grid.qmax)


def test_eigenflip_rtn_state_matches_vanilla_symmetric_grid():
    weights = vanilla_manual_weights()

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
    weights = vanilla_manual_weights()

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
    demo_cases = [
        (idx, weights, scheme)
        for idx, weights in enumerate(vanilla_grid_demo_examples(), start=1)
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
        print(reconstruction_error(weights, dequantized))


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
