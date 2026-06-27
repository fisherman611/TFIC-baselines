from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eigenflip.quantization.state import IntegerQuantizedTensorState  # noqa: E402
from grid_baselines import (  # noqa: E402
    build_asymmetric_awq_quantization_grid,
    build_awq_quantization_grid,
    build_symmetric_awq_quantization_grid,
)
from tests.examples import awq_toy_inputs, reconstruction_error  # noqa: E402


SCHEMES = ("asymmetric", "symmetric")


def test_awq_grid_matches_eigenflip_from_awq():
    weights, awq_scales = awq_toy_inputs()

    for scheme in SCHEMES:
        grid = build_awq_quantization_grid(
            weights,
            awq_scales,
            bits=3,
            group_size=4,
            scheme=scheme,
        )
        state = IntegerQuantizedTensorState.from_awq(
            weights,
            awq_scales,
            bits=3,
            group_size=4,
            scheme=scheme,
        )
        integer_weights, dequantized = grid.round_to_nearest()

        assert torch.equal(integer_weights, state.integer_weights)
        assert torch.allclose(grid.scale, state.scale)
        assert torch.equal(grid.zero_point, state.zero_point)
        assert grid.qmin == state.min_int
        assert grid.qmax == state.max_int
        assert torch.allclose(dequantized, state.dequantize())


def test_awq_grid_round_to_nearest_shape_and_range():
    weights, awq_scales = awq_toy_inputs()

    for scheme in SCHEMES:
        grid = build_awq_quantization_grid(
            weights,
            awq_scales,
            bits=3,
            group_size=4,
            scheme=scheme,
        )
        integer_weights, dequantized = grid.round_to_nearest()

        assert integer_weights.shape == (2, 8)
        assert dequantized.shape == weights.shape
        assert torch.all(integer_weights >= grid.qmin)
        assert torch.all(integer_weights <= grid.qmax)


def test_awq_grid_group_size_minus_one_is_channelwise():
    weights, awq_scales = awq_toy_inputs()

    grid = build_awq_quantization_grid(
        weights,
        awq_scales,
        bits=3,
        group_size=-1,
        scheme="asymmetric",
    )

    assert grid.group_size == weights.shape[1]
    assert grid.padded_in_features == weights.shape[1]
    assert grid.scale.shape == weights.shape
    assert grid.zero_point.shape == weights.shape


def test_awq_scheme_helpers_match_main_builder():
    weights, awq_scales = awq_toy_inputs()

    symmetric = build_symmetric_awq_quantization_grid(
        weights,
        awq_scales,
        bits=3,
        group_size=4,
    )
    asymmetric = build_asymmetric_awq_quantization_grid(
        weights,
        awq_scales,
        bits=3,
        group_size=4,
    )
    symmetric_main = build_awq_quantization_grid(
        weights,
        awq_scales,
        bits=3,
        group_size=4,
        scheme="symmetric",
    )
    asymmetric_main = build_awq_quantization_grid(
        weights,
        awq_scales,
        bits=3,
        group_size=4,
        scheme="asymmetric",
    )

    assert torch.equal(symmetric.round_to_nearest()[0], symmetric_main.round_to_nearest()[0])
    assert torch.equal(asymmetric.round_to_nearest()[0], asymmetric_main.round_to_nearest()[0])


def test_awq_grid_applies_per_output_group_clipping():
    weights, awq_scales = awq_toy_inputs()
    unclipped = build_awq_quantization_grid(
        weights, awq_scales, bits=3, group_size=4, scheme="asymmetric"
    )
    clip_max = unclipped.scaled_weights.reshape(2, 2, 4).abs().amax(-1, keepdim=True) * 0.5
    clipped = build_awq_quantization_grid(
        weights, awq_scales, bits=3, group_size=4,
        scheme="asymmetric", clip_max=clip_max,
    )

    assert clipped.clip_max.shape == (2, 2, 1)
    assert not torch.allclose(clipped.scale, unclipped.scale)
    assert torch.all(clipped.scaled_weights.reshape(2, 2, 4).abs() <= clip_max)


def test_awq_grid_is_finite_for_zero_fp16_groups():
    grid = build_awq_quantization_grid(
        torch.zeros(2, 4, dtype=torch.float16),
        torch.ones(4, dtype=torch.float16),
        bits=3,
        group_size=4,
    )
    codes, dequantized = grid.round_to_nearest()

    assert torch.isfinite(grid.scale).all()
    assert torch.isfinite(codes).all()
    assert torch.isfinite(dequantized).all()


@pytest.mark.parametrize(
    "scales",
    [torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0]), torch.full((5,), float("nan"))],
)
def test_awq_grid_rejects_invalid_scales(scales):
    weights, _ = awq_toy_inputs()
    with pytest.raises(ValueError, match="finite and positive"):
        build_awq_quantization_grid(weights, scales, bits=3, group_size=4)


def _demo_awq_grid_logic():
    weights, awq_scales = awq_toy_inputs()
    for scheme in tqdm(SCHEMES, desc="awq grid demo"):
        grid = build_awq_quantization_grid(
            weights,
            awq_scales,
            bits=3,
            group_size=4,
            scheme=scheme,
        )
        integer_weights, dequantized = grid.round_to_nearest()

        print(f"\n{scheme} AWQ")
        print("weights:")
        print(weights)
        print("awq scales:")
        print(awq_scales)
        print("scaled weights:")
        print(grid.scaled_weights[:, : grid.in_features])
        print("effective dequant scale:")
        print(grid.scale[:, : grid.in_features])
        print("zero_point:")
        print(grid.zero_point[:, : grid.in_features])
        print("integer range:")
        print((grid.qmin, grid.qmax))
        print("integer codes:")
        print(integer_weights[:, : grid.in_features])
        print("dequantized:")
        print(dequantized)
        print("reconstruction error:")
        print(reconstruction_error(weights, dequantized))


if __name__ == "__main__":
    tests = [
        test_awq_grid_matches_eigenflip_from_awq,
        test_awq_grid_round_to_nearest_shape_and_range,
        test_awq_scheme_helpers_match_main_builder,
        test_awq_grid_applies_per_output_group_clipping,
        test_awq_grid_is_finite_for_zero_fp16_groups,
    ]
    for test in tqdm(tests, desc="awq grid tests"):
        test()
    print("\nall AWQ quantization grid tests passed")
    _demo_awq_grid_logic()
