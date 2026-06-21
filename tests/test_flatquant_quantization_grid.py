from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_baselines import (  # noqa: E402
    build_asymmetric_flatquant_diag_quantization_grid,
    build_awq_quantization_grid,
    build_flatquant_diag_quantization_grid,
    build_symmetric_flatquant_diag_quantization_grid,
)
from tests.examples import flatquant_toy_inputs, reconstruction_error  # noqa: E402


SCHEMES = ("asymmetric", "symmetric")


def test_flatquant_diag_grid_matches_awq_when_clip_is_one():
    weights, scales, _clip = flatquant_toy_inputs()

    for scheme in SCHEMES:
        flatquant = build_flatquant_diag_quantization_grid(
            weights,
            scales,
            bits=3,
            group_size=4,
            scheme=scheme,
            weight_clip=1.0,
        )
        awq = build_awq_quantization_grid(
            weights,
            scales,
            bits=3,
            group_size=4,
            scheme=scheme,
        )

        flatquant_codes, flatquant_dequantized = flatquant.round_to_nearest()
        awq_codes, awq_dequantized = awq.round_to_nearest()

        assert torch.equal(flatquant_codes, awq_codes)
        assert torch.allclose(flatquant.scale, awq.scale)
        assert torch.equal(flatquant.zero_point, awq.zero_point)
        assert flatquant.qmin == awq.qmin
        assert flatquant.qmax == awq.qmax
        assert torch.allclose(flatquant_dequantized, awq_dequantized)


def test_flatquant_diag_grid_round_to_nearest_shape_and_range():
    weights, scales, clip = flatquant_toy_inputs()

    for scheme in SCHEMES:
        grid = build_flatquant_diag_quantization_grid(
            weights,
            scales,
            bits=3,
            group_size=4,
            scheme=scheme,
            weight_clip=clip,
        )
        integer_weights, dequantized = grid.round_to_nearest()

        assert integer_weights.shape == (2, 8)
        assert dequantized.shape == weights.shape
        assert torch.all(integer_weights >= grid.qmin)
        assert torch.all(integer_weights <= grid.qmax)
        assert abs(grid.weight_clip - clip) < 1e-6


def test_flatquant_diag_weight_clip_changes_quantization_range():
    weights, scales, _clip = flatquant_toy_inputs()

    for scheme in SCHEMES:
        unclipped = build_flatquant_diag_quantization_grid(
            weights,
            scales,
            bits=3,
            group_size=4,
            scheme=scheme,
            weight_clip=1.0,
        )
        clipped = build_flatquant_diag_quantization_grid(
            weights,
            scales,
            bits=3,
            group_size=4,
            scheme=scheme,
            weight_clip=0.5,
        )

        assert not torch.allclose(unclipped.scale, clipped.scale)


def test_flatquant_diag_scheme_helpers_match_main_builder():
    weights, scales, clip = flatquant_toy_inputs()

    symmetric = build_symmetric_flatquant_diag_quantization_grid(
        weights,
        scales,
        bits=3,
        group_size=4,
        weight_clip=clip,
    )
    asymmetric = build_asymmetric_flatquant_diag_quantization_grid(
        weights,
        scales,
        bits=3,
        group_size=4,
        weight_clip=clip,
    )
    symmetric_main = build_flatquant_diag_quantization_grid(
        weights,
        scales,
        bits=3,
        group_size=4,
        scheme="symmetric",
        weight_clip=clip,
    )
    asymmetric_main = build_flatquant_diag_quantization_grid(
        weights,
        scales,
        bits=3,
        group_size=4,
        scheme="asymmetric",
        weight_clip=clip,
    )

    assert torch.equal(symmetric.round_to_nearest()[0], symmetric_main.round_to_nearest()[0])
    assert torch.equal(asymmetric.round_to_nearest()[0], asymmetric_main.round_to_nearest()[0])


def _demo_flatquant_grid_logic():
    weights, scales, clip = flatquant_toy_inputs()
    for scheme in tqdm(SCHEMES, desc="flatquant grid demo"):
        grid = build_flatquant_diag_quantization_grid(
            weights,
            scales,
            bits=3,
            group_size=4,
            scheme=scheme,
            weight_clip=clip,
        )
        integer_weights, dequantized = grid.round_to_nearest()

        print(f"\n{scheme} FlatQuant diagonal-scale grid")
        print("weights:")
        print(weights)
        print("flatquant scales:")
        print(scales)
        print("weight_clip:")
        print(grid.weight_clip)
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
        test_flatquant_diag_grid_matches_awq_when_clip_is_one,
        test_flatquant_diag_grid_round_to_nearest_shape_and_range,
        test_flatquant_diag_weight_clip_changes_quantization_range,
        test_flatquant_diag_scheme_helpers_match_main_builder,
    ]
    for test in tqdm(tests, desc="flatquant grid tests"):
        test()
    print("\nall FlatQuant diagonal-scale grid tests passed")
    _demo_flatquant_grid_logic()
