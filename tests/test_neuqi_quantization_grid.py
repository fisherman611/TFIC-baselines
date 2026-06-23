from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment_methods import GPTQAssignment, RTNAssignment  # noqa: E402
from grid_baselines import (  # noqa: E402
    build_asymmetric_neuqi_quantization_grid,
    build_neuqi_quantization_grid,
    build_symmetric_neuqi_quantization_grid,
    build_vanilla_quantization_grid,
)
from tests.examples import (  # noqa: E402
    assignment_toy_weights,
    toy_correlated_stats,
    weighted_reconstruction_energy,
)
from grid_baselines.neuqi_quantization_grid import _optimal_zero_point  # noqa: E402


def _neuqi_grid():
    return build_neuqi_quantization_grid(
        assignment_toy_weights(),
        toy_correlated_stats(),
        bits=2,
        group_size=5,
        scale_candidates=64,
        coarse_candidates=8,
        row_chunk_size=1,
        candidate_chunk_size=4,
    )


@pytest.mark.parametrize(
    ("scheme", "qmin", "qmax"),
    [
        ("asymmetric", 0, 3),
        ("symmetric", -2, 1),
    ],
)
def test_neuqi_grid_round_to_nearest_shape_and_range(scheme, qmin, qmax):
    grid = build_neuqi_quantization_grid(
        assignment_toy_weights(),
        toy_correlated_stats(),
        bits=2,
        group_size=5,
        scheme=scheme,
        scale_candidates=64,
        coarse_candidates=8,
        row_chunk_size=1,
        candidate_chunk_size=4,
    )
    codes, dequantized = grid.round_to_nearest()

    assert codes.shape == (2, 5)
    assert dequantized.shape == (2, 5)
    assert grid.qmin == qmin
    assert grid.qmax == qmax
    assert grid.scheme == scheme
    assert torch.all(codes >= grid.qmin)
    assert torch.all(codes <= grid.qmax)


def test_neuqi_uses_floating_zero_points():
    grid = _neuqi_grid()
    fractional = grid.zero_point - torch.round(grid.zero_point)
    assert torch.any(fractional.abs() > 1e-5)


def test_neuqi_group_size_minus_one_is_channelwise():
    weights = assignment_toy_weights()
    grid = build_neuqi_quantization_grid(
        weights,
        toy_correlated_stats(),
        bits=2,
        group_size=-1,
        scale_candidates=32,
        coarse_candidates=8,
        row_chunk_size=1,
        candidate_chunk_size=4,
    )

    assert grid.group_size == -1
    assert grid.in_features == weights.shape[1]
    assert grid.padded_in_features == weights.shape[1]
    assert grid.scale.shape == weights.shape
    assert grid.zero_point.shape == weights.shape


def test_neuqi_zero_point_solver_matches_dense_scan():
    weights = assignment_toy_weights()[0]
    scale = torch.tensor(0.4258667)
    x = (weights / scale).reshape(1, 1, 1, -1)
    hessian_diag = torch.ones_like(x)

    z, loss = _optimal_zero_point(x, hessian_diag, qmax=3)

    candidates = torch.linspace(z.item() - 0.1, z.item() + 0.1, 2001)
    dense_error = x.reshape(-1).unsqueeze(0) + candidates.unsqueeze(1)
    dense_error = dense_error - torch.round(dense_error).clamp(0, 3)
    dense_loss = dense_error.square().sum(dim=1).amin()
    assert torch.allclose(loss.reshape(()), dense_loss, atol=1e-5)


def test_neuqi_rtn_weighted_loss_is_no_worse_than_vanilla_asymmetric():
    weights = assignment_toy_weights()
    stats = toy_correlated_stats()
    neuqi = _neuqi_grid()
    vanilla = build_vanilla_quantization_grid(
        weights,
        bits=2,
        group_size=5,
        scheme="asymmetric",
    )

    neuqi_out = neuqi.round_to_nearest()[1]
    vanilla_out = vanilla.round_to_nearest()[1]
    assert weighted_reconstruction_energy(weights, neuqi_out, stats) <= (
        weighted_reconstruction_energy(weights, vanilla_out, stats) + 1e-6
    )


def test_neuqi_symmetric_rtn_weighted_loss_is_no_worse_than_vanilla_symmetric():
    weights = assignment_toy_weights()
    stats = toy_correlated_stats()
    neuqi = build_neuqi_quantization_grid(
        weights,
        stats,
        bits=2,
        group_size=5,
        scheme="symmetric",
        scale_candidates=64,
        coarse_candidates=8,
        row_chunk_size=1,
        candidate_chunk_size=4,
    )
    vanilla = build_vanilla_quantization_grid(
        weights,
        bits=2,
        group_size=5,
        scheme="symmetric",
    )

    assert torch.all(neuqi.zero_point == 0)
    neuqi_out = neuqi.round_to_nearest()[1]
    vanilla_out = vanilla.round_to_nearest()[1]
    assert weighted_reconstruction_energy(weights, neuqi_out, stats) <= (
        weighted_reconstruction_energy(weights, vanilla_out, stats) + 1e-6
    )


def test_neuqi_helper_matches_main_builder():
    weights = assignment_toy_weights()
    stats = toy_correlated_stats()
    main = build_neuqi_quantization_grid(
        weights,
        stats,
        bits=2,
        group_size=5,
        scale_candidates=32,
        coarse_candidates=8,
        candidate_chunk_size=4,
    )
    helper = build_asymmetric_neuqi_quantization_grid(
        weights,
        stats,
        bits=2,
        group_size=5,
        scale_candidates=32,
        coarse_candidates=8,
        candidate_chunk_size=4,
    )

    assert torch.allclose(helper.scale, main.scale)
    assert torch.allclose(helper.zero_point, main.zero_point)


def test_neuqi_symmetric_helper_matches_main_builder():
    weights = assignment_toy_weights()
    stats = toy_correlated_stats()
    main = build_neuqi_quantization_grid(
        weights,
        stats,
        bits=2,
        group_size=5,
        scheme="symmetric",
        scale_candidates=32,
        coarse_candidates=8,
        candidate_chunk_size=4,
    )
    helper = build_symmetric_neuqi_quantization_grid(
        weights,
        stats,
        bits=2,
        group_size=5,
        scale_candidates=32,
        coarse_candidates=8,
        candidate_chunk_size=4,
    )

    assert torch.allclose(helper.scale, main.scale)
    assert torch.allclose(helper.zero_point, main.zero_point)


def test_assignment_methods_run_on_neuqi_grid():
    grid = _neuqi_grid()
    stats = toy_correlated_stats()

    rtn_out, rtn_info = RTNAssignment().apply_to_grid(grid)
    gptq_out, gptq_info = GPTQAssignment().apply_to_grid(grid, stats)

    assert rtn_out.shape == (2, 5)
    assert gptq_out.shape == (2, 5)
    assert rtn_info["assignment"] == "rtn"
    assert gptq_info["assignment"] == "gptq"


def test_neuqi_scale_upper_ignores_padding_values():
    weights = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    stats = toy_correlated_stats()
    stats.diag_H = torch.ones(3)
    grid = build_neuqi_quantization_grid(
        weights,
        stats,
        bits=2,
        group_size=5,
        scale_candidates=16,
        coarse_candidates=4,
        candidate_chunk_size=2,
    )

    assert grid.scale[0, 0] <= (2.0 / 3.0) + 1e-6
