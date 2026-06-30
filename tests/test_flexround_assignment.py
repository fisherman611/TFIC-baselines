from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from assignment_methods import FlexRoundAssignment, RTNAssignment
from eigenflip.statistics.trust_region import LayerStats
from grid_baselines import build_vanilla_quantization_grid
from scripts.run_quantization_baseline import assignment_needs_h, build_grid
from tests.examples import (
    assignment_toy_weights,
    toy_awq_grids,
    toy_correlated_stats,
    toy_vanilla_grids,
    weighted_reconstruction_energy,
)


def flexround_correlated_stats() -> LayerStats:
    """Represent the toy Hessian exactly as the D + V V^T surrogate."""

    source = toy_correlated_stats()
    hessian = source.Sigma
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    return LayerStats(
        d=source.d,
        mu_hat=source.mu_hat,
        diag_H=source.diag_H,
        diag_Sigma=source.diag_Sigma,
        U_k=eigenvectors,
        Lam_k=eigenvalues.clamp_min(0),
        Sigma=hessian,
        backend="toy_correlated_low_rank",
    ).build()


@pytest.mark.parametrize("grid", toy_vanilla_grids() + toy_awq_grids())
def test_flexround_runs_on_fixed_grids_and_returns_valid_final_codes(grid):
    stats = flexround_correlated_stats()
    initial_scale = grid.scale.clone()
    output, info = FlexRoundAssignment(steps=300, lr=3e-2).apply_to_grid(
        grid,
        stats,
    )
    weights = grid.float_weights[:, : grid.in_features]

    assert output.shape == weights.shape
    assert output.dtype == grid.original_dtype
    assert info["assignment"] == "flexround"
    assert info["variant"] == "official_quantizer_surrogate"
    assert info["grid_scheme"] == grid.scheme
    assert not torch.allclose(grid.scale, initial_scale)
    assert torch.isfinite(torch.tensor(info["initial_loss"]))
    assert torch.isfinite(torch.tensor(info["final_loss"]))
    assert info["final_loss"] * weights.shape[0] == pytest.approx(
        weighted_reconstruction_energy(weights, output, stats),
        rel=5e-6,
        abs=2e-6,
    )
    assert torch.all(info["codes"] >= grid.qmin)
    assert torch.all(info["codes"] <= grid.qmax)


def test_flexround_changes_codes_on_correlated_case():
    grid = toy_vanilla_grids()[1]
    rtn_output, _ = RTNAssignment().apply_to_grid(grid)
    output, info = FlexRoundAssignment(steps=300, lr=3e-2).apply_to_grid(
        grid,
        flexround_correlated_stats(),
    )

    assert info["final_loss"] < info["initial_loss"]
    assert not torch.allclose(output, rtn_output)


def test_flexround_zero_steps_is_exactly_rtn():
    grid = toy_awq_grids()[0]
    flexround_output, info = FlexRoundAssignment(steps=0).apply_to_grid(
        grid,
        flexround_correlated_stats(),
    )
    rtn_output, rtn_info = RTNAssignment().apply_to_grid(grid)

    assert torch.equal(info["codes"], rtn_info["codes"])
    assert torch.equal(flexround_output, rtn_output)
    assert info["changed_codes"] == 0


def test_flexround_assignment_uses_per_channel_qparams_on_grouped_grid():
    grid = build_vanilla_quantization_grid(
        assignment_toy_weights(),
        bits=2,
        group_size=2,
        scheme="asymmetric",
    )
    assert not torch.allclose(grid.scale, grid.scale[:, :1].expand_as(grid.scale))

    FlexRoundAssignment(steps=0).apply_to_grid(grid, flexround_correlated_stats())

    assert torch.allclose(grid.scale, grid.scale[:, :1].expand_as(grid.scale))
    assert torch.allclose(
        grid.zero_point,
        grid.zero_point[:, :1].expand_as(grid.zero_point),
    )


def test_flexround_assignment_keeps_awq_scales_with_per_channel_qparams():
    grid = toy_awq_grids()[0]

    FlexRoundAssignment(steps=0).apply_to_grid(grid, flexround_correlated_stats())

    scaled_qparams = grid.scale * grid.awq_scales
    assert torch.allclose(
        scaled_qparams,
        scaled_qparams[:, :1].expand_as(scaled_qparams),
    )


@pytest.mark.parametrize("grid", toy_vanilla_grids() + toy_awq_grids())
def test_flexround_fixed_grid_mode_preserves_qparams(grid):
    initial_scale = grid.scale.clone()
    initial_zero_point = grid.zero_point.clone()

    _output, info = FlexRoundAssignment(
        steps=2,
        lr=1e-3,
        learn_layer_scale=False,
    ).apply_to_grid(grid, flexround_correlated_stats())

    assert info["variant"] == "fixed_grid_surrogate"
    assert torch.equal(grid.scale, initial_scale)
    assert torch.equal(grid.zero_point, initial_zero_point)


def test_flexround_runner_builds_channelwise_grid():
    weights = assignment_toy_weights()
    args = SimpleNamespace(
        assignment="flexround",
        bits=2,
        group_size=-1,
        scheme="asymmetric",
    )

    grid = build_grid("vanilla", weights, args, None, None, None)

    assert grid.group_size == weights.shape[1]


def test_flexround_k_zero_uses_lightweight_statistics_path():
    assert assignment_needs_h("flexround", 0) is False
    assert assignment_needs_h("flexround", 16) is True
    assert assignment_needs_h("gptq", 0) is True
    assert assignment_needs_h("tfic", 0) is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": -1}, "steps"),
        ({"lr": 0.0}, "lr"),
        ({"log_divisor_bound": 0.0}, "log_divisor_bound"),
    ],
)
def test_flexround_rejects_invalid_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FlexRoundAssignment(**kwargs)
