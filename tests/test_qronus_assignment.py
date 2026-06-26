from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from assignment_methods import (
    GPTQAssignment,
    QronusAssignment,
    stats_from_paired_inputs,
)
from scripts.run_quantization_baseline import build_grid
from tests.test_gptaq_assignment import paired_inputs
from tests.examples import toy_awq_grids, toy_vanilla_grids


@pytest.mark.parametrize("grid", toy_vanilla_grids() + toy_awq_grids())
def test_qronus_reduces_to_gptq_when_paired_inputs_are_identical(grid):
    inputs = paired_inputs()
    stats = stats_from_paired_inputs(inputs, inputs)
    gptq_output, _ = GPTQAssignment(damp=0.0, order="natural").apply_to_grid(
        grid,
        stats,
    )
    qronus_output, info = QronusAssignment(
        alpha=0.0,
        act_order=False,
    ).apply_to_grid(grid, stats)

    assert info["assignment"] == "qronus"
    assert info["grid_scheme"] == grid.scheme
    assert torch.equal(qronus_output, gptq_output)
    assert torch.all(info["codes"] >= grid.qmin)
    assert torch.all(info["codes"] <= grid.qmax)


@pytest.mark.parametrize("grid", toy_vanilla_grids() + toy_awq_grids())
def test_qronus_runs_on_fixed_grids_and_returns_valid_codes(grid):
    quantized = paired_inputs()
    reference = quantized + 0.25 * torch.flip(quantized, dims=[1])
    stats = stats_from_paired_inputs(quantized, reference)
    output, info = QronusAssignment(alpha=1e-6).apply_to_grid(grid, stats)

    assert output.shape == grid.float_weights[:, : grid.in_features].shape
    assert output.dtype == grid.original_dtype
    assert info["grid_scheme"] == grid.scheme
    assert info["alpha"] == 1e-6
    assert info["act_order"] is True
    assert torch.all(info["codes"] >= grid.qmin)
    assert torch.all(info["codes"] <= grid.qmax)


def test_qronus_runner_builds_channelwise_grid():
    weights = torch.tensor(
        [
            [-1.0, -0.5, 0.25, 0.75, 1.5],
            [-2.0, -0.25, 0.5, 1.0, 2.0],
        ]
    )
    args = SimpleNamespace(
        assignment="qronus",
        bits=3,
        group_size=-1,
        scheme="asymmetric",
    )

    grid = build_grid("vanilla", weights, args, None, None, None)

    assert grid.group_size == weights.shape[1]
    assert grid.padded_in_features == weights.shape[1]
    assert torch.all(grid.scale == grid.scale[:, :1])
    assert torch.all(grid.zero_point == grid.zero_point[:, :1])


def test_non_qronus_runner_keeps_requested_group_size():
    weights = torch.randn(2, 5)
    args = SimpleNamespace(
        assignment="gptq",
        bits=3,
        group_size=2,
        scheme="asymmetric",
    )

    grid = build_grid("vanilla", weights, args, None, None, None)

    assert grid.group_size == 2
    assert grid.padded_in_features == 6


def test_qronus_requires_paired_cross_moment():
    grid = toy_vanilla_grids()[0]
    inputs = paired_inputs()
    stats = stats_from_paired_inputs(inputs, inputs)
    stats.delta_cross = None

    with pytest.raises(ValueError, match="delta_cross"):
        QronusAssignment().apply_to_grid(grid, stats)


def test_qronus_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="alpha"):
        QronusAssignment(alpha=-0.1)
