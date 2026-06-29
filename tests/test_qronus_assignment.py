from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from assignment_methods import (
    GPTQAssignment,
    QronusAssignment,
    stats_from_paired_inputs,
)
from scripts.run_quantization_baseline import (
    apply_qronus_paper_preset,
    build_grid,
    parse_args,
)
from tests.test_gptaq_assignment import paired_inputs
from tests.examples import toy_awq_grids, toy_vanilla_grids


def qronus_algorithm_one_oracle(grid, stats, *, alpha, act_order):
    """Direct implementation of Algorithm 1 with paper-style damping."""

    dtype = torch.float64
    weights = grid.float_weights.to(dtype).clone()
    scale = grid.scale.to(dtype)
    zero_point = grid.zero_point.to(dtype)
    mean = stats.mu_hat.to(dtype)
    h_raw = stats.Sigma.to(dtype) + torch.outer(mean, mean)
    g_raw = h_raw + stats.delta_cross.to(dtype).t()

    inverse_permutation = None
    if act_order:
        permutation = torch.argsort(torch.diagonal(h_raw), descending=True)
        inverse_permutation = torch.argsort(permutation)
        weights = weights[:, permutation]
        scale = scale[:, permutation]
        zero_point = zero_point[:, permutation]
        h_raw = h_raw[permutation][:, permutation]
        g_raw = g_raw[permutation][:, permutation]

    damping = alpha * torch.linalg.eigvalsh(h_raw).amax()
    identity = torch.eye(h_raw.shape[0], dtype=dtype)
    h_damped = h_raw + damping * identity
    g_damped = g_raw + damping * identity
    codes = torch.empty_like(weights)

    first_values = (
        weights @ g_raw[0] - weights[:, 1:] @ h_raw[0, 1:]
    ) / h_raw[0, 0]
    codes[:, 0] = torch.round(
        first_values / scale[:, 0] + zero_point[:, 0]
    ).clamp(grid.qmin, grid.qmax)
    first_dequantized = (codes[:, 0] - zero_point[:, 0]) * scale[:, 0]
    rhs = weights @ g_damped[1:].t()
    rhs -= first_dequantized.unsqueeze(1) * h_raw[1:, 0].unsqueeze(0)
    weights[:, 1:] = torch.linalg.solve(h_damped[1:, 1:], rhs.t()).t()

    inverse_hessian = torch.linalg.inv(h_damped)
    lower_factor = torch.linalg.cholesky(inverse_hessian)
    for index in range(1, weights.shape[1]):
        column = weights[:, index].clone()
        codes[:, index] = torch.round(
            column / scale[:, index] + zero_point[:, index]
        ).clamp(grid.qmin, grid.qmax)
        if index + 1 < weights.shape[1]:
            dequantized = (
                codes[:, index] - zero_point[:, index]
            ) * scale[:, index]
            error = column - dequantized
            factor = (
                lower_factor[index + 1 :, index]
                / lower_factor[index, index]
            )
            weights[:, index + 1 :] -= error.unsqueeze(1) * factor.unsqueeze(0)

    if inverse_permutation is not None:
        codes = codes[:, inverse_permutation]
    return grid.dequantize(codes), codes


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


@pytest.mark.parametrize("act_order", [False, True])
def test_qronus_matches_algorithm_one_with_damping(act_order):
    grid = toy_vanilla_grids()[1]
    quantized = paired_inputs()
    reference = quantized + 0.25 * torch.flip(quantized, dims=[1])
    stats = stats_from_paired_inputs(quantized, reference)

    expected, expected_codes = qronus_algorithm_one_oracle(
        grid,
        stats,
        alpha=1e-3,
        act_order=act_order,
    )
    actual, info = QronusAssignment(
        alpha=1e-3,
        act_order=act_order,
    ).apply_to_grid(grid, stats)

    assert torch.equal(info["codes"], expected_codes)
    assert torch.equal(actual, expected)


def test_qronus_damped_matched_inputs_preserve_first_rtn_code():
    grid = toy_vanilla_grids()[1]
    inputs = paired_inputs()
    stats = stats_from_paired_inputs(inputs, inputs)

    _output, info = QronusAssignment(
        alpha=0.5,
        act_order=False,
    ).apply_to_grid(grid, stats)
    rtn_codes = grid.quantize()

    assert torch.equal(info["codes"][:, 0], rtn_codes[:, 0])


def test_qronus_damped_matched_inputs_reduce_to_gptq():
    grid = toy_vanilla_grids()[1]
    inputs = paired_inputs()
    stats = stats_from_paired_inputs(inputs, inputs)
    hessian = stats.Sigma + torch.outer(stats.mu_hat, stats.mu_hat)
    alpha = 0.5
    damping = alpha * torch.linalg.eigvalsh(hessian).amax()
    gptq_damp = float(damping / torch.diagonal(hessian).mean())

    expected, _ = GPTQAssignment(
        damp=gptq_damp,
        order="natural",
    ).apply_to_grid(grid, stats)
    actual, _ = QronusAssignment(
        alpha=alpha,
        act_order=False,
    ).apply_to_grid(grid, stats)

    assert torch.equal(actual, expected)


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


def test_qronus_paper_preset_sets_reproduction_defaults():
    args = SimpleNamespace(
        assignment="qronus",
        qronus_paper_preset=True,
        calib_dataset="c4",
        n_calib=16,
        seqlen=512,
        group_size=128,
        qronus_act_order=False,
    )

    configured = apply_qronus_paper_preset(args)

    assert configured.calib_dataset == "c4"
    assert configured.n_calib == 128
    assert configured.seqlen == 2048
    assert configured.group_size == -1
    assert configured.qronus_act_order is True


def test_qronus_cache_dtype_defaults_to_bfloat16(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_quantization_baseline.py",
            "--model-path",
            "dummy-model",
            "--grid",
            "vanilla",
            "--assignment",
            "qronus",
            "--group-size",
            "-1",
        ],
    )

    args = parse_args()

    assert args.qronus_cache_dtype == "bfloat16"


def test_qronus_requires_explicit_channelwise_group_size():
    args = SimpleNamespace(
        assignment="qronus",
        qronus_paper_preset=False,
        group_size=128,
    )

    with pytest.raises(ValueError, match="group-size -1"):
        apply_qronus_paper_preset(args)


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
