from __future__ import annotations

import pytest
import torch

from assignment_methods import (
    GPTAQAssignment,
    GPTAQResCompAssignment,
    GPTQAssignment,
    stats_from_paired_inputs,
)
from grid_baselines import build_vanilla_quantization_grid
from tests.examples import toy_awq_grids, toy_vanilla_grids


def paired_inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.2, -0.4, 0.7, -0.1],
            [0.3, -0.8, 0.5, 0.1, 0.9],
            [-0.6, 0.4, 0.8, -0.3, 0.2],
            [0.5, 0.7, -0.2, -0.9, 0.4],
            [-0.1, 0.6, 0.3, 0.8, -0.7],
            [0.9, -0.5, 0.1, 0.2, 0.6],
            [-0.4, -0.2, 0.7, 0.5, 0.3],
            [0.2, 0.9, -0.6, 0.4, -0.5],
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("grid", toy_vanilla_grids() + toy_awq_grids())
def test_gptaq_reduces_to_gptq_when_paired_inputs_are_identical(grid):
    inputs = paired_inputs()
    stats = stats_from_paired_inputs(inputs, inputs)
    gptq_output, _ = GPTQAssignment(damp=0.01, order="natural").apply_to_grid(
        grid,
        stats,
    )
    gptaq_output, info = GPTAQAssignment(
        damp=0.01,
        block_size=2,
    ).apply_to_grid(grid, stats)

    assert info["assignment"] == "gptaq"
    assert info["grid_scheme"] == grid.scheme
    assert torch.equal(gptaq_output, gptq_output)
    assert torch.all(info["codes"] >= grid.qmin)
    assert torch.all(info["codes"] <= grid.qmax)


def test_gptaq_paired_stats_match_manual_moments():
    quantized = paired_inputs()
    reference = quantized + 0.1 * torch.flip(quantized, dims=[1])
    stats = stats_from_paired_inputs(quantized, reference)
    count = quantized.shape[0]

    assert torch.allclose(
        stats.Sigma + torch.outer(stats.mu_hat, stats.mu_hat),
        quantized.t() @ quantized / count,
    )
    assert torch.allclose(
        stats.delta_cross,
        (reference - quantized).t() @ quantized / count,
    )


def test_gptaq_requires_paired_cross_moment():
    grid = toy_vanilla_grids()[0]
    inputs = paired_inputs()
    stats = stats_from_paired_inputs(inputs, inputs)
    stats.delta_cross = None

    with pytest.raises(ValueError, match="delta_cross"):
        GPTAQAssignment().apply_to_grid(grid, stats)


def test_gptaq_uses_asymmetry_to_improve_paired_output_reconstruction():
    grid = toy_vanilla_grids()[1]
    weights = grid.float_weights[:, : grid.in_features].double()
    matched = False

    # Search a small deterministic family of asymmetric calibration cases.
    # The assertion verifies that the P correction is active and can improve
    # the actual paired objective ||W_hat X_quant - W X_full||^2.
    for seed in range(32):
        generator = torch.Generator().manual_seed(seed)
        quantized_inputs = torch.randn(
            32, 5, generator=generator, dtype=torch.float64
        )
        full_precision_inputs = quantized_inputs + 0.5 * torch.randn(
            32, 5, generator=generator, dtype=torch.float64
        )
        stats = stats_from_paired_inputs(
            quantized_inputs,
            full_precision_inputs,
        )
        gptq_output, _ = GPTQAssignment(
            damp=0.01,
            order="natural",
        ).apply_to_grid(grid, stats)
        gptaq_output, info = GPTAQAssignment(
            damp=0.01,
            block_size=2,
            alpha=0.25,
        ).apply_to_grid(grid, stats)
        target = weights @ full_precision_inputs.t()
        gptq_loss = ((gptq_output.double() @ quantized_inputs.t()) - target).square().mean()
        gptaq_loss = ((gptaq_output.double() @ quantized_inputs.t()) - target).square().mean()
        if info["changed_codes"] > 0 and gptaq_loss < gptq_loss:
            matched = True
            break

    assert matched


@pytest.mark.parametrize("grid", toy_vanilla_grids() + toy_awq_grids())
def test_gptaq_rescomp_zero_rescomp_alpha_matches_gptaq(grid):
    quantized = paired_inputs()
    reference = quantized + 0.25 * torch.flip(quantized, dims=[1])
    stats = stats_from_paired_inputs(quantized, reference)

    gptaq_output, gptaq_info = GPTAQAssignment(
        damp=0.01,
        block_size=2,
        alpha=0.25,
    ).apply_to_grid(grid, stats)
    rescomp_output, rescomp_info = GPTAQResCompAssignment(
        damp=0.01,
        block_size=2,
        alpha=0.25,
        rescomp_alpha=0.0,
    ).apply_to_grid(grid, stats)

    assert rescomp_info["assignment"] == "gptaq_rescomp"
    assert torch.equal(rescomp_info["codes"], gptaq_info["codes"])
    assert torch.equal(rescomp_output, gptaq_output)


@pytest.mark.parametrize("grid", toy_vanilla_grids() + toy_awq_grids())
def test_gptaq_rescomp_runs_on_fixed_grids_and_returns_valid_codes(grid):
    quantized = paired_inputs()
    reference = quantized + 0.25 * torch.flip(quantized, dims=[1])
    stats = stats_from_paired_inputs(quantized, reference)
    output, info = GPTAQResCompAssignment(
        damp=0.01,
        block_size=2,
        alpha=0.25,
        rescomp_alpha=1.0,
    ).apply_to_grid(grid, stats)

    assert output.shape == grid.float_weights[:, : grid.in_features].shape
    assert output.dtype == grid.original_dtype
    assert info["grid_scheme"] == grid.scheme
    assert info["rescomp_alpha"] == 1.0
    assert torch.all(info["codes"] >= grid.qmin)
    assert torch.all(info["codes"] <= grid.qmax)


def test_gptaq_rescomp_returns_expected_defaults():
    grid = build_vanilla_quantization_grid(
        torch.tensor(
            [
                [0.8240, -0.4536, 0.0149, 0.6145, 0.3457],
                [0.1212, 0.8362, 0.2279, 0.6728, -0.8618],
            ],
            dtype=torch.float32,
        ),
        bits=3,
        group_size=5,
        scheme="asymmetric",
    )
    quantized = paired_inputs()
    reference = quantized + 0.25 * torch.flip(quantized, dims=[1])
    stats = stats_from_paired_inputs(quantized, reference)

    _output, info = GPTAQResCompAssignment(
        damp=0.01,
        block_size=2,
        alpha=1.0,
    ).apply_to_grid(grid, stats)

    assert info["rescomp_alpha"] == 0.25
    assert torch.all(info["codes"] >= grid.qmin)
    assert torch.all(info["codes"] <= grid.qmax)


@pytest.mark.parametrize(("bits", "expected_mode"), [(2, "org"), (3, "allw")])
def test_gptaq_rescomp_auto_mode_matches_rescomp_bit_policy(bits, expected_mode):
    grid = build_vanilla_quantization_grid(
        torch.tensor(
            [
                [0.8240, -0.4536, 0.0149, 0.6145, 0.3457],
                [0.1212, 0.8362, 0.2279, 0.6728, -0.8618],
            ],
            dtype=torch.float32,
        ),
        bits=bits,
        group_size=5,
        scheme="asymmetric",
    )
    quantized = paired_inputs()
    reference = quantized + 0.25 * torch.flip(quantized, dims=[1])
    stats = stats_from_paired_inputs(quantized, reference)

    auto_output, auto_info = GPTAQResCompAssignment(
        damp=0.01,
        block_size=2,
        rescomp_mode="auto",
    ).apply_to_grid(grid, stats)
    explicit_output, explicit_info = GPTAQResCompAssignment(
        damp=0.01,
        block_size=2,
        rescomp_mode=expected_mode,
    ).apply_to_grid(grid, stats)

    assert auto_info["rescomp_mode"] == "auto"
    assert explicit_info["rescomp_mode"] == expected_mode
    assert torch.equal(auto_info["codes"], explicit_info["codes"])
    assert torch.equal(auto_output, explicit_output)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"damp": -0.1}, "damp"),
        ({"block_size": 0}, "block_size"),
        ({"alpha": -1.0}, "alpha"),
    ],
)
def test_gptaq_rejects_invalid_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GPTAQAssignment(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"damp": -0.1}, "damp"),
        ({"block_size": 0}, "block_size"),
        ({"alpha": -1.0}, "alpha"),
        ({"rescomp_alpha": -1.0}, "rescomp_alpha"),
        ({"rescomp_mode": "bad"}, "rescomp_mode"),
    ],
)
def test_gptaq_rescomp_rejects_invalid_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GPTAQResCompAssignment(**kwargs)
