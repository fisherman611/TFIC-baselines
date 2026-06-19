from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment_methods import (  # noqa: E402
    GPTQAssignment,
    RTNAssignment,
    TFICAssignment,
)
from tests.examples import (  # noqa: E402
    reconstruction_error,
    toy_awq_grids,
    toy_correlated_stats,
    weighted_reconstruction_energy,
)


def test_rtn_assignment_runs_on_awq_grid():
    for grid in toy_awq_grids():
        out, info = RTNAssignment().apply_to_grid(grid)

        assert out.shape == (2, 5)
        assert info["assignment"] == "rtn"
        assert info["grid_scheme"] == grid.scheme


def test_gptq_assignment_runs_on_awq_grid():
    for grid in toy_awq_grids():
        stats = toy_correlated_stats()
        rtn_out, _ = RTNAssignment().apply_to_grid(grid)
        out, info = GPTQAssignment().apply_to_grid(grid, stats)
        weights = grid.float_weights[:, : grid.in_features]

        assert out.shape == (2, 5)
        assert info["assignment"] == "gptq"
        assert info["grid_scheme"] == grid.scheme
        assert weighted_reconstruction_energy(weights, out, stats) <= (
            weighted_reconstruction_energy(weights, rtn_out, stats) + 1e-6
        )


def test_tfic_assignment_runs_on_awq_grid():
    for grid in toy_awq_grids():
        stats = toy_correlated_stats()
        rtn_out, _ = RTNAssignment().apply_to_grid(grid)
        out, info = TFICAssignment(
            n_stages=2,
            sweeps=10,
            gmax=3,
            top_m=3,
            chunk_cols=1,
            kappa=0.0,
            gamma_th=-1.0,
        ).apply_to_grid(grid, stats)
        weights = grid.float_weights[:, : grid.in_features]

        assert out.shape == (2, 5)
        assert info["assignment"] == "tfic"
        assert info["grid_scheme"] == grid.scheme
        assert weighted_reconstruction_energy(weights, out, stats) <= (
            weighted_reconstruction_energy(weights, rtn_out, stats) + 1e-6
        )


def test_assignment_methods_can_separate_on_asymmetric_awq_case():
    grid = [candidate for candidate in toy_awq_grids() if candidate.scheme == "asymmetric"][0]
    stats = toy_correlated_stats()
    rtn_out, _ = RTNAssignment().apply_to_grid(grid)
    gptq_out, _ = GPTQAssignment().apply_to_grid(grid, stats)
    tfic_out, _ = TFICAssignment(
        n_stages=2,
        sweeps=10,
        gmax=3,
        top_m=3,
        chunk_cols=1,
        kappa=0.0,
        gamma_th=-1.0,
    ).apply_to_grid(grid, stats)

    assert not torch.allclose(gptq_out, rtn_out)
    assert not torch.allclose(tfic_out, rtn_out)


def _demo_assignment_methods_with_awq():
    stats = toy_correlated_stats()
    methods = [
        ("rtn", RTNAssignment(), None),
        ("gptq", GPTQAssignment(), stats),
        (
            "tfic",
            TFICAssignment(
                n_stages=2,
                sweeps=10,
                gmax=3,
                top_m=3,
                chunk_cols=1,
                kappa=0.0,
                gamma_th=-1.0,
            ),
            stats,
        ),
    ]

    for grid in tqdm(toy_awq_grids(), desc="awq schemes"):
        print("\n" + "=" * 70)
        print("awq grid scheme:", grid.scheme)
        weights = grid.float_weights[:, : grid.in_features]
        print("float weights:")
        print(weights)
        print("awq scales:")
        print(grid.awq_scales[:, : grid.in_features])
        for name, method, method_stats in tqdm(methods, desc="assignment methods"):
            if method_stats is None:
                out, info = method.apply_to_grid(grid)
            else:
                out, info = method.apply_to_grid(grid, method_stats)
            print(f"\n{name}")
            print("info:", {k: v for k, v in info.items() if k != "codes"})
            print("dequantized:")
            print(out)
            print("reconstruction error:")
            print(reconstruction_error(weights, out))
            print("weighted reconstruction energy:")
            print(weighted_reconstruction_energy(weights, out, stats))


if __name__ == "__main__":
    tests = [
        test_rtn_assignment_runs_on_awq_grid,
        test_gptq_assignment_runs_on_awq_grid,
        test_tfic_assignment_runs_on_awq_grid,
        test_assignment_methods_can_separate_on_asymmetric_awq_case,
    ]
    for test in tqdm(tests, desc="awq assignment tests"):
        test()
    print("\nall AWQ assignment-method tests passed")
    _demo_assignment_methods_with_awq()
