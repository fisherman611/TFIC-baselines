from __future__ import annotations

import sys
from pathlib import Path

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
    toy_correlated_stats,
    toy_flatquant_diag_grids,
    weighted_reconstruction_energy,
)


def test_rtn_assignment_runs_on_flatquant_diag_grid():
    for grid in toy_flatquant_diag_grids():
        out, info = RTNAssignment().apply_to_grid(grid)

        assert out.shape == (2, 5)
        assert info["assignment"] == "rtn"
        assert info["grid_scheme"] == grid.scheme


def test_gptq_assignment_runs_on_flatquant_diag_grid():
    for grid in toy_flatquant_diag_grids():
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


def test_tfic_assignment_runs_on_flatquant_diag_grid():
    for grid in toy_flatquant_diag_grids():
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


if __name__ == "__main__":
    tests = [
        test_rtn_assignment_runs_on_flatquant_diag_grid,
        test_gptq_assignment_runs_on_flatquant_diag_grid,
        test_tfic_assignment_runs_on_flatquant_diag_grid,
    ]
    for test in tqdm(tests, desc="flatquant_diag assignment tests"):
        test()
    print("\nall FlatQuant diagonal-scale assignment-method tests passed")
