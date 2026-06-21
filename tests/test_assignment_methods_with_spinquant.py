from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment_methods import GPTQAssignment, RTNAssignment, TFICAssignment  # noqa: E402
from grid_baselines import build_spinquant_quantization_grid  # noqa: E402
from tests.examples import (  # noqa: E402
    assignment_toy_weights,
    toy_correlated_stats,
    weighted_reconstruction_energy,
)


def _spinquant_grids():
    weights = assignment_toy_weights()
    return [
        build_spinquant_quantization_grid(
            weights, bits=2, group_size=5, scheme='symmetric'
        ),
        build_spinquant_quantization_grid(
            weights, bits=2, group_size=5, scheme='asymmetric'
        ),
    ]


def test_rtn_assignment_runs_on_spinquant_grid():
    for grid in _spinquant_grids():
        out, info = RTNAssignment().apply_to_grid(grid)
        assert out.shape == (2, 5)
        assert info['assignment'] == 'rtn'


def test_gptq_assignment_runs_on_spinquant_grid():
    for grid in _spinquant_grids():
        stats = toy_correlated_stats()
        rtn_out, _ = RTNAssignment().apply_to_grid(grid)
        out, info = GPTQAssignment().apply_to_grid(grid, stats)
        weights = grid.float_weights[:, : grid.in_features]
        assert out.shape == (2, 5)
        assert info['assignment'] == 'gptq'
        assert weighted_reconstruction_energy(weights, out, stats) <= (
            weighted_reconstruction_energy(weights, rtn_out, stats) + 1e-6
        )


def test_tfic_assignment_runs_on_spinquant_grid():
    for grid in _spinquant_grids():
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
        assert info['assignment'] == 'tfic'
        assert weighted_reconstruction_energy(weights, out, stats) <= (
            weighted_reconstruction_energy(weights, rtn_out, stats) + 1e-6
        )
