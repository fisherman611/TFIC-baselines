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
from eigenflip.statistics.trust_region import LayerStats  # noqa: E402
from grid_baselines import build_vanilla_quantization_grid  # noqa: E402


def _reconstruction_error(weights: torch.Tensor, dequantized: torch.Tensor) -> dict[str, float]:
    error = dequantized.float() - weights.float()
    return {
        "mse": float((error * error).mean().item()),
        "mae": float(error.abs().mean().item()),
        "max_abs": float(error.abs().max().item()),
    }


def _toy_weights():
    return torch.tensor(
        [
            [0.8240, -0.4536, 0.0149, 0.6145, 0.3457],
            [0.1212, 0.8362, 0.2279, 0.6728, -0.8618],
        ],
        dtype=torch.float32,
    )


def _toy_vanilla_grid(scheme: str):
    return build_vanilla_quantization_grid(
        _toy_weights(),
        bits=2,
        group_size=5,
        scheme=scheme,
    )


def _toy_vanilla_grids():
    return [
        _toy_vanilla_grid("symmetric"),
        _toy_vanilla_grid("asymmetric"),
    ]


def _simple_vanilla_grid(scheme: str):
    weights = torch.tensor(
        [
            [-0.20, 0.00, 0.10, 0.26],
            [-0.50, 0.20, 0.90, 0.00],
        ],
        dtype=torch.float32,
    )
    return build_vanilla_quantization_grid(weights, bits=3, group_size=4, scheme=scheme)


def _toy_correlated_stats():
    hessian = torch.tensor(
        [
            [1.0000, 0.3682, -0.0586, -0.4635, 0.2768],
            [0.3682, 1.0000, 0.4363, -0.5636, 0.5847],
            [-0.0586, 0.4363, 1.0000, -0.4964, 0.0293],
            [-0.4635, -0.5636, -0.4964, 1.0000, -0.3726],
            [0.2768, 0.5847, 0.0293, -0.3726, 1.0000],
        ],
        dtype=torch.float32,
    )
    d = hessian.shape[0]
    mu = torch.zeros(d, dtype=torch.float32)
    diag = torch.diagonal(hessian).clone()
    return LayerStats(
        d=d,
        mu_hat=mu,
        diag_H=diag,
        diag_Sigma=diag,
        Sigma=hessian,
        backend="toy_correlated",
    ).build()


def _weighted_reconstruction_energy(
    weights: torch.Tensor,
    dequantized: torch.Tensor,
    stats: LayerStats,
) -> float:
    hessian = stats.Sigma + torch.outer(stats.mu_hat, stats.mu_hat)
    residual = dequantized.float() - weights.float()
    return float((residual @ hessian * residual).sum().item())


def test_rtn_assignment_runs_on_vanilla_grid():
    for grid in _toy_vanilla_grids():
        out, info = RTNAssignment().apply_to_grid(grid)

        assert out.shape == (2, 5)
        assert info["assignment"] == "rtn"
        assert info["grid_scheme"] == grid.scheme


def test_gptq_assignment_runs_on_vanilla_grid():
    for grid in _toy_vanilla_grids():
        stats = _toy_correlated_stats()
        rtn_out, _ = RTNAssignment().apply_to_grid(grid)
        out, info = GPTQAssignment().apply_to_grid(grid, stats)

        assert out.shape == (2, 5)
        assert info["assignment"] == "gptq"
        assert info["grid_scheme"] == grid.scheme
        assert _weighted_reconstruction_energy(grid.float_weights, out, stats) <= (
            _weighted_reconstruction_energy(grid.float_weights, rtn_out, stats) + 1e-6
        )


def test_tfic_assignment_runs_on_vanilla_grid():
    for grid in _toy_vanilla_grids():
        stats = _toy_correlated_stats()
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

        assert out.shape == (2, 5)
        assert info["assignment"] == "tfic"
        assert info["grid_scheme"] == grid.scheme
        assert _weighted_reconstruction_energy(grid.float_weights, out, stats) <= (
            _weighted_reconstruction_energy(grid.float_weights, rtn_out, stats) + 1e-6
        )


def test_assignment_methods_can_separate_on_asymmetric_toy_case():
    grid = _toy_vanilla_grid("asymmetric")
    stats = _toy_correlated_stats()
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
    assert not torch.allclose(tfic_out, gptq_out)


def _demo_assignment_methods():
    stats = _toy_correlated_stats()
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

    for grid in tqdm(_toy_vanilla_grids(), desc="vanilla schemes"):
        print("\n" + "=" * 70)
        print("vanilla grid scheme:", grid.scheme)
        weights = grid.float_weights[:, : grid.in_features]
        print("float weights:")
        print(weights)
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
            print(_reconstruction_error(weights, out))
            print("weighted reconstruction energy:")
            print(_weighted_reconstruction_energy(weights, out, stats))


if __name__ == "__main__":
    tests = [
        test_rtn_assignment_runs_on_vanilla_grid,
        test_gptq_assignment_runs_on_vanilla_grid,
        test_tfic_assignment_runs_on_vanilla_grid,
        test_assignment_methods_can_separate_on_asymmetric_toy_case,
    ]
    for test in tqdm(tests, desc="vanilla assignment tests"):
        test()
    print("\nall vanilla assignment-method tests passed")
    _demo_assignment_methods()
