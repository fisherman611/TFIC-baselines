from __future__ import annotations

import torch

from eigenflip.statistics.trust_region import LayerStats
from grid_baselines import build_awq_quantization_grid, build_vanilla_quantization_grid


def reconstruction_error(weights: torch.Tensor, dequantized: torch.Tensor) -> dict[str, float]:
    error = dequantized.float() - weights.float()
    return {
        "mse": float((error * error).mean().item()),
        "mae": float(error.abs().mean().item()),
        "max_abs": float(error.abs().max().item()),
    }


def vanilla_manual_weights() -> torch.Tensor:
    return assignment_toy_weights()


def vanilla_padding_weights() -> torch.Tensor:
    return assignment_toy_weights()


def vanilla_fixed_grid_weights() -> torch.Tensor:
    return assignment_toy_weights()


def vanilla_new_assignment_weights() -> torch.Tensor:
    perturbation = torch.tensor(
        [
            [0.02, -0.01, 0.03, -0.04, 0.01],
            [-0.03, 0.02, -0.02, 0.01, 0.04],
        ],
        dtype=torch.float32,
    )
    return assignment_toy_weights() + perturbation


def vanilla_grid_demo_examples() -> list[torch.Tensor]:
    return [
        assignment_toy_weights(),
    ]


def assignment_toy_weights() -> torch.Tensor:
    return torch.tensor(
        [
            [0.8240, -0.4536, 0.0149, 0.6145, 0.3457],
            [0.1212, 0.8362, 0.2279, 0.6728, -0.8618],
        ],
        dtype=torch.float32,
    )


def awq_toy_scales() -> torch.Tensor:
    return torch.tensor([1.0, 0.5, 2.0, 1.5, 0.75], dtype=torch.float32)


def awq_toy_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return assignment_toy_weights(), awq_toy_scales()


def toy_correlated_hessian() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0000, 0.3682, -0.0586, -0.4635, 0.2768],
            [0.3682, 1.0000, 0.4363, -0.5636, 0.5847],
            [-0.0586, 0.4363, 1.0000, -0.4964, 0.0293],
            [-0.4635, -0.5636, -0.4964, 1.0000, -0.3726],
            [0.2768, 0.5847, 0.0293, -0.3726, 1.0000],
        ],
        dtype=torch.float32,
    )


def toy_correlated_stats() -> LayerStats:
    hessian = toy_correlated_hessian()
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


def weighted_reconstruction_energy(
    weights: torch.Tensor,
    dequantized: torch.Tensor,
    stats: LayerStats,
) -> float:
    hessian = stats.Sigma + torch.outer(stats.mu_hat, stats.mu_hat)
    residual = dequantized.float() - weights.float()
    return float((residual @ hessian * residual).sum().item())


def toy_vanilla_grid(scheme: str):
    return build_vanilla_quantization_grid(
        assignment_toy_weights(),
        bits=2,
        group_size=5,
        scheme=scheme,
    )


def toy_vanilla_grids():
    return [
        toy_vanilla_grid("symmetric"),
        toy_vanilla_grid("asymmetric"),
    ]


def toy_awq_grid(scheme: str):
    weights, awq_scales = awq_toy_inputs()
    return build_awq_quantization_grid(
        weights,
        awq_scales,
        bits=2,
        group_size=5,
        scheme=scheme,
    )


def toy_awq_grids():
    return [
        toy_awq_grid("symmetric"),
        toy_awq_grid("asymmetric"),
    ]
