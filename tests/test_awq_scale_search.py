from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eigenflip.quantization.awq_scales import (  # noqa: E402
    _groupwise_quant,
    compute_awq_clip,
    compute_awq_scales,
    layer_params_from_awq_run,
)
from scripts.generate_awq_scales import (  # noqa: E402
    AWQScaleAccumulator,
    effective_awq_group_size,
)
from scripts.run_quantization_baseline import load_awq_layer_params  # noqa: E402


def _group_outputs(weights: torch.Tensor, inputs: torch.Tensor, group_size: int):
    rows, in_features = weights.shape
    groups = in_features // group_size
    wg = weights.reshape(rows, groups, group_size)
    xg = inputs.reshape(inputs.shape[0], groups, group_size)
    return torch.einsum("cgd,tgd->ctg", wg, xg)


def test_accumulator_uses_mean_absolute_activation_scale():
    accumulator = AWQScaleAccumulator(in_features=3, sample_tokens=3)
    activations = torch.tensor(
        [[[-2.0, 1.0, 0.0], [4.0, -3.0, 2.0]], [[1.0, 2.0, -4.0], [3.0, 0.0, 2.0]]]
    )
    accumulator.add(activations)

    assert torch.allclose(
        accumulator.activation_scale(), activations.reshape(-1, 3).abs().mean(0)
    )
    assert accumulator.x_sample().shape == (3, 3)


def test_awq_scale_search_returns_normalized_positive_scales():
    torch.manual_seed(0)
    weights = torch.randn(4, 8)
    inputs = torch.randn(32, 8)
    activation_scale = inputs.abs().mean(0)

    scales, alpha, error = compute_awq_scales(
        weights, activation_scale, inputs, bits=3, group_size=4, n_grid=20
    )

    assert alpha in {index / 20 for index in range(20)}
    assert error >= 0
    assert torch.isfinite(scales).all()
    assert torch.all(scales > 0)
    assert torch.allclose(scales.max() * scales.min(), torch.tensor(1.0), atol=1e-5)


def test_awq_generation_group_size_minus_one_is_channelwise():
    weights = torch.randn(4, 7)

    assert effective_awq_group_size(-1, weights) == 7
    assert effective_awq_group_size(4, weights) == 4


def test_awq_clip_search_does_not_increase_sampled_group_error():
    torch.manual_seed(1)
    weights = torch.randn(6, 8)
    inputs = torch.randn(48, 8)
    clip_max = compute_awq_clip(
        weights,
        inputs,
        bits=3,
        group_size=4,
        n_grid=20,
        max_shrink=0.5,
        output_chunk_size=3,
    )
    baseline = _groupwise_quant(weights, 3, 4)
    clipped = _groupwise_quant(weights, 3, 4, clip_max=clip_max)
    target = _group_outputs(weights, inputs, 4)
    baseline_error = (_group_outputs(baseline, inputs, 4) - target).pow(2).mean((1, 2))
    clipped_error = (_group_outputs(clipped, inputs, 4) - target).pow(2).mean((1, 2))

    assert clip_max.shape == (6, 2, 1)
    assert torch.all(clipped_error <= baseline_error + 1e-6)


def test_awq_artifact_loader_keeps_clipping_and_legacy_format():
    scales = torch.tensor([1.0, 2.0])
    clip_max = torch.ones(3, 1, 1)
    current = layer_params_from_awq_run(
        {"layer": {"scales": scales, "clip_max": clip_max, "bits": 3}}
    )
    legacy = layer_params_from_awq_run({"layer": scales})

    assert torch.equal(current["layer"]["clip_max"], clip_max)
    assert current["layer"]["bits"] == 3
    assert torch.equal(legacy["layer"]["scales"], scales)


def test_awq_runner_rejects_mismatched_artifact(tmp_path):
    path = tmp_path / "awq.pt"
    torch.save(
        {
            "layer": {
                "scales": torch.ones(4),
                "bits": 4,
                "group_size": 128,
                "scheme": "asymmetric",
                "model_path": "model-a",
            }
        },
        path,
    )
    args = SimpleNamespace(
        bits=3, group_size=128, scheme="asymmetric", model_path="model-a"
    )

    with pytest.raises(ValueError, match="bits=4"):
        load_awq_layer_params(str(path), args)
