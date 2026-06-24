from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_baselines.flatquant_model import (  # noqa: E402
    flatquant_model_identity,
    load_flatquant_transforms,
    validate_flatquant_artifact_identity,
)
from grid_baselines.flatquant_calibration import (  # noqa: E402
    FlatQuantCalibrationConfig,
    factor_dimensions,
    calibrate_flatquant_block,
)
from scripts.check_flatquant_parity import parity_metrics  # noqa: E402
from scripts.calibrate_flatquant import capture_first_layer_inputs  # noqa: E402
from tests.test_flatquant_model_integration import (  # noqa: E402
    MODEL_FACTORIES,
    _tiny_llama,
)


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)

    def forward(self, values):
        mixed = (
            self.q_proj(values) + self.k_proj(values) + self.v_proj(values)
        ) / 3
        return self.o_proj(mixed)


class _ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(4, 8)
        self.gate_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)

    def forward(self, values):
        return self.down_proj(torch.sigmoid(self.gate_proj(values)) * self.up_proj(values))


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToyMLP()

    def forward(self, hidden_states, **_kwargs):
        return (hidden_states + self.self_attn(hidden_states) + self.mlp(hidden_states),)


def test_factor_dimensions_choose_exact_near_square_factors():
    assert factor_dimensions(8) == (2, 4)
    assert factor_dimensions(16) == (4, 4)
    assert factor_dimensions(7) == (1, 7)


def test_flatquant_block_calibration_exports_seven_linears():
    torch.manual_seed(0)
    inputs = [torch.randn(1, 3, 4) for _ in range(3)]
    config = FlatQuantCalibrationConfig(
        weight_bits=4,
        activation_bits=4,
        weight_group_size=4,
        activation_group_size=-1,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
    )
    artifact, outputs, history = calibrate_flatquant_block(
        _ToyBlock(),
        inputs,
        [{}, {}, {}],
        config=config,
        device="cpu",
    )

    assert len(artifact) == 7
    assert len(outputs) == 3
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(history)))
    for values in artifact.values():
        assert {"matrix_left", "matrix_right", "diag_scale"} <= values.keys()
        assert torch.all(values["weight_clip_max"] > 0)
        assert torch.all(values["activation_clip_max"] > 0)


def test_flatquant_artifact_identity_validation(tmp_path):
    model = _tiny_llama()
    identity = flatquant_model_identity(model)
    valid = tmp_path / "valid.pt"
    torch.save({"model": identity, "layers": {}}, valid)
    validate_flatquant_artifact_identity(valid, model)

    invalid = tmp_path / "invalid.pt"
    torch.save(
        {"model": {**identity, "hidden_size": identity["hidden_size"] + 1}, "layers": {}},
        invalid,
    )
    with pytest.raises(ValueError, match="hidden_size"):
        validate_flatquant_artifact_identity(invalid, model)

    unidentified = tmp_path / "unidentified.pt"
    torch.save({"layers": {}}, unidentified)
    with pytest.raises(ValueError, match="model identity"):
        validate_flatquant_artifact_identity(unidentified, model)

    settings = tmp_path / "settings.pt"
    torch.save(
        {
            "model": identity,
            "training": {"weight_bits": 4},
            "layers": {},
        },
        settings,
    )
    with pytest.raises(ValueError, match="quantization setting mismatch"):
        validate_flatquant_artifact_identity(
            settings,
            model,
            requested_quantization={"weight_bits": 3},
        )


def test_calibrated_artifact_is_loadable_by_runtime(tmp_path):
    torch.manual_seed(1)
    config = FlatQuantCalibrationConfig(
        weight_group_size=4,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
    )
    artifact, _outputs, _history = calibrate_flatquant_block(
        _ToyBlock(),
        [torch.randn(1, 2, 4)],
        [{}],
        config=config,
        device="cpu",
    )
    attention = {}
    if "self_attn.kcache_trans" in artifact:
        attention["self_attn"] = artifact.pop("self_attn.kcache_trans")["matrix"]
    path = tmp_path / "trained.pt"
    torch.save({"layers": artifact, "attention": attention}, path)
    transforms, clips = load_flatquant_transforms(path)
    assert len(transforms) == 7
    assert len(clips) == 7


def test_parity_metrics_report_exact_match():
    logits = torch.randn(2, 3, 5)
    metrics = parity_metrics(logits, logits.clone())
    assert metrics == {
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "top_token_agreement": 1.0,
    }


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_flatquant_calibration_runs_on_real_transformers_decoder_block(model_factory):
    torch.manual_seed(2)
    model = model_factory().float()
    inputs, kwargs = capture_first_layer_inputs(
        model,
        tokenizer=None,
        calibration=[torch.tensor([[1, 2, 3, 4]])],
        input_device=torch.device("cpu"),
    )
    config = FlatQuantCalibrationConfig(
        weight_group_size=4,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
    )
    artifact, outputs, history = calibrate_flatquant_block(
        model.model.layers[0],
        inputs,
        kwargs,
        config=config,
        device="cpu",
    )
    assert len(artifact) == 8
    assert outputs[0].shape == inputs[0].shape
    assert torch.isfinite(torch.tensor(history)).all()
