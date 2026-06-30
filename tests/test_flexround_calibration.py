from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment_methods.flexround import (  # noqa: E402
    FlexRoundCalibrationConfig,
    apply_flexround_artifact,
    calibrate_flexround_block,
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


def test_flexround_block_calibration_exports_official_quantizer_params():
    torch.manual_seed(0)
    inputs = [torch.randn(1, 3, 4) for _ in range(3)]
    block = _ToyBlock()
    config = FlexRoundCalibrationConfig(
        weight_bits=4,
        weight_symmetric=False,
        iters=4,
        batch_size=2,
        learning_rate=1e-3,
    )
    artifact, outputs, history = calibrate_flexround_block(
        block,
        inputs,
        [{}, {}, {}],
        config=config,
        device="cpu",
    )

    assert len(artifact) == 7
    assert len(outputs) == 3
    assert len(history) == 4
    assert torch.isfinite(torch.tensor(history)).all()
    for values in artifact.values():
        expected = {"weight", "codes", "delta1", "delta2", "delta3", "zero_point"}
        assert expected <= values.keys()
        assert values["weight"].shape == values["codes"].shape
        assert values["delta2"].shape == values["weight"].shape
        assert values["delta3"].shape == values["weight"][:, :1].shape

    apply_flexround_artifact(block, artifact)
    for name, values in artifact.items():
        assert torch.equal(
            block.get_submodule(name).weight,
            values["weight"].to(block.get_submodule(name).weight),
        )
    with torch.no_grad():
        applied_outputs = [block(values)[0] for values in inputs]
    for actual, expected in zip(outputs, applied_outputs):
        assert torch.allclose(actual, expected)


def test_flexround_can_retain_teacher_output_propagation():
    torch.manual_seed(0)
    block = _ToyBlock()
    inputs = [torch.randn(1, 3, 4)]
    with torch.no_grad():
        expected = block(inputs[0])[0]
    config = FlexRoundCalibrationConfig(
        weight_bits=4,
        iters=1,
        propagate_quantized_inputs=False,
    )

    _artifact, outputs, _history = calibrate_flexround_block(
        block,
        inputs,
        [{}],
        config=config,
        device="cpu",
    )

    assert torch.equal(outputs[0], expected)
