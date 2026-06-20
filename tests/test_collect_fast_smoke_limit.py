from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from eigenflip.statistics.collect_fast import collect_and_encode_awq_style


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 3, bias=False)
        self.second = nn.Linear(3, 2, bias=False)

    def forward(self, input_ids, use_cache=False):
        del use_cache
        return self.second(self.first(input_ids.float()))


def test_collect_fast_max_layers_limits_smoke_to_first_linear():
    model = TinyCausalModel().eval()
    visited = []

    def callback(name, module, stats):
        visited.append((name, module, stats.d))

    collect_and_encode_awq_style(
        model,
        tokenizer=None,
        calib=[torch.tensor([[1, 2]])],
        device="cpu",
        need_H=False,
        k=0,
        eps=1e-6,
        callback=callback,
        layer_batch_size=1,
        max_length=2,
        stats_device="cpu",
        max_layers=1,
    )

    assert [(name, stats_d) for name, _module, stats_d in visited] == [
        ("first", 2)
    ]


def test_collect_fast_rejects_non_positive_max_layers():
    with pytest.raises(ValueError, match="max_layers"):
        collect_and_encode_awq_style(
            TinyCausalModel().eval(),
            tokenizer=None,
            calib=[],
            device="cpu",
            need_H=False,
            k=0,
            eps=1e-6,
            callback=lambda *_args: None,
            max_layers=0,
        )
