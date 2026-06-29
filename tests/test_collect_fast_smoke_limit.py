from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from eigenflip.statistics.collect_fast import collect_and_encode_awq_style
from grid_baselines.transformed_linear import ActivationQuantizedLinear


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 3, bias=False)
        self.second = nn.Linear(3, 2, bias=False)

    def forward(self, input_ids, use_cache=False):
        del use_cache
        return self.second(self.first(input_ids.float()))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)

    def forward(self, values):
        return self.second(self.first(values))


class TinyBlockedCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, input_ids, use_cache=False):
        del use_cache
        values = input_ids.float()
        for layer in self.model.layers:
            values = layer(values)
        return values


class TinyActivationQuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        source = nn.Linear(2, 2, bias=False)
        source.weight.data.copy_(torch.eye(2))
        self.first = ActivationQuantizedLinear(
            source,
            bits=2,
            symmetric=True,
        )

    def forward(self, input_ids, use_cache=False):
        del use_cache
        return self.first(input_ids.float())


class TinyBlockResetActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        quantized_source = nn.Linear(2, 2, bias=False)
        quantized_source.weight.data.copy_(torch.eye(2))
        first_block = nn.Module()
        first_block.proj = ActivationQuantizedLinear(
            quantized_source,
            bits=2,
            symmetric=True,
        )
        second_block = nn.Module()
        second_block.proj = nn.Linear(2, 2, bias=False)
        second_block.proj.weight.data.copy_(torch.eye(2))
        self.model.layers = nn.ModuleList([first_block, second_block])

    def forward(self, input_ids, use_cache=False):
        del use_cache
        values = input_ids.float()
        for layer in self.model.layers:
            values = layer.proj(values)
        return values


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


def test_collect_fast_paired_gptaq_stats_track_quantized_path_asymmetry():
    model = TinyCausalModel().eval()
    stats_by_name = {}

    with torch.no_grad():
        model.first.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        )
        model.second.weight.copy_(torch.eye(2, 3))

    def callback(name, module, stats):
        stats_by_name[name] = stats
        if name == "first":
            module.weight.data.mul_(0.5)

    collect_and_encode_awq_style(
        model,
        tokenizer=None,
        calib=[
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[2.0, -1.0]]),
        ],
        device="cpu",
        need_H=True,
        k=0,
        eps=1e-6,
        callback=callback,
        layer_batch_size=2,
        keep_sigma=True,
        max_length=2,
        stats_device="cpu",
        paired_full_precision=True,
    )

    assert list(stats_by_name) == ["first", "second"]
    assert stats_by_name["first"].backend == "paired_gram"
    assert torch.count_nonzero(stats_by_name["first"].delta_cross) == 0
    assert torch.count_nonzero(stats_by_name["second"].delta_cross) > 0


def test_collect_fast_qronus_resets_paired_inputs_at_block_boundaries():
    model = TinyBlockedCausalModel().eval()
    stats_by_name = {}
    original_weights = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    def callback(name, module, stats):
        stats_by_name[name] = stats
        module.weight.data.mul_(0.5)

    collect_and_encode_awq_style(
        model,
        tokenizer=None,
        calib=[torch.tensor([[0.75, -1.25]])],
        device="cpu",
        need_H=True,
        k=0,
        eps=1e-6,
        callback=callback,
        keep_sigma=True,
        stats_device="cpu",
        paired_full_precision=True,
        paired_cache_dtype=torch.float32,
        paired_reset_by_block=True,
    )

    assert torch.count_nonzero(
        stats_by_name["model.layers.0.first"].delta_cross
    ) == 0
    assert torch.count_nonzero(
        stats_by_name["model.layers.0.second"].delta_cross
    ) > 0
    assert torch.count_nonzero(
        stats_by_name["model.layers.1.first"].delta_cross
    ) == 0
    assert torch.count_nonzero(
        stats_by_name["model.layers.1.second"].delta_cross
    ) > 0
    for name, module in model.named_modules():
        if name in original_weights:
            assert torch.equal(module.weight, original_weights[name] * 0.5)


def test_collect_fast_qronus_reference_pass_disables_activation_quantization():
    model = TinyActivationQuantizedModel().eval()
    stats_by_name = {}

    collect_and_encode_awq_style(
        model,
        tokenizer=None,
        calib=[torch.tensor([[0.3, 0.8]])],
        device="cpu",
        need_H=True,
        k=0,
        eps=1e-6,
        callback=lambda name, _module, stats: stats_by_name.setdefault(
            name, stats
        ),
        keep_sigma=True,
        stats_device="cpu",
        paired_full_precision=True,
        paired_disable_reference_quantization=True,
    )

    assert torch.count_nonzero(stats_by_name["first"].delta_cross) > 0
    assert model.first.activation_bits == 2


def test_collect_fast_qronus_block_reset_disables_prior_block_quantizers():
    model = TinyBlockResetActivationModel().eval()
    stats_by_name = {}

    collect_and_encode_awq_style(
        model,
        tokenizer=None,
        calib=[torch.tensor([[0.3, 0.8]])],
        device="cpu",
        need_H=True,
        k=0,
        eps=1e-6,
        callback=lambda name, _module, stats: stats_by_name.setdefault(
            name, stats
        ),
        keep_sigma=True,
        stats_device="cpu",
        paired_full_precision=True,
        paired_cache_dtype=torch.float32,
        paired_reset_by_block=True,
        paired_disable_reference_quantization=True,
    )

    assert torch.count_nonzero(
        stats_by_name["model.layers.0.proj"].delta_cross
    ) > 0
    assert torch.count_nonzero(
        stats_by_name["model.layers.1.proj"].delta_cross
    ) == 0
    assert model.model.layers[0].proj.activation_bits == 2
