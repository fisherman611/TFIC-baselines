from __future__ import annotations

import pytest
import torch

from baseline_utils.calibration import _tokenizer_cache_key
from scripts.generate_awq_scales import (
    InvalidCalibrationTokenIds,
    validate_calibration_token_ids,
)


class _DummyTokenizer:
    def __init__(self, name_or_path: str, vocab_size: int):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size


def test_calibration_cache_key_is_tokenizer_specific():
    llama3 = _DummyTokenizer("meta-llama/Meta-Llama-3.1-8B", 128256)
    tinyllama = _DummyTokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 32000)

    assert _tokenizer_cache_key(llama3) != _tokenizer_cache_key(tinyllama)


def test_calibration_token_ids_accept_embedding_range():
    validate_calibration_token_ids(
        torch.tensor([[0, 1, 31999]]),
        embedding_vocab_size=32000,
        model_path="tinyllama",
    )


@pytest.mark.parametrize("input_ids", [torch.tensor([[-1]]), torch.tensor([[32000]])])
def test_calibration_token_ids_reject_embedding_overflow(input_ids):
    with pytest.raises(InvalidCalibrationTokenIds, match="embedding_vocab_size=32000"):
        validate_calibration_token_ids(
            input_ids,
            embedding_vocab_size=32000,
            model_path="tinyllama",
        )
