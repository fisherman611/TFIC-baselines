"""Version-local attention dispatch for transform-aware quantization paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True)
class AttentionRuntime:
    family: str
    apply_rotary_pos_emb: Callable
    attention_interfaces: object
    eager_attention_forward: Callable

    def extra_attention_kwargs(self, module: nn.Module) -> dict:
        if self.family == "qwen2":
            return {"sliding_window": module.sliding_window}
        if self.family == "mistral":
            return {
                "sliding_window": getattr(module.config, "sliding_window", None)
            }
        return {}


def resolve_attention_runtime(module: nn.Module) -> AttentionRuntime:
    """Resolve supported attention behavior from the concrete module class."""

    from transformers.models.llama.modeling_llama import (
        ALL_ATTENTION_FUNCTIONS as LLAMA_INTERFACES,
        LlamaAttention,
        apply_rotary_pos_emb as llama_rope,
        eager_attention_forward as llama_eager,
    )
    from transformers.models.mistral.modeling_mistral import (
        ALL_ATTENTION_FUNCTIONS as MISTRAL_INTERFACES,
        MistralAttention,
        apply_rotary_pos_emb as mistral_rope,
        eager_attention_forward as mistral_eager,
    )
    from transformers.models.qwen2.modeling_qwen2 import (
        ALL_ATTENTION_FUNCTIONS as QWEN2_INTERFACES,
        Qwen2Attention,
        apply_rotary_pos_emb as qwen2_rope,
        eager_attention_forward as qwen2_eager,
    )

    if isinstance(module, LlamaAttention):
        return AttentionRuntime(
            family="llama",
            apply_rotary_pos_emb=llama_rope,
            attention_interfaces=LLAMA_INTERFACES,
            eager_attention_forward=llama_eager,
        )
    if isinstance(module, Qwen2Attention):
        return AttentionRuntime(
            family="qwen2",
            apply_rotary_pos_emb=qwen2_rope,
            attention_interfaces=QWEN2_INTERFACES,
            eager_attention_forward=qwen2_eager,
        )
    if isinstance(module, MistralAttention):
        return AttentionRuntime(
            family="mistral",
            apply_rotary_pos_emb=mistral_rope,
            attention_interfaces=MISTRAL_INTERFACES,
            eager_attention_forward=mistral_eager,
        )
    raise TypeError(
        "transform-aware cache quantization supports LlamaAttention, "
        f"Qwen2Attention, and MistralAttention; got {type(module).__name__}"
    )
