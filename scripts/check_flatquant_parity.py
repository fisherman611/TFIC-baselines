"""Check unquantized FlatQuant reparameterization parity on a real HF model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenflip.statistics.collect_fast import _resolve_input_device
from grid_baselines import (
    apply_flatquant_transforms,
    load_flatquant_transforms,
    validate_flatquant_artifact_identity,
)


def parity_metrics(reference: torch.Tensor, transformed: torch.Tensor) -> dict[str, float]:
    difference = (reference.float() - transformed.float()).abs()
    return {
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "top_token_agreement": float(
            (reference.argmax(-1) == transformed.argmax(-1)).float().mean().item()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--flatquant-transforms-pt", required=True)
    parser.add_argument(
        "--prompt",
        default="The purpose of quantization in neural networks is",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--input-device", default="auto")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--min-top-token-agreement", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


@torch.no_grad()
def _run_logits(model, inputs, input_device):
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    return model(**inputs, use_cache=False).logits.detach().cpu()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    device_map = None if args.device_map.lower() == "none" else args.device_map

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    if device_map is None:
        base.to("cuda:0" if torch.cuda.is_available() else "cpu")
    base_input_device = _resolve_input_device(base, args.input_device)
    reference = _run_logits(base, inputs, base_input_device)
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformed = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    if device_map is None:
        transformed.to("cuda:0" if torch.cuda.is_available() else "cpu")
    validate_flatquant_artifact_identity(
        args.flatquant_transforms_pt, transformed, require_identity=True
    )
    transforms, clips = load_flatquant_transforms(args.flatquant_transforms_pt)
    apply_flatquant_transforms(
        transformed,
        transforms,
        activation_bits=16,
        clips=clips,
    )
    transformed_input_device = _resolve_input_device(transformed, args.input_device)
    actual = _run_logits(transformed, inputs, transformed_input_device)
    metrics = parity_metrics(reference, actual)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if metrics["max_abs_error"] > args.atol:
        raise SystemExit(
            f"FlatQuant parity failed: max_abs_error={metrics['max_abs_error']:.6g} "
            f"> atol={args.atol:.6g}"
        )
    if metrics["top_token_agreement"] < args.min_top_token_agreement:
        raise SystemExit(
            "FlatQuant parity failed: "
            f"top_token_agreement={metrics['top_token_agreement']:.3f}"
        )


if __name__ == "__main__":
    main()
