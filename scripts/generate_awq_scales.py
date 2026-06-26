"""Generate AWQ per-channel scales for the modular baseline runner.

The output is a .pt file compatible with:

    AWQ_SCALES_PT=<out.pt> bash scripts/run_full_baselines.sh
    python -m scripts.run_quantization_baseline --grid awq --awq-scales-pt <out.pt> ...

For each Linear layer, this script collects:

* activation_scale = E[abs(x_j)] from calibration activations
* a uniform reservoir sample X_sample

It then runs the AWQ alpha grid search followed by the paper's per-output,
per-group weight-clipping search.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenflip.quantization.awq_scales import compute_awq_clip, compute_awq_scales
from eigenflip.statistics.collect_fast import _resolve_input_device, is_lm_head
from baseline_utils.runtime import build_model_slug, load_runtime_env

try:
    from baseline_utils.calibration import (
        get_c4_calibration_data,
        get_wikitext2_calibration_data,
    )
except ImportError:
    get_c4_calibration_data = get_wikitext2_calibration_data = None


class AWQScaleAccumulator:
    def __init__(self, in_features: int, sample_tokens: int):
        self.in_features = in_features
        self.sample_tokens = sample_tokens
        self.sum_abs = torch.zeros(in_features, dtype=torch.float64)
        self.n = 0
        self.sample_buffer: torch.Tensor | None = None
        self.sample_priorities: torch.Tensor | None = None
        self.sampled = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        xf = x.detach().reshape(-1, x.shape[-1]).float().cpu()
        self.sum_abs += xf.abs().sum(dim=0).double()
        self.n += xf.shape[0]

        priorities = torch.rand(xf.shape[0])
        if self.sample_buffer is not None:
            xf = torch.cat([self.sample_buffer, xf], dim=0)
            priorities = torch.cat([self.sample_priorities, priorities], dim=0)
        keep = min(self.sample_tokens, xf.shape[0])
        selected = torch.topk(priorities, keep, sorted=False).indices
        self.sample_buffer = xf.index_select(0, selected).clone()
        self.sample_priorities = priorities.index_select(0, selected).clone()
        self.sampled = keep

    def activation_scale(self) -> torch.Tensor:
        if self.n <= 0:
            raise RuntimeError("no calibration activations were collected")
        return (self.sum_abs / self.n).float()

    def x_sample(self) -> torch.Tensor:
        if self.sample_buffer is None:
            raise RuntimeError("no activation sample was collected")
        return self.sample_buffer.float()

    def free(self):
        self.sum_abs = None
        self.sample_buffer = None
        self.sample_priorities = None


class InvalidCalibrationTokenIds(ValueError):
    """Raised before CUDA when token IDs cannot index model embeddings."""


def validate_calibration_token_ids(
    input_ids: torch.Tensor,
    *,
    embedding_vocab_size: int,
    model_path: str,
) -> None:
    if input_ids.numel() == 0:
        raise InvalidCalibrationTokenIds("calibration input_ids is empty")

    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    if min_id < 0 or max_id >= embedding_vocab_size:
        raise InvalidCalibrationTokenIds(
            "calibration token IDs are incompatible with the model embeddings: "
            f"min_id={min_id}, max_id={max_id}, embedding_vocab_size="
            f"{embedding_vocab_size}, model={model_path!r}. This usually means a "
            "calibration cache created with a different tokenizer was reused. "
            "Regenerate calibration data with the current tokenizer."
        )


def load_calibration(tokenizer, args):
    if get_c4_calibration_data is None:
        raise RuntimeError("baseline_utils.calibration is not importable")
    if args.calib_dataset == "c4":
        return get_c4_calibration_data(
            tokenizer,
            n_samples=args.n_calib,
            seqlen=args.seqlen,
            seed=args.seed,
            return_tensors=True,
            cache_dir=args.cache_dir,
        )
    return get_wikitext2_calibration_data(
        tokenizer,
        n_samples=args.n_calib,
        seqlen=args.seqlen,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )


def default_output_path(args) -> str:
    slug = build_model_slug(args.model_path)
    return os.path.join(
        args.output_dir,
        f"{slug}_awq_scales_{args.scheme}_w{args.bits}g{args.group_size}_"
        f"{args.calib_dataset}n{args.n_calib}.pt",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AWQ scales for a HF causal LM.")
    parser.add_argument("--model-path", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--out", default=None)
    parser.add_argument("--output-dir", default="./outputs/awq_scales")
    parser.add_argument("--scheme", choices=["asymmetric", "symmetric"], default="asymmetric")
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 8])
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Weight group size. Use -1 for per-channel AWQ scale search.",
    )
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument(
        "--weight-clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run AWQ per-output-channel/per-group clipping search.",
    )
    parser.add_argument("--clip-grid", type=int, default=20)
    parser.add_argument("--clip-max-shrink", type=float, default=0.5)
    parser.add_argument("--layer-batch-size", type=int, default=4)
    parser.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./calibration_cache")
    parser.add_argument("--skip-lm-head", action="store_true", default=True)
    parser.add_argument("--device-map", default=os.getenv("MODEL_DEVICE_MAP", "auto"))
    parser.add_argument("--input-device", default=os.getenv("INPUT_DEVICE", "auto"))
    return parser.parse_args()


def effective_awq_group_size(requested_group_size: int, weights: torch.Tensor) -> int:
    """Return the layer-local AWQ group size, with -1 denoting per-channel."""

    if requested_group_size == -1:
        return int(weights.shape[1])
    if requested_group_size <= 0:
        raise ValueError(
            f"--group-size must be positive or -1 for per-channel, got {requested_group_size}"
        )
    return int(requested_group_size)


@torch.no_grad()
def main():
    args = parse_args()
    load_runtime_env()
    if args.sample_tokens <= 0:
        raise ValueError("--sample-tokens must be positive")
    if args.clip_grid <= 0:
        raise ValueError("--clip-grid must be positive")
    if not (0 < args.clip_max_shrink <= 1):
        raise ValueError("--clip-max-shrink must be in (0, 1]")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = None if args.device_map.lower() == "none" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    if device_map is None:
        if args.input_device == "auto":
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            target_device = args.input_device
        model.to(target_device)
    input_device = _resolve_input_device(model, args.input_device)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "num_embeddings"):
        raise RuntimeError("model does not expose an input embedding vocabulary size")
    embedding_vocab_size = int(input_embeddings.num_embeddings)

    calibration = load_calibration(tokenizer, args)
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not (args.skip_lm_head and is_lm_head(name))
    ]

    out_path = args.out or default_output_path(args)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AWQ scale generation")
    print("model:", args.model_path)
    print("layers:", len(layers))
    print("scheme:", args.scheme)
    print("bits/group:", args.bits, args.group_size)
    print("calibration:", args.calib_dataset, args.n_calib, args.seqlen)
    print("devices:", "device_map", args.device_map, "input_device", input_device)
    print("sample tokens per layer:", args.sample_tokens)
    print(
        "weight clipping:",
        args.weight_clipping,
        "grid",
        args.clip_grid,
        "max_shrink",
        args.clip_max_shrink,
    )
    print("out:", out_path)
    print("=" * 70)

    result = {}
    n_batches = (len(layers) + args.layer_batch_size - 1) // args.layer_batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * args.layer_batch_size
        end = min(start + args.layer_batch_size, len(layers))
        batch = layers[start:end]
        print(f"\n[batch {batch_idx + 1}/{n_batches}] layers {start}-{end - 1}")

        accs = {
            name: AWQScaleAccumulator(module.weight.shape[1], args.sample_tokens)
            for name, module in batch
        }

        def make_hook(layer_name: str):
            def hook(_module, inputs, _output):
                x = inputs[0] if isinstance(inputs, tuple) else inputs
                accs[layer_name].add(x)
            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in batch]

        for sample in tqdm(calibration, desc="  calib", leave=False):
            try:
                if torch.is_tensor(sample):
                    ids = sample
                    if ids.dim() == 1:
                        ids = ids.unsqueeze(0)
                    validate_calibration_token_ids(
                        ids,
                        embedding_vocab_size=embedding_vocab_size,
                        model_path=args.model_path,
                    )
                    ids = ids.to(input_device, non_blocking=True)
                    model(input_ids=ids, use_cache=False)
                    del ids
                else:
                    encoded = tokenizer(
                        sample,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.seqlen,
                    )
                    validate_calibration_token_ids(
                        encoded["input_ids"],
                        embedding_vocab_size=embedding_vocab_size,
                        model_path=args.model_path,
                    )
                    encoded = {
                        key: value.to(input_device, non_blocking=True)
                        for key, value in encoded.items()
                    }
                    model(**encoded, use_cache=False)
                    del encoded
            except InvalidCalibrationTokenIds:
                raise
            except Exception as exc:
                if "device-side assert" in str(exc).lower():
                    raise RuntimeError(
                        "CUDA device-side assert during calibration; aborting because "
                        "the CUDA context is no longer safe to reuse. Re-run with "
                        "CUDA_LAUNCH_BLOCKING=1 for the originating operation."
                    ) from exc
                print(f"  warning: skipped calibration sample due to {type(exc).__name__}: {exc}")

        for handle in handles:
            handle.remove()

        for name, module in tqdm(batch, desc="  awq scale search", leave=False):
            acc = accs[name]
            x_sample = acc.x_sample()
            layer_group_size = effective_awq_group_size(
                args.group_size,
                module.weight.data,
            )
            scales, alpha, error = compute_awq_scales(
                module.weight.data,
                acc.activation_scale(),
                x_sample,
                bits=args.bits,
                group_size=layer_group_size,
                n_grid=args.n_grid,
                scheme=args.scheme,
            )
            clip_max = None
            clip_skipped = any(
                token in name for token in ("q_", "k_", "query", "key", "Wqkv")
            )
            if args.weight_clipping and not clip_skipped:
                weight = module.weight.data
                scales_on_weight = scales.to(device=weight.device, dtype=weight.dtype)
                clip_max = compute_awq_clip(
                    weight * scales_on_weight.unsqueeze(0),
                    x_sample.to(device=weight.device, dtype=weight.dtype)
                    / scales_on_weight.unsqueeze(0),
                    bits=args.bits,
                    group_size=layer_group_size,
                    scheme=args.scheme,
                    n_grid=args.clip_grid,
                    max_shrink=args.clip_max_shrink,
                    sample_tokens=args.sample_tokens,
                )
            result[name] = {
                "scales": scales.detach().cpu(),
                "clip_max": None if clip_max is None else clip_max.detach().cpu(),
                "alpha": float(alpha),
                "error": float(error),
                "scheme": args.scheme,
                "bits": args.bits,
                "group_size": args.group_size,
                "effective_group_size": layer_group_size,
                "model_path": args.model_path,
                "format_version": 2,
                "activation_statistic": "mean_abs",
                "weight_clipping": bool(args.weight_clipping),
                "clip_skipped": bool(clip_skipped),
                "clip_grid": args.clip_grid,
                "clip_max_shrink": args.clip_max_shrink,
                "n_calib_tokens": int(acc.n),
                "sample_tokens": int(acc.sampled),
            }
            clipping = "skipped" if clip_max is None else "searched"
            print(
                f"  {name}: alpha={alpha:.4f} error={error:.6g} "
                f"clipping={clipping}"
            )
            acc.free()

        torch.save(result, out_path)
        print(f"  saved checkpoint scales -> {out_path}")
        accs.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAWQ scales saved:", out_path)


if __name__ == "__main__":
    main()
