"""
eval_ppl.py
===========
Perplexity on WikiText-2 and C4 for a (quantized) causal LM, matching the
Table-1 setup in the paper: fixed-length non-overlapping windows over the test
split, NLL summed in the model's own tokenization, exp(mean NLL).

This is the standard GPTQ/AWQ-style harness so numbers are comparable to the
literature. Lower is better.

Usage:
  python eval_ppl.py --model-path ./quantized_models/ablate/c1c_gated_p2_w3g128
  python eval_ppl.py --model-path <dir> --datasets wikitext2 c4 --seqlen 2048

Outputs a JSON next to the checkpoint (ppl.json) and prints a one-line summary.
"""

from __future__ import annotations

import os
import json
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from eigenflip.statistics.collect_fast import _resolve_input_device
from runtime_utils import load_runtime_env
from wandb_utils import collect_ppl_wandb_metrics, log_to_wandb, wandb_enabled_from_env


# --------------------------------------------------------------------------
# Test-set token streams. We tokenize the whole split once and chunk it.
# --------------------------------------------------------------------------
def get_wikitext2_testenc(tokenizer):
    from datasets import load_dataset
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    return tokenizer(text, return_tensors="pt").input_ids


def get_c4_testenc(tokenizer, n_samples, seqlen, seed, cache_dir="./dataset_cache"):
    """
    C4 has no fixed test perplexity split convention; follow GPTQ/AWQ: draw
    n_samples random seqlen-length windows from the validation stream and
    concatenate. Deterministic under seed.

    Streamed: we iterate the validation shard lazily and stop as soon as we
    have n_samples windows, so we never download/materialize all 8 shards.

    Cached: the resulting (1, n_samples*seqlen) token tensor is saved as a .pt
    keyed by (tokenizer, n_samples, seqlen, seed). Subsequent runs with the same
    key skip streaming and tokenization entirely.
    """
    import hashlib
    from datasets import load_dataset
    import random

    # Cache key includes tokenizer identity so different models don't collide.
    tok_id = getattr(tokenizer, "name_or_path", "") or type(tokenizer).__name__
    vocab = getattr(tokenizer, "vocab_size", 0)
    key = f"{tok_id}|{vocab}|n{n_samples}|L{seqlen}|s{seed}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"c4_testenc_{h}.pt")

    if os.path.exists(cache_file):
        print(f"  [c4] loading cached windows: {cache_file}")
        return torch.load(cache_file)

    # Stream a single validation shard; lazy, no full download.
    val = load_dataset(
        "allenai/c4", "en",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        streaming=True,
    )

    rng = random.Random(seed)
    chunks = []
    # Skip a deterministic, seed-dependent number of leading docs so different
    # seeds see different windows without needing random access.
    skip = rng.randint(0, 2000)
    for idx, item in enumerate(tqdm(val, desc="c4 stream", unit="doc",
                                    leave=False)):
        if idx < skip:
            continue
        if len(chunks) >= n_samples:
            break
        enc = tokenizer(item["text"], return_tensors="pt").input_ids
        if enc.shape[1] >= seqlen:
            start = rng.randint(0, enc.shape[1] - seqlen)
            chunks.append(enc[:, start:start + seqlen])

    if not chunks:
        raise RuntimeError("C4: no sample reached seqlen; lower --seqlen.")
    enc = torch.cat(chunks, dim=1)
    torch.save(enc, cache_file)
    print(f"  [c4] cached windows -> {cache_file}")
    return enc


@torch.no_grad()
def perplexity(model, input_ids, seqlen, device):
    """Non-overlapping windows; mean token NLL -> exp."""
    n = input_ids.shape[1]
    n_chunks = n // seqlen
    if n_chunks == 0:
        raise RuntimeError(f"test stream ({n} tok) shorter than seqlen={seqlen}")
    nlls = []
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    for c in tqdm(range(n_chunks), desc="ppl", unit="win", leave=False):
        ids = input_ids[:, c * seqlen:(c + 1) * seqlen].to(device)
        out = model(ids, use_cache=False)
        logits = out.logits  # (1, seqlen, V)
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        nlls.append(loss.item())
        total_tokens += shift_labels.numel()
        if (c + 1) % 8 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_nll = sum(nlls) / total_tokens
    return float(torch.exp(torch.tensor(mean_nll)))


def main():
    load_runtime_env()
    p = argparse.ArgumentParser(description="WikiText-2 / C4 perplexity.")
    p.add_argument("--model-path", required=True,
                   help="HF dir (quantized checkpoint or base model).")
    p.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"],
                   choices=["wikitext2", "c4"])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--c4-samples", type=int, default=128,
                   help="number of seqlen windows drawn for C4.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=str, default="./dataset_cache",
                   help="dir for cached C4 token windows.")
    p.add_argument("--out", type=str, default=None,
                   help="JSON path; default <model-path>/ppl.json")
    p.add_argument("--use-wandb", action="store_true", default=wandb_enabled_from_env(False),
                   help="log perplexity metrics to Weights & Biases.")
    p.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    p.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "tfic-baselines"))
    p.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY"))
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--device-map", default=os.getenv("MODEL_DEVICE_MAP", "auto"))
    p.add_argument("--input-device", default=os.getenv("INPUT_DEVICE", "auto"))
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Perplexity eval | model:", args.model_path)
    print("datasets:", args.datasets, "| seqlen:", args.seqlen)
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device_map = None if args.device_map.lower() == "none" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device_map, trust_remote_code=True)
    model.eval()
    if device_map is None:
        if args.input_device == "auto":
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            target_device = args.input_device
        model.to(target_device)
    input_device = _resolve_input_device(model, args.input_device)
    print("devices:", "device_map", args.device_map, "input_device", input_device)

    results = {"model_path": args.model_path, "seqlen": args.seqlen, "ppl": {}}
    for ds in args.datasets:
        print(f"\n[{ds}] building test stream ...")
        if ds == "wikitext2":
            enc = get_wikitext2_testenc(tok)
        else:
            enc = get_c4_testenc(tok, args.c4_samples, args.seqlen, args.seed,
                                 cache_dir=args.cache_dir)
        ppl = perplexity(model, enc, args.seqlen, input_device)
        results["ppl"][ds] = ppl
        print(f"[{ds}] perplexity = {ppl:.4f}")

    out = args.out or os.path.join(args.model_path, "ppl.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    summary = " | ".join(f"{k}={v:.4f}" for k, v in results["ppl"].items())
    print(f"\nSUMMARY  {os.path.basename(args.model_path.rstrip('/'))}  {summary}")
    print(f"Wrote {out}")

    if args.use_wandb:
        run_name = args.wandb_run_name or f"ppl_{os.path.basename(args.model_path.rstrip('/'))}"
        log_to_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=run_name,
            metrics=collect_ppl_wandb_metrics(results),
            config={
                "model_path": args.model_path,
                "datasets": args.datasets,
                "seqlen": args.seqlen,
                "c4_samples": args.c4_samples,
                "seed": args.seed,
                "cache_dir": args.cache_dir,
                "out": out,
                "device_map": args.device_map,
                "input_device": str(input_device),
            },
            tags=["eval:ppl"],
            summary={"model_path": args.model_path, "ppl_json": out},
        )


if __name__ == "__main__":
    main()
