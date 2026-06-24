"""Calibrate official-style FlexRound quantizers on block calibration data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_utils.calibration import (
    get_c4_calibration_data,
    get_wikitext2_calibration_data,
)
from baseline_utils.runtime import build_model_slug, load_runtime_env
from baseline_utils.wandb import log_to_wandb, wandb_enabled_from_env
from eigenflip.statistics.collect_fast import _resolve_input_device
from assignment_methods.flexround import (
    FlexRoundCalibrationConfig,
    calibrate_flexround_block,
)
from scripts.calibrate_flatquant import capture_first_layer_inputs, model_identity


def load_calibration(tokenizer, args):
    loader = (
        get_c4_calibration_data
        if args.calib_dataset == "c4"
        else get_wikitext2_calibration_data
    )
    kwargs = dict(
        n_samples=args.n_calib,
        seqlen=args.seqlen,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    if args.calib_dataset == "c4":
        kwargs["return_tensors"] = True
    return loader(tokenizer, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Block-wise FlexRound calibration optimizer."
    )
    parser.add_argument("--model-path", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--out", default=None)
    parser.add_argument("--output-dir", default="./outputs/flexround")
    parser.add_argument("--weight-bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument(
        "--weight-symmetric",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./calibration_cache")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--device-map", default=os.getenv("MODEL_DEVICE_MAP", "auto"))
    parser.add_argument("--input-device", default=os.getenv("INPUT_DEVICE", "auto"))
    parser.add_argument("--train-device", default=os.getenv("TRAIN_DEVICE", "auto"))
    parser.add_argument("--wandb-project", default="tfic-baselines")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_runtime_env()
    torch.manual_seed(args.seed)
    train_device = (
        "cuda:0"
        if args.train_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.train_device == "auto"
        else args.train_device
    )

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
        model.to(train_device)
    input_device = _resolve_input_device(model, args.input_device)
    calibration = load_calibration(tokenizer, args)
    inputs, block_kwargs = capture_first_layer_inputs(
        model, tokenizer, calibration, input_device
    )

    config = FlexRoundCalibrationConfig(
        weight_bits=args.weight_bits,
        weight_symmetric=args.weight_symmetric,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    identity = model_identity(model, args.model_path)
    artifact = {
        "format": "tfic-flexround",
        "format_version": 1,
        "model": identity,
        "calibration": {
            "dataset": args.calib_dataset,
            "n_samples": len(inputs),
            "sequence_length": args.seqlen,
            "seed": args.seed,
        },
        "optimization": vars(config),
        "layers": {},
        "history": {},
    }
    slug = build_model_slug(args.model_path)
    out_path = Path(
        args.out
        or Path(args.output_dir)
        / f"{slug}_flexround_w{args.weight_bits}_{args.calib_dataset}n{len(inputs)}.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layers = list(model.model.layers)
    layer_count = len(layers) if args.max_layers is None else min(args.max_layers, len(layers))
    for index in tqdm(range(layer_count), desc="FlexRound blocks"):
        block = model.model.layers[index].cpu()
        local_artifact, inputs, history = calibrate_flexround_block(
            block,
            inputs,
            block_kwargs,
            config=config,
            device=train_device,
        )
        prefix = f"model.layers.{index}."
        artifact["layers"].update(
            {prefix + name: values for name, values in local_artifact.items()}
        )
        artifact["history"][str(index)] = history
        torch.save(artifact, out_path)

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "artifact": str(out_path),
                "model": identity,
                "calibration": artifact["calibration"],
                "optimization": artifact["optimization"],
                "final_loss_by_block": {
                    key: values[-1] for key, values in artifact["history"].items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if wandb_enabled_from_env(default=False):
        metrics = {
            f"flexround/block_{key}_final_mse": values[-1]
            for key, values in artifact["history"].items()
        }
        log_to_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.run_name or f"{slug}-flexround-calibration",
            metrics=metrics,
            config={**artifact["calibration"], **artifact["optimization"], **identity},
            tags=["flexround", "calibration"],
        )
    print("FlexRound artifact saved:", out_path)


if __name__ == "__main__":
    main()
