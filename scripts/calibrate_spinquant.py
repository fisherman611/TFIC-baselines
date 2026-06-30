"""Calibrate SpinQuant R1/R2 rotations and save an artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_utils.calibration import (
    get_c4_calibration_data,
    get_wikitext2_calibration_data,
)
from baseline_utils.runtime import build_model_slug, load_runtime_env
from eigenflip.statistics.collect_fast import _resolve_input_device
from grid_baselines.spinquant_calibration import (
    SpinQuantCalibrationConfig,
    capture_spinquant_layer_inputs,
    hadamard_spinquant_rotations,
    identity_spinquant_rotations,
    summarize_history,
    calibrate_spinquant_cross_entropy,
    calibrate_spinquant_layer_rotations,
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


def calibration_input_ids(tokenizer, calibration, args) -> list[torch.Tensor]:
    inputs = []
    for sample in calibration:
        if torch.is_tensor(sample):
            input_ids = sample.unsqueeze(0) if sample.dim() == 1 else sample
        else:
            encoded = tokenizer(
                sample,
                return_tensors="pt",
                truncation=True,
                max_length=args.seqlen,
            )
            input_ids = encoded["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        inputs.append(input_ids)
    if not inputs:
        raise RuntimeError("no SpinQuant calibration token batches were produced")
    return inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cayley-style SpinQuant R1/R2 calibration trainer."
    )
    parser.add_argument("--model-path", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--out", default=None)
    parser.add_argument("--output-dir", default="./outputs/spinquant")
    parser.add_argument("--weight-bits", type=int, default=3, choices=[3, 4, 8, 16])
    parser.add_argument("--weight-group-size", type=int, default=128)
    parser.add_argument(
        "--weight-scheme",
        choices=["asymmetric", "symmetric"],
        default="asymmetric",
    )
    parser.add_argument("--activation-bits", type=int, default=16, choices=[4, 8, 16])
    parser.add_argument(
        "--activation-symmetric",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--activation-group-size", type=int, default=-1)
    parser.add_argument("--activation-clip-ratio", type=float, default=1.0)
    parser.add_argument("--r1-steps", type=int, default=100)
    parser.add_argument("--r2-steps", type=int, default=100)
    parser.add_argument(
        "--rotation-init",
        choices=["hadamard", "random", "identity"],
        default="hadamard",
        help=(
            "Initial R1/R2 rotations before Cayley optimization. "
            "Hadamard uses random-signed Hadamard matrices and is the "
            "paper-aligned default; identity/random are ablations."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--objective",
        choices=["cross_entropy", "reconstruction"],
        default="cross_entropy",
        help=(
            "cross_entropy optimizes the full fake-quantized model loss; "
            "reconstruction uses the older local linear MSE surrogate."
        ),
    )
    parser.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./calibration_cache")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--device-map", default=os.getenv("MODEL_DEVICE_MAP", "auto"))
    parser.add_argument("--input-device", default=os.getenv("INPUT_DEVICE", "auto"))
    parser.add_argument("--train-device", default=os.getenv("TRAIN_DEVICE", "auto"))
    return parser.parse_args()


def main():
    args = parse_args()
    load_runtime_env()
    torch.manual_seed(args.seed)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.r1_steps < 0 or args.r2_steps < 0:
        raise ValueError("R1/R2 steps must be non-negative")
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
    calibration = load_calibration(tokenizer, args)

    layers = list(model.model.layers)
    layer_count = len(layers) if args.max_layers is None else min(args.max_layers, len(layers))
    head_dim = int(layers[0].self_attn.head_dim)
    rotation_kwargs = dict(
        num_layers=len(layers),
        hidden_size=int(model.config.hidden_size),
        head_dim=head_dim,
    )
    if args.rotation_init == "hadamard":
        rotations = hadamard_spinquant_rotations(**rotation_kwargs, seed=args.seed)
    elif args.rotation_init == "random":
        from grid_baselines.spinquant_quantization_grid import (
            random_spinquant_rotations,
        )

        rotations = random_spinquant_rotations(**rotation_kwargs, seed=args.seed)
    else:
        rotations = identity_spinquant_rotations(**rotation_kwargs)
    config = SpinQuantCalibrationConfig(
        weight_bits=args.weight_bits,
        weight_group_size=args.weight_group_size,
        weight_scheme=args.weight_scheme,
        activation_bits=args.activation_bits,
        activation_symmetric=args.activation_symmetric,
        activation_group_size=args.activation_group_size,
        activation_clip_ratio=args.activation_clip_ratio,
        r1_steps=args.r1_steps,
        r2_steps=args.r2_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        objective=args.objective,
    )
    slug = build_model_slug(args.model_path)
    out_path = Path(
        args.out
        or Path(args.output_dir)
        / f"{slug}_spinquant_R_{args.calib_dataset}n{args.n_calib}_s{args.seqlen}.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "format": "tfic-spinquant-rotations",
        "format_version": 1,
        "model": model_identity(model, args.model_path),
        "calibration": {
            "dataset": args.calib_dataset,
            "n_samples": len(calibration),
            "sequence_length": args.seqlen,
            "seed": args.seed,
        },
        "training": vars(config),
        "rotation_init": args.rotation_init,
        "R1": rotations.R1,
        "history": {},
    }
    for index in range(len(layers)):
        artifact[f"model.layers.{index}.self_attn.R2"] = rotations.R2[index]

    if args.objective == "cross_entropy":
        ce_inputs = calibration_input_ids(tokenizer, calibration, args)
        rotations, history = calibrate_spinquant_cross_entropy(
            model,
            ce_inputs,
            rotations,
            config=config,
            device=train_device,
        )
        artifact["R1"] = rotations.R1
        for index in range(len(layers)):
            artifact[f"model.layers.{index}.self_attn.R2"] = rotations.R2[index]
        artifact["history"]["cross_entropy"] = summarize_history(history)
        torch.save(artifact, out_path)
    else:
        input_device = _resolve_input_device(model, args.input_device)
        inputs, block_kwargs = capture_first_layer_inputs(
            model, tokenizer, calibration, input_device
        )
        artifact["calibration"]["n_samples"] = len(inputs)
        for index in range(layer_count):
            print(f"SpinQuant rotation calibration block {index + 1}/{layer_count}")
            layer = model.model.layers[index]
            captured, inputs = capture_spinquant_layer_inputs(
                layer,
                inputs,
                block_kwargs,
                device=train_device,
            )
            r1, r2, r1_history = calibrate_spinquant_layer_rotations(
                layer,
                captured,
                r1=rotations.R1,
                r2=rotations.R2[index],
                config=config,
                device=train_device,
                train_r1=args.r1_steps > 0,
                train_r2=False,
            )
            rotations.R1 = r1
            r1, r2, r2_history = calibrate_spinquant_layer_rotations(
                layer,
                captured,
                r1=rotations.R1,
                r2=r2,
                config=config,
                device=train_device,
                train_r1=False,
                train_r2=args.r2_steps > 0,
            )
            rotations.R1 = r1
            rotations.R2[index] = r2
            artifact["R1"] = rotations.R1
            artifact[f"model.layers.{index}.self_attn.R2"] = rotations.R2[index]
            artifact["history"][str(index)] = {
                "r1": summarize_history(r1_history),
                "r2": summarize_history(r2_history),
            }
            torch.save(artifact, out_path)
            print(
                "  R1 loss:",
                artifact["history"][str(index)]["r1"],
                "R2 loss:",
                artifact["history"][str(index)]["r2"],
            )

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "artifact": str(out_path),
                "model": artifact["model"],
                "calibration": artifact["calibration"],
                "training": artifact["training"],
                "history": artifact["history"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("SpinQuant rotations saved:", out_path)


if __name__ == "__main__":
    main()
