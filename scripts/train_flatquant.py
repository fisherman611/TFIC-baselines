"""Optimize FlatQuant transforms on calibration data and save a model-bound artifact."""

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
from grid_baselines.flatquant_training import (
    FlatQuantTrainingConfig,
    train_flatquant_block,
)


class _CapturedFirstLayer(RuntimeError):
    pass


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


def model_identity(model, model_path: str) -> dict[str, int | str]:
    attention = model.model.layers[0].self_attn
    return {
        "model_path": model_path,
        "model_type": str(model.config.model_type),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(attention.head_dim),
    }


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


@torch.no_grad()
def capture_first_layer_inputs(model, tokenizer, calibration, input_device):
    first_layer = model.model.layers[0]
    inputs: list[torch.Tensor] = []
    kwargs_list: list[dict] = []

    def hook(_module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        inputs.append(hidden.detach().cpu())
        kwargs_list.append(
            {
                key: _cpu_tree(value)
                for key, value in kwargs.items()
                if key != "hidden_states"
            }
        )
        raise _CapturedFirstLayer

    handle = first_layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for sample in tqdm(calibration, desc="capture calibration"):
            try:
                if torch.is_tensor(sample):
                    input_ids = sample.unsqueeze(0) if sample.dim() == 1 else sample
                    model(input_ids=input_ids.to(input_device), use_cache=False)
                else:
                    encoded = tokenizer(
                        sample,
                        return_tensors="pt",
                        truncation=True,
                        max_length=model.config.max_position_embeddings,
                    )
                    encoded = {key: value.to(input_device) for key, value in encoded.items()}
                    model(**encoded, use_cache=False)
            except _CapturedFirstLayer:
                continue
    finally:
        handle.remove()
    if not inputs:
        raise RuntimeError("no FlatQuant calibration inputs were captured")
    return inputs, kwargs_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Block-wise FlatQuant calibration optimizer."
    )
    parser.add_argument("--model-path", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--out", default=None)
    parser.add_argument("--output-dir", default="./outputs/flatquant")
    parser.add_argument("--weight-bits", type=int, default=3, choices=[3, 4, 8, 16])
    parser.add_argument("--activation-bits", type=int, default=16, choices=[4, 8, 16])
    parser.add_argument(
        "--weight-symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--activation-symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--weight-group-size", type=int, default=128)
    parser.add_argument("--activation-group-size", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument(
        "--add-diagonal", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--learn-weight-clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--learn-activation-clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--wandb-project", default="tfic-baselines")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_runtime_env()
    torch.manual_seed(args.seed)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
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

    config = FlatQuantTrainingConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        weight_symmetric=args.weight_symmetric,
        activation_symmetric=args.activation_symmetric,
        weight_group_size=args.weight_group_size,
        activation_group_size=args.activation_group_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        add_diagonal=args.add_diagonal,
        learn_weight_clipping=args.learn_weight_clipping,
        learn_activation_clipping=args.learn_activation_clipping,
    )
    identity = model_identity(model, args.model_path)
    layers = list(model.model.layers)
    layer_count = len(layers) if args.max_layers is None else min(args.max_layers, len(layers))
    artifact = {
        "format": "tfic-flatquant",
        "format_version": 1,
        "model": identity,
        "calibration": {
            "dataset": args.calib_dataset,
            "n_samples": len(inputs),
            "sequence_length": args.seqlen,
            "seed": args.seed,
        },
        "training": vars(config),
        "layers": {},
        "attention": {},
        "attention_clips": {},
        "history": {},
    }
    slug = build_model_slug(args.model_path)
    out_path = Path(
        args.out
        or Path(args.output_dir)
        / f"{slug}_flatquant_w{args.weight_bits}a{args.activation_bits}_{args.calib_dataset}n{len(inputs)}.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for index in range(layer_count):
        print(f"FlatQuant calibration block {index + 1}/{layer_count}")
        block = model.model.layers[index].cpu()
        local_artifact, inputs, history = train_flatquant_block(
            block,
            inputs,
            block_kwargs,
            config=config,
            device=train_device,
        )
        prefix = f"model.layers.{index}."
        if "self_attn.kcache_trans" in local_artifact:
            artifact["attention"][prefix + "self_attn"] = local_artifact.pop("self_attn.kcache_trans")["matrix"]

        layer_clips = {}
        for proj in ["q_proj", "k_proj", "v_proj"]:
            name = f"self_attn.{proj}"
            if name in local_artifact:
                clips = {}
                if "act_quantizer.clip_factor_a_max" in local_artifact[name]:
                    clips["clip_factor_a_max"] = local_artifact[name].pop("act_quantizer.clip_factor_a_max")
                if "act_quantizer.clip_factor_a_min" in local_artifact[name]:
                    clips["clip_factor_a_min"] = local_artifact[name].pop("act_quantizer.clip_factor_a_min")
                if clips:
                    layer_clips[proj] = clips
        if layer_clips:
            artifact["attention_clips"][prefix + "self_attn"] = layer_clips

        artifact["layers"].update(
            {prefix + name: values for name, values in local_artifact.items()}
        )
        artifact["history"][str(index)] = history
        torch.save(artifact, out_path)
        print(f"  loss: {history[0]:.6g} -> {history[-1]:.6g}")

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "artifact": str(out_path),
                "model": identity,
                "calibration": artifact["calibration"],
                "training": artifact["training"],
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
            f"flatquant/block_{key}_final_mse": values[-1]
            for key, values in artifact["history"].items()
        }
        log_to_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.run_name or f"{slug}-flatquant-calibration",
            metrics=metrics,
            config={**artifact["calibration"], **artifact["training"], **identity},
            tags=["flatquant", "calibration"],
        )
    print("FlatQuant artifact saved:", out_path)


if __name__ == "__main__":
    main()
