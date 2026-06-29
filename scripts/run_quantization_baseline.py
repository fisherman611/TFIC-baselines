"""Run grid-baseline x assignment-method quantization on a real HF model.

This is the experiment entrypoint for the modular baseline folders:

* ``grid_baselines`` builds the quantization grid.
* ``assignment_methods`` assigns integer codes on that grid.

The script saves a Hugging Face checkpoint under:

    <output-dir>/<grid>_<scheme>_<assignment>

Evaluate the saved checkpoint with ``scripts.eval_ppl`` or
``scripts.lm_eval_runner``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from assignment_methods import (
    FlexRoundAssignment,
    GPTAQAssignment,
    GPTAQResCompAssignment,
    GPTQAssignment,
    QronusAssignment,
    RTNAssignment,
    TFICAssignment,
)
from eigenflip.quantization.awq_scales import (
    layer_params_from_awq_run,
    scales_from_awq_run,
)
from eigenflip.statistics.collect_fast import collect_and_encode_awq_style
from grid_baselines import (
    apply_flatquant_attention_transforms,
    add_spinquant_k_cache_quantization,
    add_spinquant_activation_quantization,
    apply_flatquant_transforms,
    validate_flatquant_artifact_identity,
    apply_spinquant_no_had,
    build_awq_quantization_grid,
    build_flatquant_diag_quantization_grid,
    build_neuqi_quantization_grid,
    build_spinquant_quantization_grid,
    build_vanilla_quantization_grid,
    load_flatquant_transforms,
    load_flatquant_attention_clips,
    load_flatquant_attention_transforms,
    load_spinquant_rotations,
    random_spinquant_rotations,
    serialize_flatquant_transforms,
)
from grid_baselines.spinquant_quantization_grid import (
    apply_spinquant_r4,
    load_spinquant_r4,
)
from baseline_utils.model_loading import TRANSFORM_MANIFEST

try:
    from baseline_utils.calibration import (
        get_c4_calibration_data,
        get_wikitext2_calibration_data,
    )
except ImportError:
    get_c4_calibration_data = get_wikitext2_calibration_data = None


NEED_H = {
    "flexround": True,
    "rtn": False,
    "gptaq": True,
    "gptaq_rescomp": True,
    "gptq": True,
    "qronus": True,
    "tfic": True,
}
KEEP_SIGMA = {"gptaq", "gptaq_rescomp", "gptq", "qronus", "tfic"}
PAIRED_ASSIGNMENTS = {"gptaq", "gptaq_rescomp", "qronus"}


def assignment_needs_h(name: str, k: int) -> bool:
    """Return whether collection needs a full Gram matrix for this run."""

    if name == "flexround" and k == 0:
        return False
    return NEED_H[name]


def build_assignment(name: str, args):
    if name == "rtn":
        return RTNAssignment()
    if name == "gptq":
        return GPTQAssignment(damp=args.gptq_damp, order=args.gptq_order)
    if name == "gptaq":
        return GPTAQAssignment(
            damp=args.gptaq_damp,
            block_size=args.gptaq_block_size,
            alpha=args.gptaq_alpha,
            act_order=args.gptaq_act_order,
        )
    if name == "gptaq_rescomp":
        return GPTAQResCompAssignment(
            damp=args.gptaq_damp,
            block_size=args.gptaq_block_size,
            alpha=args.gptaq_alpha,
            rescomp_alpha=args.rescomp_alpha,
            rescomp_mode=args.rescomp_mode,
            act_order=args.gptaq_act_order,
        )
    if name == "qronus":
        qronus_alpha = args.qronus_alpha
        if args.qronus_damp is not None:
            qronus_alpha = args.qronus_damp
        if qronus_alpha is None:
            has_activation_quantization = (
                args.activation_bits < 16
                or args.q_bits < 16
                or args.k_bits < 16
                or args.v_bits < 16
            )
            qronus_alpha = 1e-3 if has_activation_quantization else 1e-6
        return QronusAssignment(
            alpha=qronus_alpha,
            act_order=args.qronus_act_order,
        )
    if name == "flexround":
        return FlexRoundAssignment(
            steps=args.flexround_steps,
            lr=args.flexround_lr,
            log_divisor_bound=args.flexround_log_divisor_bound,
            learn_layer_scale=args.flexround_learn_layer_scale,
            learn_row_scale=args.flexround_row_scale,
        )
    if name == "tfic":
        return TFICAssignment(
            alpha=args.tfic_alpha,
            beta=args.tfic_beta,
            eta=args.tfic_eta,
            gamma_th=args.tfic_gamma,
            kappa=args.tfic_kappa,
            gmax=args.tfic_gmax,
            n_stages=args.tfic_stages,
            sweeps=args.tfic_sweeps,
            c_cand=args.tfic_ccand,
            top_m=args.tfic_topm,
            chunk_cols=args.tfic_chunk,
        )
    raise ValueError(f"unknown assignment method: {name}")


def load_awq_scales(path: str | None) -> dict[str, torch.Tensor]:
    if not path:
        raise ValueError("--grid awq requires --awq-scales-pt")

    raw = torch.load(path, map_location="cpu")
    return scales_from_awq_run(raw)


def load_awq_layer_params(path: str | None, args=None) -> dict[str, dict]:
    if not path:
        raise ValueError("--grid awq requires --awq-scales-pt")

    params = layer_params_from_awq_run(torch.load(path, map_location="cpu"))
    if args is None:
        return params

    for layer_name, info in params.items():
        expected_fields = [
            ("bits", args.bits),
            ("group_size", args.group_size),
            ("scheme", args.scheme),
        ]
        for field, expected in expected_fields:
            actual = info.get(field)
            if (
                field == "group_size"
                and getattr(args, "assignment", None) == "qronus"
                and actual is None
            ):
                raise ValueError(
                    f"AWQ artifact for {layer_name!r} must record "
                    "group_size=-1 for Qronus"
                )
            if actual is not None and actual != expected:
                raise ValueError(
                    f"AWQ artifact mismatch for {layer_name!r}: "
                    f"{field}={actual!r}, requested {expected!r}"
                )
        artifact_model = info.get("model_path")
        if artifact_model is not None and artifact_model != args.model_path:
            raise ValueError(
                f"AWQ artifact for {layer_name!r} was generated from "
                f"{artifact_model!r}, requested model is {args.model_path!r}"
            )
    return params


def effective_weight_group_size(args, weights: torch.Tensor) -> int:
    """Return the layer-local weight group size for the requested assignment."""

    if getattr(args, "assignment", None) == "qronus":
        return int(weights.shape[1])
    return int(args.group_size)


def apply_qronus_paper_preset(args):
    """Apply and validate the Qronus preset experiment settings."""

    if getattr(args, "qronus_paper_preset", False):
        if args.assignment != "qronus":
            raise ValueError("--qronus-paper-preset requires --assignment qronus")
        args.calib_dataset = "c4"
        args.n_calib = 128
        args.seqlen = 2048
        args.group_size = -1
        args.qronus_act_order = True
    if args.assignment == "qronus" and args.group_size != -1:
        raise ValueError(
            "Qronus requires --group-size -1 for the paper's per-output-channel grid"
        )
    return args


def load_flatquant_diag_params(path: str | None) -> dict[str, dict[str, torch.Tensor]]:
    if not path:
        raise ValueError("--grid flatquant_diag requires --flatquant-params-pt")

    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "layers" in raw and isinstance(raw["layers"], dict):
        raw = raw["layers"]
    if not isinstance(raw, dict):
        raise ValueError("--flatquant-params-pt must contain a dict")

    scale_keys = (
        "scales",
        "scale",
        "flatquant_scales",
        "channel_scales",
        "channel_scale",
        "c",
    )
    clip_keys = ("weight_clip", "alpha_w", "clip", "clip_ratio")
    affine_keys = ("p", "P", "p1", "p2", "P1", "P2", "u", "v", "sigma")
    parsed: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name, value in raw.items():
        if torch.is_tensor(value):
            parsed[layer_name] = {"scales": torch.as_tensor(value)}
            continue
        if not isinstance(value, dict):
            raise ValueError(
                "FlatQuant diag layer params must be a tensor scale vector or a dict, "
                f"got {type(value)!r} for {layer_name!r}"
            )

        layer: dict[str, torch.Tensor] = {}
        for key in scale_keys:
            if key in value:
                layer["scales"] = torch.as_tensor(value[key])
                break
        for key in clip_keys:
            if key in value:
                layer["weight_clip"] = torch.as_tensor(value[key])
                break

        if "scales" not in layer:
            if any(key in value for key in affine_keys):
                raise ValueError(
                    "Full FlatQuant affine/Kronecker transforms are not supported "
                    "by this fixed-grid runner because they require online "
                    f"activation transforms. Use grid=flatquant_diag only with "
                    f"per-channel scales; missing scale for {layer_name!r}."
                )
            raise ValueError(f"missing FlatQuant diag scale vector for {layer_name!r}")
        parsed[layer_name] = layer
    return parsed


def build_grid(
    name: str,
    weights: torch.Tensor,
    args,
    stats,
    awq_scales: torch.Tensor | None,
    flatquant_diag_params: dict[str, torch.Tensor] | None,
    awq_clip_max: torch.Tensor | None = None,
):
    group_size = effective_weight_group_size(args, weights)
    if name == "vanilla":
        return build_vanilla_quantization_grid(
            weights,
            bits=args.bits,
            group_size=group_size,
            scheme=args.scheme,
        )
    if name == "awq":
        if awq_scales is None:
            raise ValueError("AWQ grid requires per-layer AWQ scales")
        return build_awq_quantization_grid(
            weights,
            awq_scales,
            bits=args.bits,
            group_size=group_size,
            scheme=args.scheme,
            clip_max=awq_clip_max,
        )
    if name in {"flatquant", "flatquant_diag"}:
        if flatquant_diag_params is None:
            raise ValueError("FlatQuant grid requires per-layer params")
        return build_flatquant_diag_quantization_grid(
            weights,
            flatquant_diag_params.get(
                "scales",
                torch.ones(weights.shape[1], device=weights.device),
            ),
            bits=args.bits,
            group_size=group_size,
            scheme=args.scheme,
            weight_clip=flatquant_diag_params.get("weight_clip", 1.0),
            weight_clip_max=flatquant_diag_params.get("weight_clip_max"),
            weight_clip_min=flatquant_diag_params.get("weight_clip_min"),
        )
    if name == "neuqi":
        if stats is None:
            raise ValueError("NeUQI grid requires per-layer activation stats")
        return build_neuqi_quantization_grid(
            weights,
            stats,
            bits=args.bits,
            group_size=group_size,
            scheme=args.scheme,
            scale_candidates=args.neuqi_scale_candidates,
            coarse_candidates=args.neuqi_coarse_candidates,
            row_chunk_size=args.neuqi_row_chunk_size,
            candidate_chunk_size=args.neuqi_candidate_chunk_size,
        )
    if name in {'spinquant', 'spinquant_had'}:
        return build_spinquant_quantization_grid(
            weights,
            bits=args.bits,
            group_size=group_size,
            scheme=args.scheme,
        )
    raise ValueError(f"unknown grid baseline: {name}")


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


def output_path(args) -> str:
    run_name = args.run_name or f"{args.grid}_{args.scheme}_{args.assignment}"
    return os.path.join(args.output_dir, run_name)


def transform_model_metadata(model) -> dict[str, int | str]:
    attention = model.model.layers[0].self_attn
    return {
        "model_type": str(model.config.model_type),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(attention.head_dim),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a real HF model using grid_baselines + assignment_methods."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="./quantized_models/baselines")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run quantization without writing a checkpoint; intended for smoke tests.",
    )

    parser.add_argument(
        '--grid',
        choices=[
            'vanilla',
            'awq',
            'flatquant',
            'flatquant_diag',
            'spinquant',
            'spinquant_had',
            'neuqi',
        ],
        required=True,
    )
    parser.add_argument("--assignment", choices=sorted(NEED_H), required=True)
    parser.add_argument("--scheme", choices=["asymmetric", "symmetric"], default="asymmetric")
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 8])
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Weight group size. For assignment=qronus, -1 denotes per-channel.",
    )
    parser.add_argument("--awq-scales-pt", default=None)
    parser.add_argument(
        "--flatquant-params-pt",
        default=None,
        help=(
            "Path to per-layer FlatQuant diagonal-scale grid params. Expected formats: "
            "{layer: scale_tensor} or {layer: {'scales': tensor, "
            "'weight_clip': scalar}}."
        ),
    )
    parser.add_argument(
        "--flatquant-transforms-pt",
        default=None,
        help=(
            "Normalized full FlatQuant per-linear Kronecker transforms. "
            "Required by --grid flatquant."
        ),
    )
    parser.add_argument(
        "--allow-unidentified-flatquant-artifact",
        action="store_true",
        help=(
            "Allow legacy normalized FlatQuant artifacts without model identity. "
            "Official numeric-key flat_matrices.pth files remain accepted."
        ),
    )
    parser.add_argument("--activation-bits", type=int, default=16, choices=[4, 8, 16])
    parser.add_argument(
        "--activation-symmetric",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--activation-group-size", type=int, default=-1)
    parser.add_argument("--activation-clip-ratio", type=float, default=1.0)

    parser.add_argument(
        '--spinquant-rotations-pt',
        default=None,
        help=(
            'Official learned SpinQuant checkpoint containing R1 and '
            'model.layers.{i}.self_attn.R2.'
        ),
    )
    parser.add_argument(
        '--spinquant-work-dtype',
        choices=['float32', 'float64'],
        default='float64',
        help='Compute dtype used while absorbing SpinQuant rotations.',
    )
    parser.add_argument(
        '--spinquant-r4-pt',
        default=None,
        help=(
            'Factorized R4 artifact with had_K and K from official get_hadK. '
            'Required when intermediate_size is not a power of two.'
        ),
    )
    parser.add_argument('--v-bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument(
        '--v-symmetric',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--v-clip-ratio', type=float, default=1.0)
    parser.add_argument('--k-bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument(
        '--k-symmetric',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--k-group-size', type=int, default=-1)
    parser.add_argument('--k-clip-ratio', type=float, default=1.0)
    parser.add_argument('--q-bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument(
        '--q-symmetric',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--q-group-size', type=int, default=-1)
    parser.add_argument('--q-clip-ratio', type=float, default=1.0)
    parser.add_argument(
        '--spinquant-random-rotations',
        action='store_true',
        help=(
            'Generate random orthogonal R1/R2 rotations instead of loading a '
            'learned SpinQuant checkpoint. Intended for smoke/debug runs.'
        ),
    )
    parser.add_argument(
        '--spinquant-random-seed',
        type=int,
        default=None,
        help='Seed for --spinquant-random-rotations. Defaults to --seed.',
    )
    parser.add_argument(
        "--neuqi-scale-candidates",
        type=int,
        default=2048,
        help="Number of fine scale candidates T for NeUQI.",
    )
    parser.add_argument(
        "--neuqi-coarse-candidates",
        type=int,
        default=64,
        help="Number of coarse scale candidates T_c for NeUQI.",
    )
    parser.add_argument(
        "--neuqi-candidate-chunk-size",
        type=int,
        default=16,
        help="Scale candidates evaluated per chunk for NeUQI.",
    )
    parser.add_argument(
        "--neuqi-row-chunk-size",
        type=int,
        default=16,
        help="Rows processed per chunk while searching NeUQI parameters.",
    )

    parser.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--cache-dir", default="./calibration_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--layer-batch-size", type=int, default=4)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Quantize only the first N linear layers; intended for smoke tests.",
    )
    parser.add_argument("--eig-on-cpu", action="store_true")
    parser.add_argument(
        "--device-map",
        default=os.getenv("MODEL_DEVICE_MAP", "auto"),
        help="Transformers device_map. Use auto/balanced/balanced_low_0/sequential or none.",
    )
    parser.add_argument(
        "--input-device",
        default=os.getenv("INPUT_DEVICE", "auto"),
        help="Device for input_ids during calibration. Default auto uses the first model device.",
    )
    parser.add_argument(
        "--stats-device",
        default=os.getenv("STATS_DEVICE", "layer"),
        help="Where streaming stats live: layer, input, cpu, cuda:0, cuda:1, ...",
    )

    parser.add_argument("--gptq-damp", type=float, default=0.01)
    parser.add_argument("--gptq-order", choices=["diag", "natural"], default="diag")

    parser.add_argument("--gptaq-damp", type=float, default=0.01)
    parser.add_argument("--gptaq-block-size", type=int, default=128)
    parser.add_argument("--gptaq-alpha", type=float, default=0.25)
    parser.add_argument("--gptaq-act-order", action="store_true")
    parser.add_argument("--rescomp-alpha", type=float, default=0.25)
    parser.add_argument(
        "--rescomp-mode", choices=["auto", "org", "allw"], default="auto"
    )
    parser.add_argument(
        "--gptaq-cache-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="CPU dtype for temporary full-precision activation cache.",
    )
    parser.add_argument(
        "--qronus-alpha",
        type=float,
        default=None,
        help=(
            "Qronus spectral damping coefficient. Defaults to 1e-6 for "
            "weight-only runs and 1e-3 when activation/K/V/Q quantization is enabled."
        ),
    )
    parser.add_argument(
        "--qronus-damp",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qronus-act-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process Qronus columns in descending Hessian diagonal order.",
    )
    parser.add_argument(
        "--qronus-cache-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="CPU dtype for Qronus reference activations.",
    )
    parser.add_argument(
        "--qronus-paper-preset",
        action="store_true",
        help=(
            "Use the Qronus preset: C4, 128x2048 calibration, "
            "per-output-channel grid, and descending Hessian order."
        ),
    )

    parser.add_argument("--flexround-steps", type=int, default=5000)
    parser.add_argument("--flexround-lr", type=float, default=3e-3)
    parser.add_argument(
        "--flexround-log-divisor-bound",
        type=float,
        default=float("inf"),
    )
    parser.add_argument(
        "--flexround-learn-layer-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn delta1 in FlexRound (updates dequantization scale).",
    )
    parser.add_argument(
        "--flexround-row-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn the additional output-channel factor from FlexRound.",
    )

    parser.add_argument("--tfic-alpha", type=float, default=1.0)
    parser.add_argument("--tfic-beta", type=float, default=1.0)
    parser.add_argument("--tfic-eta", type=float, default=1.0)
    parser.add_argument("--tfic-gamma", type=float, default=0.5)
    parser.add_argument("--tfic-kappa", type=float, default=2.0)
    parser.add_argument("--tfic-gmax", type=int, default=6)
    parser.add_argument("--tfic-stages", type=int, default=2)
    parser.add_argument("--tfic-sweeps", type=int, default=3)
    parser.add_argument("--tfic-ccand", type=float, default=8.0)
    parser.add_argument("--tfic-topm", type=int, default=32)
    parser.add_argument("--tfic-chunk", type=int, default=256)
    return parser.parse_args()


def main():
    args = apply_qronus_paper_preset(parse_args())
    torch.manual_seed(args.seed)

    input_device = args.input_device
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
        if input_device == "auto":
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            target_device = input_device
        model.to(target_device)

    flatquant_transforms = {}
    flatquant_clips = {}
    flatquant_attention_transforms = {}
    flatquant_attention_clips = {}
    if args.grid == "flatquant":
        if not args.flatquant_transforms_pt:
            raise ValueError(
                "--grid flatquant requires --flatquant-transforms-pt"
            )
        validate_flatquant_artifact_identity(
            args.flatquant_transforms_pt,
            model,
            require_identity=not args.allow_unidentified_flatquant_artifact,
            requested_quantization={
                "weight_bits": args.bits,
                "activation_bits": args.activation_bits,
                "weight_symmetric": args.scheme == "symmetric",
                "activation_symmetric": args.activation_symmetric,
                "weight_group_size": args.group_size,
                "activation_group_size": args.activation_group_size,
            },
        )
        flatquant_transforms, flatquant_clips = load_flatquant_transforms(
            args.flatquant_transforms_pt
        )
        flatquant_attention_transforms = load_flatquant_attention_transforms(
            args.flatquant_transforms_pt
        )
        flatquant_attention_clips = load_flatquant_attention_clips(
            args.flatquant_transforms_pt
        )
        apply_flatquant_transforms(
            model,
            flatquant_transforms,
            activation_bits=args.activation_bits,
            activation_symmetric=args.activation_symmetric,
            activation_group_size=args.activation_group_size,
            activation_clip_ratio=args.activation_clip_ratio,
            clips=flatquant_clips,
        )
        apply_flatquant_attention_transforms(
            model,
            flatquant_attention_transforms,
            q_bits=args.q_bits,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            q_symmetric=args.q_symmetric,
            k_symmetric=args.k_symmetric,
            v_symmetric=args.v_symmetric,
            q_group_size=args.q_group_size,
            k_group_size=args.k_group_size,
            q_clip_ratio=args.q_clip_ratio,
            k_clip_ratio=args.k_clip_ratio,
            v_clip_ratio=args.v_clip_ratio,
            clips=flatquant_attention_clips,
        )

    spinquant_r4 = None
    if args.grid in {'spinquant', 'spinquant_had'}:
        if args.spinquant_rotations_pt and args.spinquant_random_rotations:
            raise ValueError(
                'use either --spinquant-rotations-pt or '
                '--spinquant-random-rotations, not both'
            )
        if not args.spinquant_rotations_pt and not args.spinquant_random_rotations:
            raise ValueError(
                f'--grid {args.grid} requires --spinquant-rotations-pt or '
                '--spinquant-random-rotations'
            )
        head_dim = int(model.model.layers[0].self_attn.head_dim)
        if args.spinquant_random_rotations:
            rotations = random_spinquant_rotations(
                num_layers=model.config.num_hidden_layers,
                hidden_size=model.config.hidden_size,
                head_dim=head_dim,
                seed=args.seed
                if args.spinquant_random_seed is None
                else args.spinquant_random_seed,
            )
        else:
            rotations = load_spinquant_rotations(
                args.spinquant_rotations_pt,
                num_layers=model.config.num_hidden_layers,
                hidden_size=model.config.hidden_size,
                head_dim=head_dim,
            )
        work_dtype = {
            'float32': torch.float32,
            'float64': torch.float64,
        }[args.spinquant_work_dtype]
        apply_spinquant_no_had(
            model,
            rotations,
            work_dtype=work_dtype,
        )
        if args.grid == 'spinquant_had':
            if args.spinquant_r4_pt:
                spinquant_r4 = load_spinquant_r4(
                    args.spinquant_r4_pt,
                    width=model.config.intermediate_size,
                )
            spinquant_r4 = apply_spinquant_r4(model, spinquant_r4)
        elif args.spinquant_r4_pt:
            raise ValueError(
                '--spinquant-r4-pt is only valid with --grid spinquant_had'
            )
        if args.grid == 'spinquant' and args.k_bits < 16:
            raise ValueError(
                'K-cache quantization requires --grid spinquant_had because '
                'the post-RoPE R3 transform is online'
            )
        add_spinquant_activation_quantization(
            model,
            bits=args.activation_bits,
            symmetric=args.activation_symmetric,
            group_size=args.activation_group_size,
            clip_ratio=args.activation_clip_ratio,
            v_bits=args.v_bits,
            v_symmetric=args.v_symmetric,
            v_clip_ratio=args.v_clip_ratio,
        )
        if args.grid == 'spinquant_had':
            add_spinquant_k_cache_quantization(
                model,
                bits=args.k_bits,
                symmetric=args.k_symmetric,
                group_size=args.k_group_size,
                clip_ratio=args.k_clip_ratio,
            )

    calibration = load_calibration(tokenizer, args)
    awq_params_by_layer = (
        load_awq_layer_params(args.awq_scales_pt, args)
        if args.grid == "awq"
        else {}
    )
    flatquant_diag_by_layer = (
        load_flatquant_diag_params(args.flatquant_params_pt)
        if args.grid == "flatquant_diag"
        else (
            {
                name: dict(flatquant_clips.get(name, {}))
                for name in flatquant_transforms
            }
            if args.grid == "flatquant"
            else {}
        )
    )
    assignment = build_assignment(args.assignment, args)

    need_h = assignment_needs_h(args.assignment, args.k)
    # FlexRound can optimize the diagonal-plus-mean surrogate when k=0.
    # This avoids materializing per-layer d x d Gram matrices and provides a
    # practical one-pass smoke path. Full benchmark runs should keep k > 0.
    keep_sigma = args.assignment in KEEP_SIGMA
    cache_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    paired_cache_dtype = cache_dtypes[
        args.qronus_cache_dtype
        if args.assignment == "qronus"
        else args.gptaq_cache_dtype
    ]

    def callback(layer_name, module, stats):
        weights = module.weight.data
        layer_awq_scales = None
        layer_awq_clip_max = None
        if args.grid == "awq":
            layer_awq_params = awq_params_by_layer.get(layer_name)
            if layer_awq_params is None:
                raise KeyError(f"no AWQ scales for layer {layer_name!r}")
            layer_awq_scales = layer_awq_params["scales"]
            layer_awq_clip_max = layer_awq_params.get("clip_max")
        layer_flatquant_diag_params = None
        if args.grid in {"flatquant", "flatquant_diag"}:
            layer_flatquant_diag_params = flatquant_diag_by_layer.get(layer_name)
            if layer_flatquant_diag_params is None:
                raise KeyError(f"no FlatQuant diag params for layer {layer_name!r}")
        grid = build_grid(
            args.grid,
            weights,
            args,
            stats,
            layer_awq_scales,
            layer_flatquant_diag_params,
            awq_clip_max=layer_awq_clip_max,
        )
        if args.assignment == "rtn":
            corrected, _info = assignment.apply_to_grid(grid)
        else:
            corrected, _info = assignment.apply_to_grid(grid, stats)
        if args.assignment == "flexround":
            print(
                f"  {layer_name}: loss {_info['initial_loss']:.6g} -> "
                f"{_info['final_loss']:.6g}, changed_codes="
                f"{_info['changed_codes']} ({_info['changed_fraction']:.2%})"
            )
        module.weight.data = corrected.to(module.weight.dtype)
        del grid, corrected

    effective_group_size = (
        "per-channel" if args.assignment == "qronus" else str(args.group_size)
    )
    print(
        "quantizing",
        f"grid={args.grid}",
        f"scheme={args.scheme}",
        f"assignment={args.assignment}",
        f"bits={args.bits}",
        f"group_size={args.group_size}",
        f"effective_group_size={effective_group_size}",
        f"need_H={need_h}",
        f"device_map={args.device_map}",
        f"input_device={args.input_device}",
        f"stats_device={args.stats_device}",
    )
    collect_and_encode_awq_style(
        model,
        tokenizer,
        calibration,
        input_device,
        need_H=need_h,
        k=args.k,
        eps=args.eps,
        callback=callback,
        layer_batch_size=args.layer_batch_size,
        keep_sigma=keep_sigma,
        skip_lm_head=True,
        eig_on_cpu=args.eig_on_cpu,
        max_length=args.seqlen,
        stats_device=args.stats_device,
        max_layers=args.max_layers,
        paired_full_precision=args.assignment in PAIRED_ASSIGNMENTS,
        paired_cache_dtype=paired_cache_dtype,
        paired_reset_by_block=args.assignment == "qronus",
        paired_disable_reference_quantization=args.assignment == "qronus",
    )

    if args.no_save:
        print("quantization completed; checkpoint save skipped (--no-save)")
        return

    out = output_path(args)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    if args.grid == "flatquant":
        transform_file = "flatquant_transforms.pt"
        torch.save(
            {
                "format": "tfic-flatquant",
                "format_version": 1,
                "model": transform_model_metadata(model),
                "training": {
                    "weight_bits": args.bits,
                    "activation_bits": args.activation_bits,
                    "weight_symmetric": args.scheme == "symmetric",
                    "activation_symmetric": args.activation_symmetric,
                    "weight_group_size": args.group_size,
                    "activation_group_size": args.activation_group_size,
                },
                "layers": serialize_flatquant_transforms(
                    flatquant_transforms,
                    flatquant_clips,
                ),
                "attention": {
                    name: matrix.detach().cpu()
                    for name, matrix in flatquant_attention_transforms.items()
                },
                "attention_clips": {
                    name: {
                        key: value.detach().cpu()
                        for key, value in values.items()
                    }
                    for name, values in flatquant_attention_clips.items()
                },
            },
            Path(out) / transform_file,
        )
        manifest = {
            "version": 1,
            "method": "flatquant",
            "model": transform_model_metadata(model),
            "transform_file": transform_file,
            "activation_bits": args.activation_bits,
            "activation_symmetric": args.activation_symmetric,
            "activation_group_size": args.activation_group_size,
            "activation_clip_ratio": args.activation_clip_ratio,
            "q_bits": args.q_bits,
            "q_symmetric": args.q_symmetric,
            "q_group_size": args.q_group_size,
            "q_clip_ratio": args.q_clip_ratio,
            "k_bits": args.k_bits,
            "k_symmetric": args.k_symmetric,
            "k_group_size": args.k_group_size,
            "k_clip_ratio": args.k_clip_ratio,
            "v_bits": args.v_bits,
            "v_symmetric": args.v_symmetric,
            "v_clip_ratio": args.v_clip_ratio,
        }
        (Path(out) / TRANSFORM_MANIFEST).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    elif args.grid in {"spinquant", "spinquant_had"}:
        manifest = {
            "version": 1,
            "method": (
                "spinquant_had"
                if args.grid == "spinquant_had"
                else "spinquant_no_had"
            ),
            "model": transform_model_metadata(model),
            "activation_bits": args.activation_bits,
            "activation_symmetric": args.activation_symmetric,
            "activation_group_size": args.activation_group_size,
            "activation_clip_ratio": args.activation_clip_ratio,
            "v_bits": args.v_bits,
            "v_symmetric": args.v_symmetric,
            "v_clip_ratio": args.v_clip_ratio,
            "k_bits": args.k_bits,
            "k_symmetric": args.k_symmetric,
            "k_group_size": args.k_group_size,
            "k_clip_ratio": args.k_clip_ratio,
        }
        if args.grid == "spinquant_had":
            runtime_file = "spinquant_runtime.pt"
            torch.save(
                {
                    "r4": {
                        "K": spinquant_r4.k,
                        "had_K": spinquant_r4.had_k,
                    }
                },
                Path(out) / runtime_file,
            )
            manifest["runtime_file"] = runtime_file
        (Path(out) / TRANSFORM_MANIFEST).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
