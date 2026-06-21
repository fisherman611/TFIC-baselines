"""Run grid-baseline x assignment-method quantization on a real HF model.

This is the experiment entrypoint for the modular baseline folders:

* ``grid_baselines`` builds the quantization grid.
* ``assignment_methods`` assigns integer codes on that grid.

The script saves a Hugging Face checkpoint under:

    <output-dir>/<grid>_<scheme>_<assignment>

Evaluate the saved checkpoint with ``eval_ppl.py`` or ``lm_eval_runner.py``.
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from assignment_methods import (
    FlexRoundAssignment,
    GPTAQAssignment,
    GPTAQResCompAssignment,
    GPTQAssignment,
    RTNAssignment,
    TFICAssignment,
)
from eigenflip.quantization.awq_scales import scales_from_awq_run
from eigenflip.statistics.collect_fast import collect_and_encode_awq_style
from grid_baselines import (
    apply_spinquant_no_had,
    build_awq_quantization_grid,
    build_flatquant_diag_quantization_grid,
    build_spinquant_quantization_grid,
    build_vanilla_quantization_grid,
    load_spinquant_rotations,
    random_spinquant_rotations,
)

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    get_c4_calibration_data = get_wikitext2_calibration_data = None


NEED_H = {
    "flexround": True,
    "rtn": False,
    "gptaq": True,
    "gptaq_rescomp": True,
    "gptq": True,
    "tfic": True,
}
KEEP_SIGMA = {"gptaq", "gptaq_rescomp", "gptq", "tfic"}
PAIRED_ASSIGNMENTS = {"gptaq", "gptaq_rescomp"}


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
            act_order=args.gptaq_act_order,
        )
    if name == "flexround":
        return FlexRoundAssignment(
            steps=args.flexround_steps,
            lr=args.flexround_lr,
            log_divisor_bound=args.flexround_log_divisor_bound,
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
    if raw and isinstance(next(iter(raw.values())), dict):
        return scales_from_awq_run(raw)
    return {key: torch.as_tensor(value) for key, value in raw.items()}


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
    awq_scales: torch.Tensor | None,
    flatquant_diag_params: dict[str, torch.Tensor] | None,
):
    if name == "vanilla":
        return build_vanilla_quantization_grid(
            weights,
            bits=args.bits,
            group_size=args.group_size,
            scheme=args.scheme,
        )
    if name == "awq":
        if awq_scales is None:
            raise ValueError("AWQ grid requires per-layer AWQ scales")
        return build_awq_quantization_grid(
            weights,
            awq_scales,
            bits=args.bits,
            group_size=args.group_size,
            scheme=args.scheme,
        )
    if name == "flatquant_diag":
        if flatquant_diag_params is None:
            raise ValueError("FlatQuant diag grid requires per-layer scale params")
        return build_flatquant_diag_quantization_grid(
            weights,
            flatquant_diag_params["scales"],
            bits=args.bits,
            group_size=args.group_size,
            scheme=args.scheme,
            weight_clip=flatquant_diag_params.get("weight_clip", 1.0),
        )
    if name == 'spinquant':
        return build_spinquant_quantization_grid(
            weights,
            bits=args.bits,
            group_size=args.group_size,
            scheme=args.scheme,
        )
    raise ValueError(f"unknown grid baseline: {name}")


def load_calibration(tokenizer, args):
    if get_c4_calibration_data is None:
        raise RuntimeError("calibration_utils.py is not importable")
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
        choices=['vanilla', 'awq', 'flatquant_diag', 'spinquant'],
        required=True,
    )
    parser.add_argument("--assignment", choices=sorted(NEED_H), required=True)
    parser.add_argument("--scheme", choices=["asymmetric", "symmetric"], default="asymmetric")
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
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
        "--gptaq-cache-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="CPU dtype for temporary full-precision activation cache.",
    )

    parser.add_argument("--flexround-steps", type=int, default=5000)
    parser.add_argument("--flexround-lr", type=float, default=2e-4)
    parser.add_argument("--flexround-log-divisor-bound", type=float, default=6.0)
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
    args = parse_args()
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

    if args.grid == 'spinquant':
        if args.spinquant_rotations_pt and args.spinquant_random_rotations:
            raise ValueError(
                'use either --spinquant-rotations-pt or '
                '--spinquant-random-rotations, not both'
            )
        if not args.spinquant_rotations_pt and not args.spinquant_random_rotations:
            raise ValueError(
                '--grid spinquant requires --spinquant-rotations-pt or '
                '--spinquant-random-rotations'
            )
        head_dim = model.config.hidden_size // model.config.num_attention_heads
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

    calibration = load_calibration(tokenizer, args)
    awq_scales_by_layer = load_awq_scales(args.awq_scales_pt) if args.grid == "awq" else {}
    flatquant_diag_by_layer = (
        load_flatquant_diag_params(args.flatquant_params_pt)
        if args.grid == "flatquant_diag"
        else {}
    )
    assignment = build_assignment(args.assignment, args)

    need_h = assignment_needs_h(args.assignment, args.k)
    # FlexRound can optimize the diagonal-plus-mean surrogate when k=0.
    # This avoids materializing per-layer d x d Gram matrices and provides a
    # practical one-pass smoke path. Full benchmark runs should keep k > 0.
    keep_sigma = args.assignment in KEEP_SIGMA
    gptaq_cache_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.gptaq_cache_dtype]

    def callback(layer_name, module, stats):
        weights = module.weight.data
        layer_awq_scales = None
        if args.grid == "awq":
            layer_awq_scales = awq_scales_by_layer.get(layer_name)
            if layer_awq_scales is None:
                raise KeyError(f"no AWQ scales for layer {layer_name!r}")
        layer_flatquant_diag_params = None
        if args.grid == "flatquant_diag":
            layer_flatquant_diag_params = flatquant_diag_by_layer.get(layer_name)
            if layer_flatquant_diag_params is None:
                raise KeyError(f"no FlatQuant diag params for layer {layer_name!r}")
        grid = build_grid(
            args.grid,
            weights,
            args,
            layer_awq_scales,
            layer_flatquant_diag_params,
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

    print(
        "quantizing",
        f"grid={args.grid}",
        f"scheme={args.scheme}",
        f"assignment={args.assignment}",
        f"bits={args.bits}",
        f"group_size={args.group_size}",
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
        paired_cache_dtype=gptaq_cache_dtype,
    )

    if args.no_save:
        print("quantization completed; checkpoint save skipped (--no-save)")
        return

    out = output_path(args)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
