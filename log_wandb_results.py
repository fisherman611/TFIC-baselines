from __future__ import annotations

import argparse
import os

from runtime_utils import load_runtime_env
from wandb_utils import (
    collect_lm_eval_summary_wandb_metrics,
    collect_ppl_wandb_metrics,
    load_json,
    log_to_wandb,
    wandb_enabled_from_env,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log saved PPL/lm-eval result JSON files to Weights & Biases."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--ppl-json", default=None)
    parser.add_argument("--lm-eval-summary-json", default=None)

    parser.add_argument("--grid", default=None)
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--assignment", default=None)
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--calib-dataset", default=None)
    parser.add_argument("--n-calib", type=int, default=None)
    parser.add_argument("--seqlen", type=int, default=None)

    parser.add_argument("--use-wandb", action="store_true", default=wandb_enabled_from_env(False))
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "tfic-baselines"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    return parser.parse_args()


def main():
    load_runtime_env()
    args = parse_args()

    if not args.use_wandb:
        print("wandb: disabled; pass --use-wandb or set USE_WANDB=1")
        return

    ppl_payload = load_json(args.ppl_json)
    lm_eval_summary = load_json(args.lm_eval_summary_json)

    metrics = {}
    metrics.update(collect_ppl_wandb_metrics(ppl_payload))
    metrics.update(collect_lm_eval_summary_wandb_metrics(lm_eval_summary))

    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"use_wandb", "wandb_project", "wandb_entity"}
        and value is not None
    }

    tags = []
    for key in ("grid", "scheme", "assignment"):
        value = getattr(args, key)
        if value:
            tags.append(f"{key}:{value}")
    if args.model_path:
        tags.append(f"model:{args.model_path}")

    summary = {
        "model_path": args.model_path,
        "checkpoint_dir": args.checkpoint_dir,
        "ppl_json": args.ppl_json,
        "lm_eval_summary_json": args.lm_eval_summary_json,
    }

    log_to_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.run_name,
        metrics=metrics,
        config=config,
        tags=tags,
        summary=summary,
    )


if __name__ == "__main__":
    main()
