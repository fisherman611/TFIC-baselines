from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from baseline_utils.runtime import (
    collect_lm_eval_wandb_metrics,
    is_numeric_metric_value,
    resolve_wandb_api_key,
)


PPL_WANDB_NAMES = {
    "wikitext2": "perplexity/WikiText-2",
    "wiki": "perplexity/WikiText-2",
    "c4": "perplexity/C4",
}


def load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def collect_ppl_wandb_metrics(payload: dict[str, Any]) -> dict[str, float]:
    ppl_values = payload.get("ppl", payload)
    if not isinstance(ppl_values, dict):
        return {}

    metrics = {}
    for dataset_name, value in ppl_values.items():
        if not is_numeric_metric_value(value):
            continue
        key = PPL_WANDB_NAMES.get(str(dataset_name).lower(), f"perplexity/{dataset_name}")
        metrics[key] = float(value)
    return metrics


def collect_lm_eval_summary_wandb_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    return collect_lm_eval_wandb_metrics(payload)


def log_to_wandb(
    *,
    project: str,
    entity: str | None,
    run_name: str,
    metrics: dict[str, float],
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> bool:
    if not metrics:
        print("wandb: no numeric metrics to log")
        return False

    try:
        import wandb
    except ImportError:
        print("wandb: package is not installed; skipping log")
        return False

    api_key = resolve_wandb_api_key()
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception as exc:
            print(f"wandb: login failed ({exc}); continuing with local/default auth")

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags or [],
        config=config or {},
        reinit=True,
    )
    try:
        wandb.log(metrics)
        for key, value in (summary or {}).items():
            wandb.summary[key] = value
    finally:
        run.finish()

    print(f"wandb: logged {len(metrics)} metrics to project={project} run={run_name}")
    return True


def wandb_enabled_from_env(default: bool = False) -> bool:
    value = os.getenv("USE_WANDB")
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
