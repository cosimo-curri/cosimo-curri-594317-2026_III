from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.low_light_enhancement.framework.io import (
    read_jsonl_rows,
    write_csv_rows
)


PARSED_LOGS_DIR = Path("parsed_logs")

CONFIG_COLUMNS = [
    "config_id",
    "experiment_name",
    "seed",
    "model_name",
    "channels",
    "normalization",
    "activation",
    "group_count",
    "upsampling",
    "lr",
    "weight_decay",
    "loss_name",
    "l1_weight",
    "ssim_weight",
    "epochs",
    "validation_rate",
    "batch_size",
    "num_workers",
    "mixed_precision",
    "deterministic",
    "monitor",
    "mode_monitor",
    "min_delta",
    "patience",
    "tie_breaker",
    "mode_tie_breaker"
]


# Dataclasses keep simple data containers concise (no __init__, etc.)
@dataclass
class ParsedRun:
    log_path: Path
    events_by_name: dict[str, list[dict[str, Any]]]

    def last(self, event_name: str) -> dict[str, Any]:
        return self.events_by_name[event_name][-1]


def parse_run(log_path: Path) -> ParsedRun:
    events_by_name: dict[str, list[dict[str, Any]]] = {}

    for event in read_jsonl_rows(log_path):
        events_by_name.setdefault(event["event"], []).append(event)

    return ParsedRun(
        log_path=log_path,
        events_by_name=events_by_name
    )


def load_runs(log_dir: Path) -> list[ParsedRun]:
    log_paths = sorted(log_dir.rglob("*.jsonl"))
    runs = [parse_run(log_path) for log_path in log_paths]

    if not runs:
        raise RuntimeError(f"No JSONL logs found in {log_dir}.")

    return runs


def build_csv_path(csv_name: str) -> Path:
    csv_path = Path(csv_name)

    if csv_path.suffix != ".csv":
        csv_path = csv_path.with_suffix(".csv")

    return PARSED_LOGS_DIR / csv_path.name


def format_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    return value


def format_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            column: format_cell(value)
            for column, value in row.items()
        }
        for row in rows
    ]


def metric_sort_value(value: Any, mode: str) -> float:
    value = float(value)

    if mode == "max":
        return -value

    if mode == "min":
        return value

    raise ValueError(f"Unsupported metric mode: {mode!r}.")


def extract_config_fields(
    run_start: dict[str, Any],
    *,  # next options are keyword-only
    include_seed: bool
) -> dict[str, Any]:
    config = run_start["resolved_config"]

    model_config = config["model"]
    optimizer_config = config["optimizer"]
    loss_config = config["loss"]
    training_config = config["training"]
    early_stopping_config = config["early_stopping"]

    row = {
        "config_id": run_start["config_id"],
        "experiment_name": run_start["experiment_name"],
        "model_name": model_config["name"],
        "channels": model_config["channels"],
        "normalization": model_config["normalization"],
        "activation": model_config["activation"],
        "group_count": model_config["group_count"],
        "upsampling": model_config["upsampling"],
        "lr": optimizer_config["lr"],
        "weight_decay": optimizer_config["weight_decay"],
        "loss_name": loss_config["name"],
        "l1_weight": loss_config["l1_weight"],
        "ssim_weight": loss_config["ssim_weight"],
        "epochs": training_config["epochs"],
        "validation_rate": training_config["validation_rate"],
        "batch_size": training_config["batch_size"],
        "num_workers": training_config["num_workers"],
        "mixed_precision": training_config["mixed_precision"],
        "deterministic": training_config["deterministic"],
        "monitor": early_stopping_config["monitor"],
        "mode_monitor": early_stopping_config["mode_monitor"],
        "min_delta": early_stopping_config["min_delta"],
        "patience": early_stopping_config["patience"],
        "tie_breaker": early_stopping_config["tie_breaker"],
        "mode_tie_breaker": early_stopping_config["mode_tie_breaker"]
    }

    if include_seed:
        row["seed"] = run_start["seed"]

    return row


def write_parsed_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str]
) -> None:
    write_csv_rows(
        output_path=output_path,
        rows=format_rows(rows),
        fieldnames=fieldnames
    )