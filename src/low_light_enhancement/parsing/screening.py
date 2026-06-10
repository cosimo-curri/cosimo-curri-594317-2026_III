from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.low_light_enhancement.framework.io import relative_path
from src.low_light_enhancement.parsing.common import (
    ParsedRun,
    build_csv_path,
    load_runs,
    metric_sort_value,
    write_parsed_csv
)


@dataclass  # dataclasses keep simple data containers concise (no __init__, etc.)
class ScreeningRecord:
    row: dict[str, Any]


# Screening CSV columns include the resolved training configuration +
# the best validation results needed to rank the runs
SCREENING_COLUMNS = [
    "rank",
    "num_parameters",
    "best_epoch",
    "best_val_ssim",
    "best_val_psnr",
    "best_val_mae",
    "best_val_loss",
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


def build_screening_record(run: ParsedRun) -> ScreeningRecord:
    run_start = run.last("run_start")
    best_summary = run.last("best_validation_summary")

    config = run_start["resolved_config"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    loss_config = config["loss"]
    training_config = config["training"]
    early_stopping_config = config["early_stopping"]
    metrics = best_summary["metrics"]

    row = {
        "num_parameters": run_start["num_parameters"],
        "best_epoch": best_summary["best_epoch"],
        "best_val_ssim": metrics["ssim"],
        "best_val_psnr": metrics["psnr"],
        "best_val_mae": metrics["mae"],
        "best_val_loss": metrics["loss"],
        "config_id": run_start["config_id"],
        "experiment_name": run_start["experiment_name"],
        "seed": run_start["seed"],
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

    return ScreeningRecord(row=row)


def sort_screening_records(
    records: list[ScreeningRecord]
) -> list[ScreeningRecord]:
    first_row = records[0].row

    monitor = first_row["monitor"]
    tie_breaker = first_row["tie_breaker"]
    mode_monitor = first_row["mode_monitor"]
    mode_tie_breaker = first_row["mode_tie_breaker"]

    return sorted(
        records,
        key=lambda record: (
            metric_sort_value(
                record.row[f"best_val_{monitor}"],
                mode_monitor,
            ),
            metric_sort_value(
                record.row[f"best_val_{tie_breaker}"],
                mode_tie_breaker,
            ),
            record.row["config_id"]
        )
    )


def rank_screening_records(
    records: list[ScreeningRecord]
) -> list[ScreeningRecord]:
    ranked_records = []

    for rank, record in enumerate(records, start=1):
        row = dict(record.row)
        row["rank"] = rank
        ranked_records.append(ScreeningRecord(row=row))

    return ranked_records


def run_screening_parsing(logs_dir: Path, csv_name: str) -> None:
    runs = load_runs(logs_dir)
    records = [build_screening_record(run) for run in runs]
    records = rank_screening_records(sort_screening_records(records))
    rows = [record.row for record in records]
    csv_path = build_csv_path(csv_name)

    write_parsed_csv(
        output_path=csv_path,
        rows=rows,
        fieldnames=SCREENING_COLUMNS
    )

    print(f"Parsed {len(records)} screening run(s)")
    print(f"CSV: {relative_path(csv_path)}")