from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.low_light_enhancement.framework.io import relative_path
from src.low_light_enhancement.parsing.common import (
    CONFIG_COLUMNS,
    ParsedRun,
    build_csv_path,
    extract_config_fields,
    load_runs,
    metric_sort_value,
    write_parsed_csv
)


MULTI_SEED_CONFIG_COLUMNS = [
    column
    for column in CONFIG_COLUMNS
    if column != "seed"
]

MULTI_SEED_RESULT_COLUMNS = [
    "rank",
    "config_id",
    "seeds",
    "num_parameters",
    "mean_val_ssim",
    "std_val_ssim",
    "mean_val_psnr",
    "std_val_psnr",
    "mean_val_mae",
    "std_val_mae",
    "mean_val_loss",
    "std_val_loss"
]

# The multi-seed CSV stores one aggregated row per configuration
MULTI_SEED_COLUMNS = MULTI_SEED_RESULT_COLUMNS + [
    column
    for column in MULTI_SEED_CONFIG_COLUMNS
    if column != "config_id"
]


# Dataclasses keep simple data containers concise (no __init__, etc.)
@dataclass
class MultiSeedRunRecord:
    row: dict[str, Any]


@dataclass
class MultiSeedRecord:
    row: dict[str, Any]


def build_multi_seed_run_record(run: ParsedRun) -> MultiSeedRunRecord:
    run_start = run.last("run_start")
    best_summary = run.last("best_validation_summary")
    metrics = best_summary["metrics"]

    row = {
        "num_parameters": run_start["num_parameters"],
        "val_ssim": metrics["ssim"],
        "val_psnr": metrics["psnr"],
        "val_mae": metrics["mae"],
        "val_loss": metrics["loss"]
    }

    row.update(extract_config_fields(run_start, include_seed=True))
    return MultiSeedRunRecord(row=row)


def group_records_by_config_id(
    records: list[MultiSeedRunRecord]
) -> dict[str, list[MultiSeedRunRecord]]:
    grouped_records: dict[str, list[MultiSeedRunRecord]] = defaultdict(list)

    for record in records:
        grouped_records[record.row["config_id"]].append(record)

    return dict(grouped_records)


def metric_values(
    records: list[MultiSeedRunRecord],
    metric_name: str
) -> list[float]:
    return [
        float(record.row[metric_name])
        for record in records
    ]


def build_metric_summary(
    records: list[MultiSeedRunRecord],
    metric_name: str
) -> dict[str, float]:
    values = metric_values(records, metric_name)

    return {
        f"mean_{metric_name}": mean(values),
        f"std_{metric_name}": stdev(values)
    }


def build_multi_seed_record(
    config_id: str,
    records: list[MultiSeedRunRecord]
) -> MultiSeedRecord:
    first_row = records[0].row
    seeds = sorted(record.row["seed"] for record in records)

    row = {
        "config_id": config_id,
        "seeds": seeds,
        "num_parameters": first_row["num_parameters"]
    }

    for metric_name in ["val_ssim", "val_psnr", "val_mae", "val_loss"]:
        row.update(build_metric_summary(records, metric_name))

    for column in MULTI_SEED_CONFIG_COLUMNS:
        if column not in {"config_id", "seed"}:
            row[column] = first_row[column]

    return MultiSeedRecord(row=row)


def build_multi_seed_records(
    run_records: list[MultiSeedRunRecord]
) -> list[MultiSeedRecord]:
    grouped_records = group_records_by_config_id(run_records)

    return [
        build_multi_seed_record(config_id, records)
        for config_id, records in grouped_records.items()
    ]


def sort_multi_seed_records(
    records: list[MultiSeedRecord]
) -> list[MultiSeedRecord]:
    first_row = records[0].row

    monitor = first_row["monitor"]
    tie_breaker = first_row["tie_breaker"]
    mode_monitor = first_row["mode_monitor"]
    mode_tie_breaker = first_row["mode_tie_breaker"]

    return sorted(
        records,
        key=lambda record: (
            metric_sort_value(
                record.row[f"mean_val_{monitor}"],
                mode_monitor,
            ),
            float(record.row[f"std_val_{monitor}"]),
            metric_sort_value(
                record.row[f"mean_val_{tie_breaker}"],
                mode_tie_breaker,
            ),
            record.row["config_id"]
        )
    )


def rank_multi_seed_records(
    records: list[MultiSeedRecord]
) -> list[MultiSeedRecord]:
    ranked_records = []

    for rank, record in enumerate(records, start=1):
        row = dict(record.row)
        row["rank"] = rank
        ranked_records.append(MultiSeedRecord(row=row))

    return ranked_records


def run_multi_seed_parsing(logs_dir: Path, output_name: str) -> None:
    runs = load_runs(logs_dir)
    run_records = [build_multi_seed_run_record(run) for run in runs]
    records = build_multi_seed_records(run_records)
    records = rank_multi_seed_records(sort_multi_seed_records(records))
    rows = [record.row for record in records]
    csv_path = build_csv_path(output_name)

    write_parsed_csv(
        output_path=csv_path,
        rows=rows,
        fieldnames=MULTI_SEED_COLUMNS
    )

    print(f"Parsed {len(run_records)} multi-seed run(s)")
    print(f"Aggregated {len(records)} configuration(s)")
    print(f"CSV: {relative_path(csv_path)}")