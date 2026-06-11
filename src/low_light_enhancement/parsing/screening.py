from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


SCREENING_RESULT_COLUMNS = [
    "rank",
    "num_parameters",
    "best_epoch",
    "best_val_ssim",
    "best_val_psnr",
    "best_val_mae",
    "best_val_loss"
]

# The screening CSV stores training configuration fields +
# best validation metrics for ranking the runs
SCREENING_COLUMNS = SCREENING_RESULT_COLUMNS + CONFIG_COLUMNS


# Dataclasses keep simple data containers concise (no __init__, etc.)
@dataclass
class ScreeningRecord:
    row: dict[str, Any]


def build_screening_record(run: ParsedRun) -> ScreeningRecord:
    run_start = run.last("run_start")
    best_summary = run.last("best_validation_summary")
    metrics = best_summary["metrics"]

    row = {
        "num_parameters": run_start["num_parameters"],
        "best_epoch": best_summary["best_epoch"],
        "best_val_ssim": metrics["ssim"],
        "best_val_psnr": metrics["psnr"],
        "best_val_mae": metrics["mae"],
        "best_val_loss": metrics["loss"]
    }

    row.update(extract_config_fields(run_start, include_seed=True))
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


def run_screening_parsing(logs_dir: Path, output_name: str) -> None:
    runs = load_runs(logs_dir)
    records = [build_screening_record(run) for run in runs]
    records = rank_screening_records(sort_screening_records(records))
    rows = [record.row for record in records]
    csv_path = build_csv_path(output_name)

    write_parsed_csv(
        output_path=csv_path,
        rows=rows,
        fieldnames=SCREENING_COLUMNS
    )

    print(f"Parsed {len(records)} screening run(s)")
    print(f"CSV: {relative_path(csv_path)}")