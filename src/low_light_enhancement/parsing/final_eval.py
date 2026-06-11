from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.low_light_enhancement.framework.io import relative_path
from src.low_light_enhancement.parsing.common import (
    DETAILED_CONFIG_COLUMNS,
    ParsedRun,
    build_csv_path,
    compute_std,
    extract_detailed_config_fields,
    group_sort_key,
    load_runs,
    write_parsed_csv
)


FINAL_EVAL_ID_COLUMNS = [
    "row_type",
    "dataset",
    "group_by",
    "group_value",
    "num_samples",
    "config_id",
    "num_parameters"
]

FINAL_EVAL_VALIDATION_COLUMNS = [
    "mean_val_ssim",
    "std_val_ssim",
    "mean_val_psnr",
    "std_val_psnr",
    "mean_val_mae",
    "std_val_mae",
    "mean_val_loss",
    "std_val_loss"
]

FINAL_EVAL_TEST_METRIC_COLUMNS = [
    "mean_test_ssim",
    "std_test_ssim",
    "mean_test_psnr",
    "std_test_psnr",
    "mean_test_mae",
    "std_test_mae",
    "mean_test_niqe",
    "std_test_niqe",
    "mean_test_brisque",
    "std_test_brisque",
    "mean_test_mean_luminance",
    "std_test_mean_luminance",
    "mean_test_dark_pixel_ratio",
    "std_test_dark_pixel_ratio",
    "mean_test_bright_clipping_ratio",
    "std_test_bright_clipping_ratio",
    "mean_test_luminance_std",
    "std_test_luminance_std",
    "mean_test_rgb_channel_imbalance",
    "std_test_rgb_channel_imbalance",
    "mean_test_laplacian_sharpness",
    "std_test_laplacian_sharpness"
]

FINAL_EVAL_COLUMNS = (
    FINAL_EVAL_ID_COLUMNS
    + DETAILED_CONFIG_COLUMNS
    + FINAL_EVAL_VALIDATION_COLUMNS
    + FINAL_EVAL_TEST_METRIC_COLUMNS
)

FINAL_EVAL_TEST_METRICS = [
    "ssim",
    "psnr",
    "mae",
    "niqe",
    "brisque",
    "mean_luminance",
    "dark_pixel_ratio",
    "bright_clipping_ratio",
    "luminance_std",
    "rgb_channel_imbalance",
    "laplacian_sharpness"
]

ROW_TYPE_ORDER = {
    "dataset": 0,
    "group": 1
}


# Dataclasses keep simple data containers concise (no __init__, etc.)
@dataclass
class FinalEvalRunRecord:
    row: dict[str, Any]
    metrics_mean: dict[str, float]


@dataclass
class FinalEvalRecord:
    row: dict[str, Any]


def build_base_run_fields(run: ParsedRun) -> dict[str, Any]:
    run_start = run.last("run_start")
    best_summary = run.last("best_validation_summary")
    metrics = best_summary["metrics"]

    row = {
        "config_id": run_start["config_id"],
        "num_parameters": run_start["num_parameters"],
        "val_ssim": metrics["ssim"],
        "val_psnr": metrics["psnr"],
        "val_mae": metrics["mae"],
        "val_loss": metrics["loss"]
    }

    row.update(extract_detailed_config_fields(run_start))
    return row


def build_final_eval_run_record(
    base_row: dict[str, Any],
    event: dict[str, Any],
    row_type: str
) -> FinalEvalRunRecord:
    row = dict(base_row)

    row.update({
        "row_type": row_type,
        "dataset": event["dataset"],
        "group_by": event.get("group_by", ""),
        "group_value": event.get("group_value", ""),
        "num_samples": event["num_samples"]
    })

    return FinalEvalRunRecord(
        row=row,
        metrics_mean=event["metrics_mean"]
    )


def build_final_eval_run_records(run: ParsedRun) -> list[FinalEvalRunRecord]:
    base_row = build_base_run_fields(run)
    records = []

    for event in run.events_by_name.get("test_dataset_metrics", []):
        records.append(
            build_final_eval_run_record(
                base_row=base_row,
                event=event,
                row_type="dataset"
            )
        )

    for event in run.events_by_name.get("test_group_metrics", []):
        records.append(
            build_final_eval_run_record(
                base_row=base_row,
                event=event,
                row_type="group"
            )
        )

    if not records:
        raise RuntimeError(
            f"No final evaluation metrics found in {run.log_path}."
        )

    return records


def group_records_by_result_key(
    records: list[FinalEvalRunRecord]
) -> dict[tuple[str, str, str, str, str], list[FinalEvalRunRecord]]:
    grouped_records: dict[
        tuple[str, str, str, str, str],
        list[FinalEvalRunRecord]
    ] = defaultdict(list)

    for record in records:
        row = record.row

        key = (
            row["config_id"],
            row["row_type"],
            row["dataset"],
            row.get("group_by") or "",
            row.get("group_value") or ""
        )

        grouped_records[key].append(record)

    return dict(grouped_records)


def values_for(
    records: list[FinalEvalRunRecord],
    field_name: str
) -> list[Any]:
    return [
        record.row[field_name]
        for record in records
    ]


def metric_values(
    records: list[FinalEvalRunRecord],
    metric_name: str
) -> list[float]:
    return [
        float(record.metrics_mean[metric_name])
        for record in records
        if metric_name in record.metrics_mean
    ]


def add_validation_summary(
    row: dict[str, Any],
    records: list[FinalEvalRunRecord]
) -> None:
    for metric_name in ["val_ssim", "val_psnr", "val_mae", "val_loss"]:
        values = [float(value) for value in values_for(records, metric_name)]

        row[f"mean_{metric_name}"] = mean(values)
        row[f"std_{metric_name}"] = compute_std(values)


def add_test_metric_summary(
    row: dict[str, Any],
    records: list[FinalEvalRunRecord]
) -> None:
    for metric_name in FINAL_EVAL_TEST_METRICS:
        values = metric_values(records, metric_name)

        if not values:
            continue

        row[f"mean_test_{metric_name}"] = mean(values)
        row[f"std_test_{metric_name}"] = compute_std(values)


def add_common_config_fields(
    row: dict[str, Any],
    first_row: dict[str, Any]
) -> None:
    for column in DETAILED_CONFIG_COLUMNS:
        row[column] = first_row[column]


def common_sample_count(records: list[FinalEvalRunRecord]) -> Any:
    sample_counts = values_for(records, "num_samples")
    unique_counts = sorted(set(sample_counts))

    if len(unique_counts) == 1:
        return unique_counts[0]

    return sample_counts


def build_final_eval_record(
    records: list[FinalEvalRunRecord]
) -> FinalEvalRecord:
    first_row = records[0].row

    row = {
        "row_type": first_row["row_type"],
        "dataset": first_row["dataset"],
        "group_by": first_row.get("group_by") or "",
        "group_value": first_row.get("group_value") or "",
        "num_samples": common_sample_count(records),
        "config_id": first_row["config_id"],
        "num_parameters": first_row["num_parameters"]
    }

    add_common_config_fields(row, first_row)
    add_validation_summary(row, records)
    add_test_metric_summary(row, records)

    return FinalEvalRecord(row=row)


def build_final_eval_records(
    run_records: list[FinalEvalRunRecord]
) -> list[FinalEvalRecord]:
    grouped_records = group_records_by_result_key(run_records)

    return [
        build_final_eval_record(records)
        for records in grouped_records.values()
    ]


def build_dataset_order(
    records: list[FinalEvalRunRecord]
) -> dict[str, int]:
    dataset_order: dict[str, int] = {}

    for record in records:
        dataset = record.row["dataset"]

        if dataset not in dataset_order:
            dataset_order[dataset] = len(dataset_order)

    return dataset_order


def sort_final_eval_records(
    records: list[FinalEvalRecord],
    dataset_order: dict[str, int]
) -> list[FinalEvalRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.row["config_id"],
            dataset_order.get(record.row["dataset"], len(dataset_order)),
            ROW_TYPE_ORDER[record.row["row_type"]],
            record.row.get("group_by") or "",
            group_sort_key(record.row.get("group_value") or "")
        )
    )


def run_final_eval_parsing(logs_dir: Path, output_name: str) -> None:
    runs = load_runs(logs_dir)

    run_records = [
        record
        for run in runs
        for record in build_final_eval_run_records(run)
    ]

    dataset_order = build_dataset_order(run_records)
    records = build_final_eval_records(run_records)
    records = sort_final_eval_records(records, dataset_order)
    rows = [record.row for record in records]
    csv_path = build_csv_path(output_name)

    write_parsed_csv(
        output_path=csv_path,
        rows=rows,
        fieldnames=FINAL_EVAL_COLUMNS
    )

    print(f"Parsed {len(runs)} final evaluation run(s)")
    print(f"Aggregated {len(records)} final evaluation row(s)")
    print(f"CSV: {relative_path(csv_path)}")