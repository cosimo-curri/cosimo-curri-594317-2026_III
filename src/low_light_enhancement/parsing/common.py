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


@dataclass  # dataclasses keep simple data containers concise (no __init__, etc.)
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


def flatten_metrics(
    metrics: dict[str, Any],
    *,  # next options are keyword-only
    prefix: str
) -> dict[str, Any]:
    return {
        f"{prefix}_{metric_name}": value
        for metric_name, value in sorted(metrics.items())
    }


def metric_sort_value(value: Any, mode: str) -> float:
    value = float(value)

    if mode == "max":
        return -value

    if mode == "min":
        return value

    raise ValueError(f"Unsupported metric mode: {mode!r}.")


def build_fieldnames(
    rows: list[dict[str, Any]],
    preferred_columns: list[str]
) -> list[str]:
    all_columns = set().union(*(row.keys() for row in rows))

    ordered_columns = [
        column
        for column in preferred_columns
        if column in all_columns
    ]

    extra_columns = sorted(all_columns - set(ordered_columns))
    return ordered_columns + extra_columns


def write_parsed_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    write_csv_rows(
        output_path=output_path,
        rows=format_rows(rows),
        fieldnames=fieldnames
    )