from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
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
    "family_specific_config",
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

DETAILED_CONFIG_COLUMNS = [
    "model_name",
    "channels",
    "normalization",
    "activation",
    "group_count",
    "upsampling",
    "family_specific_config",
    "optimizer_name",
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
    "early_stopping_monitor",
    "early_stopping_mode_monitor",
    "early_stopping_min_delta",
    "early_stopping_patience",
    "early_stopping_tie_breaker",
    "early_stopping_mode_tie_breaker",
    "augmentation_enabled",
    "horizontal_flip_p",
    "random_resized_crop_enabled",
    "random_resized_crop_scale",
    "random_resized_crop_ratio",
    "color_jitter_enabled",
    "color_jitter_p",
    "color_jitter_brightness",
    "color_jitter_contrast",
    "color_jitter_saturation",
    "gamma_jitter_enabled",
    "gamma_jitter_p",
    "gamma_jitter_range",
    "noise_enabled",
    "noise_p",
    "noise_std_range"
]

COMMON_CONFIG_KEYS_BY_SECTION = {
    "model": {
        "name",
        "channels",
        "normalization",
        "activation",
        "group_count",
        "upsampling"
    },
    "optimizer": {
        "name",
        "lr",
        "weight_decay"
    },
    "loss": {
        "name",
        "l1_weight",
        "ssim_weight"
    },
    "training": {
        "epochs",
        "validation_rate",
        "batch_size",
        "num_workers",
        "mixed_precision",
        "deterministic"
    },
    "early_stopping": {
        "monitor",
        "mode_monitor",
        "min_delta",
        "patience",
        "tie_breaker",
        "mode_tie_breaker"
    }
}


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


def build_csv_path(output_name: str) -> Path:
    csv_path = Path(output_name)

    if csv_path.suffix != ".csv":
        csv_path = csv_path.with_suffix(".csv")

    return PARSED_LOGS_DIR / csv_path.name


def build_json_path(output_name: str) -> Path:
    json_path = Path(output_name)

    if json_path.suffix != ".json":
        json_path = json_path.with_suffix(".json")

    return PARSED_LOGS_DIR / json_path.name


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


def compute_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    return stdev(values)


def group_sort_key(value: Any) -> tuple[int, int | str]:
    value = "" if value is None else str(value)

    if value.isdigit():
        return (0, int(value))

    return (1, value)


def value_range(
    config: dict[str, Any],
    min_key: str,
    max_key: str
) -> list[Any]:
    return [config[min_key], config[max_key]]


def extract_family_specific_config(
    config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    family_specific_config: dict[str, dict[str, Any]] = {}

    for section_name, common_keys in COMMON_CONFIG_KEYS_BY_SECTION.items():
        section_config = config.get(section_name, {})

        if not isinstance(section_config, dict):
            continue

        specific_values = {
            key: value
            for key, value in section_config.items()
            if key not in common_keys
        }

        if specific_values:
            family_specific_config[section_name] = specific_values

    return family_specific_config


def extract_detailed_config_fields(
    run_start: dict[str, Any]
) -> dict[str, Any]:
    config = run_start["resolved_config"]

    model_config = config["model"]
    optimizer_config = config["optimizer"]
    loss_config = config["loss"]
    training_config = config["training"]
    early_stopping_config = config["early_stopping"]
    augmentation_config = config["augmentation"]

    horizontal_flip_config = augmentation_config["horizontal_flip"]
    random_crop_config = augmentation_config["random_resized_crop"]
    color_jitter_config = augmentation_config["color_jitter"]
    gamma_jitter_config = augmentation_config["gamma_jitter"]
    noise_config = augmentation_config["sensor_noise"]

    return {
        "model_name": model_config["name"],
        "channels": model_config.get("channels", ""),
        "normalization": model_config.get("normalization", ""),
        "activation": model_config.get("activation", ""),
        "group_count": model_config.get("group_count", ""),
        "upsampling": model_config.get("upsampling", ""),
        "family_specific_config": extract_family_specific_config(config),
        "optimizer_name": optimizer_config.get("name", "adamw"),
        "lr": optimizer_config.get("lr", ""),
        "weight_decay": optimizer_config.get("weight_decay", ""),
        "loss_name": loss_config["name"],
        "l1_weight": loss_config.get("l1_weight", ""),
        "ssim_weight": loss_config.get("ssim_weight", ""),
        "epochs": training_config.get("epochs", ""),
        "validation_rate": training_config.get("validation_rate", ""),
        "batch_size": training_config.get("batch_size", ""),
        "num_workers": training_config.get("num_workers", ""),
        "mixed_precision": training_config.get("mixed_precision", ""),
        "deterministic": training_config.get("deterministic", ""),
        "early_stopping_monitor": early_stopping_config.get("monitor", ""),
        "early_stopping_mode_monitor": (
            early_stopping_config.get("mode_monitor", "")
        ),
        "early_stopping_min_delta": early_stopping_config.get(
            "min_delta",
            ""
        ),
        "early_stopping_patience": early_stopping_config.get("patience", ""),
        "early_stopping_tie_breaker": early_stopping_config.get(
            "tie_breaker",
            ""
        ),
        "early_stopping_mode_tie_breaker": (
            early_stopping_config.get("mode_tie_breaker", "")
        ),
        "augmentation_enabled": augmentation_config.get("enabled", ""),
        "horizontal_flip_p": horizontal_flip_config.get("p", ""),
        "random_resized_crop_enabled": random_crop_config.get("enabled", ""),
        "random_resized_crop_scale": value_range(
            random_crop_config,
            "scale_min",
            "scale_max"
        ),
        "random_resized_crop_ratio": value_range(
            random_crop_config,
            "ratio_min",
            "ratio_max"
        ),
        "color_jitter_enabled": color_jitter_config.get("enabled", ""),
        "color_jitter_p": color_jitter_config.get("p", ""),
        "color_jitter_brightness": value_range(
            color_jitter_config,
            "brightness_min",
            "brightness_max"
        ),
        "color_jitter_contrast": value_range(
            color_jitter_config,
            "contrast_min",
            "contrast_max"
        ),
        "color_jitter_saturation": value_range(
            color_jitter_config,
            "saturation_min",
            "saturation_max"
        ),
        "gamma_jitter_enabled": gamma_jitter_config.get("enabled", ""),
        "gamma_jitter_p": gamma_jitter_config.get("p", ""),
        "gamma_jitter_range": value_range(
            gamma_jitter_config,
            "gamma_min",
            "gamma_max"
        ),
        "noise_enabled": noise_config.get("enabled", ""),
        "noise_p": noise_config.get("p", ""),
        "noise_std_range": value_range(
            noise_config,
            "sigma_min",
            "sigma_max"
        )
    }


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
        "channels": model_config.get("channels", ""),
        "normalization": model_config.get("normalization", ""),
        "activation": model_config.get("activation", ""),
        "group_count": model_config.get("group_count", ""),
        "upsampling": model_config.get("upsampling", ""),
        "family_specific_config": extract_family_specific_config(config),
        "lr": optimizer_config.get("lr", ""),
        "weight_decay": optimizer_config.get("weight_decay", ""),
        "loss_name": loss_config["name"],
        "l1_weight": loss_config.get("l1_weight", ""),
        "ssim_weight": loss_config.get("ssim_weight", ""),
        "epochs": training_config.get("epochs", ""),
        "validation_rate": training_config.get("validation_rate", ""),
        "batch_size": training_config.get("batch_size", ""),
        "num_workers": training_config.get("num_workers", ""),
        "mixed_precision": training_config.get("mixed_precision", ""),
        "deterministic": training_config.get("deterministic", ""),
        "monitor": early_stopping_config.get("monitor", ""),
        "mode_monitor": early_stopping_config.get("mode_monitor", ""),
        "min_delta": early_stopping_config.get("min_delta", ""),
        "patience": early_stopping_config.get("patience", ""),
        "tie_breaker": early_stopping_config.get("tie_breaker", ""),
        "mode_tie_breaker": early_stopping_config.get(
            "mode_tie_breaker",
            ""
        )
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