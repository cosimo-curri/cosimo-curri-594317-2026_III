from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from src.low_light_enhancement.framework.io import load_config


GRID_SECTIONS = [
    "model",
    "optimizer",
    "loss",
    "training",
    "early_stopping"
]

NON_GRID_KEYS = {
    "name",
    "monitor",
    "mode_monitor",
    "tie_breaker",
    "mode_tie_breaker"
}

SELECTION_CONFIG_HASH_EXCLUDED_PATHS: tuple[tuple[str, ...], ...] = (
    ("experiment", "name"),
    ("experiment", "seed"),
    ("experiment", "restore_checkpoint"),
    ("evaluation",),
    ("data", "test_manifests")
)


def build_selection_config_hash(config: dict[str, Any]) -> str:
    config_for_hash = copy.deepcopy(config)

    for path in SELECTION_CONFIG_HASH_EXCLUDED_PATHS:
        remove_nested_key(config_for_hash, path)

    serialized_config = json.dumps(
        config_for_hash,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":")
    )

    return hashlib.sha1(
        serialized_config.encode("utf-8")
    ).hexdigest()[:12]


def remove_nested_key(
    config: dict[str, Any],
    path: tuple[str, ...]
) -> None:
    current: Any = config

    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return

        current = current[key]

    if isinstance(current, dict):
        current.pop(path[-1], None)


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    return load_config(config_path)


def build_resolved_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = as_list(config["experiment"]["seed"])
    base_config = copy.deepcopy(config)
    base_config["experiment"].pop("seed")

    resolved_configs: list[dict[str, Any]] = []

    for section_config in expand_grid_sections(base_config):
        for seed in seeds:
            resolved_config = copy.deepcopy(section_config)
            resolved_config["experiment"]["seed"] = seed
            resolved_configs.append(resolved_config)

    return resolved_configs


def expand_grid_sections(config: dict[str, Any]) -> list[dict[str, Any]]:
    section_alternatives: list[list[tuple[str, dict[str, Any]]]] = []

    for section_name in GRID_SECTIONS:
        if section_name in config:
            alternatives = expand_section(section_name, config[section_name])
        else:
            alternatives = [(section_name, {})]

        section_alternatives.append(alternatives)

    resolved_configs: list[dict[str, Any]] = []

    for combination in itertools.product(*section_alternatives):
        resolved_config = copy.deepcopy(config)

        for section_name, section_config in combination:
            resolved_config[section_name] = section_config

        resolved_configs.append(resolved_config)

    return resolved_configs


def expand_section(
    section_name: str,
    section_config: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    keys = list(section_config)
    value_options = []

    for key in keys:
        value = section_config[key]

        if key in NON_GRID_KEYS:
            value_options.append([value])
        elif isinstance(value, list):
            value_options.append(value)
        else:
            value_options.append([value])

    alternatives: list[tuple[str, dict[str, Any]]] = []

    for values in itertools.product(*value_options):
        alternatives.append(
            (
                section_name,
                {
                    key: copy.deepcopy(value)
                    for key, value in zip(keys, values, strict=True)
                }
            )
        )

    return alternatives


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value

    return [value]