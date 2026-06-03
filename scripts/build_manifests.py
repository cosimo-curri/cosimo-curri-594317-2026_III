from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from bootstrap import add_project_root_to_path


add_project_root_to_path()  # allows imports from src


from src.low_light_enhancement.framework.io import (
    load_config,
    relative_path,
    write_csv_rows
)


MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "input_path",
    "target_path",
    "category",
    "illumination_level"
]

SPLIT_ORDER = {  # manifests' row order
    "train": 0,
    "validation": 1,
    "test": 2
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(  # --help
        description=(
            "Build ExDark, LOL-v1, LOL-v2, and MILLs manifests "
            "from a YAML configuration file."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file."
    )

    return parser.parse_args()


def list_image_files(directory: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"

    return sorted(
        path.resolve()
        for path in directory.glob(pattern)
        if path.is_file()
    )


def iter_split_sources(split_config: dict[str, Any]) -> list[dict[str, Any]]:
    return split_config.get("sources", [split_config])


def build_category_dirs_rows(
    dataset_id: str,
    dataset_root: Path,
    split_name: str,
    split_config: dict[str, Any],
    pairing_config: dict[str, Any]  # single dispatcher
) -> list[dict[str, str]]:
    input_dir = (dataset_root / split_config["input_dir"]).resolve()
    image_paths = list_image_files(input_dir, recursive=True)

    rows: list[dict[str, str]] = []

    for input_path in image_paths:
        category = input_path.parent.relative_to(input_dir).parts[0]

        rows.append(
            {
                "dataset": dataset_id,
                "split": split_name,
                "input_path": relative_path(input_path),
                "target_path": "",
                "category": category,
                "illumination_level": ""
            }
        )

    return rows


def build_same_filename_rows(
    dataset_id: str,
    dataset_root: Path,
    split_name: str,
    split_config: dict[str, Any],
    pairing_config: dict[str, Any]  # single dispatcher
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for source_config in iter_split_sources(split_config):
        input_dir = (dataset_root / source_config["input_dir"]).resolve()
        target_dir = (dataset_root / source_config["target_dir"]).resolve()

        input_paths = list_image_files(input_dir, recursive=False)

        target_paths = {
            target_path.name: target_path
            for target_path in list_image_files(target_dir, recursive=False)
        }

        for input_path in input_paths:
            target_path = target_paths[input_path.name]

            rows.append(
                {
                    "dataset": dataset_id,
                    "split": split_name,
                    "input_path": relative_path(input_path),
                    "target_path": relative_path(target_path),
                    "category": "",
                    "illumination_level": ""
                }
            )

    return rows


def build_prefix_replace_rows(
    dataset_id: str,
    dataset_root: Path,
    split_name: str,
    split_config: dict[str, Any],
    pairing_config: dict[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    input_prefix = pairing_config["input_prefix"]
    target_prefix = pairing_config["target_prefix"]

    for source_config in iter_split_sources(split_config):
        input_dir = (dataset_root / source_config["input_dir"]).resolve()
        target_dir = (dataset_root / source_config["target_dir"]).resolve()

        input_paths = list_image_files(input_dir, recursive=False)

        target_paths = {
            target_path.name: target_path
            for target_path in list_image_files(target_dir, recursive=False)
        }

        for input_path in input_paths:
            target_name = target_prefix + input_path.name.removeprefix(input_prefix)
            target_path = target_paths[target_name]

            rows.append(
                {
                    "dataset": dataset_id,
                    "split": split_name,
                    "input_path": relative_path(input_path),
                    "target_path": relative_path(target_path),
                    "category": "",
                    "illumination_level": ""
                }
            )

    return rows


def build_suffix_level_to_gt_rows(
    dataset_id: str,
    dataset_root: Path,
    split_name: str,
    split_config: dict[str, Any],
    pairing_config: dict[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    separator = pairing_config["separator"]

    for source_config in iter_split_sources(split_config):
        input_dir = (dataset_root / source_config["input_dir"]).resolve()
        target_dir = (dataset_root / source_config["target_dir"]).resolve()

        input_paths = list_image_files(input_dir, recursive=False)

        target_paths = {
            target_path.name: target_path
            for target_path in list_image_files(target_dir, recursive=False)
        }

        for input_path in input_paths:
            base_stem, level = input_path.stem.rsplit(separator, maxsplit=1)
            target_name = f"{base_stem}{input_path.suffix}"
            target_path = target_paths[target_name]

            rows.append(
                {
                    "dataset": dataset_id,
                    "split": split_name,
                    "input_path": relative_path(input_path),
                    "target_path": relative_path(target_path),
                    "category": "",
                    "illumination_level": str(int(level))
                }
            )

    return rows


PAIRING_BUILDERS = {
    "category_dirs": build_category_dirs_rows,
    "same_filename": build_same_filename_rows,
    "prefix_replace": build_prefix_replace_rows,
    "suffix_level_to_gt": build_suffix_level_to_gt_rows
}


def build_split_rows(
    dataset_id: str,
    dataset_root: Path,
    split_name: str,
    split_config: dict[str, Any],
    pairing_config: dict[str, Any]
) -> list[dict[str, str]]:
    builder = PAIRING_BUILDERS[pairing_config["strategy"]]

    return builder(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        split_name=split_name,
        split_config=split_config,
        pairing_config=pairing_config
    )


def apply_validation_split(
    rows: list[dict[str, str]],
    validation_config: dict[str, Any]
) -> list[dict[str, str]]:
    from_split = validation_config["from_split"]
    fraction = validation_config["fraction"]
    seed = validation_config["seed"]

    source_indices = [
        index
        for index, row in enumerate(rows)
        if row["split"] == from_split
    ]

    validation_size = math.ceil(len(source_indices) * fraction)

    rng = random.Random(seed)
    validation_indices = set(rng.sample(source_indices, validation_size))

    updated_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        updated_row = dict(row)

        if index in validation_indices:
            updated_row["split"] = "validation"

        updated_rows.append(updated_row)

    return updated_rows


def sort_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            SPLIT_ORDER.get(row["split"], 99),
            row["input_path"]
        )
    )


def build_dataset_rows(
    dataset_id: str,
    dataset_config: dict[str, Any],
    raw_dir: Path
) -> list[dict[str, str]]:
    dataset_root = (raw_dir / dataset_config["root"]).resolve()
    pairing_config = dataset_config["pairing"]

    rows: list[dict[str, str]] = []

    for split_name, split_config in dataset_config["splits"].items():
        rows.extend(
            build_split_rows(
                dataset_id=dataset_id,
                dataset_root=dataset_root,
                split_name=split_name,
                split_config=split_config,
                pairing_config=pairing_config
            )
        )

    if "validation" in dataset_config:
        rows = apply_validation_split(
            rows=rows,
            validation_config=dataset_config["validation"]
        )

    return sort_manifest_rows(rows)


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    raw_dir = Path(config["manifests"]["raw_dir"]).resolve()
    output_dir = Path(config["manifests"]["output_dir"]).resolve()

    for dataset_id, dataset_config in config["datasets"].items():
        rows = build_dataset_rows(
            dataset_id=dataset_id,
            dataset_config=dataset_config,
            raw_dir=raw_dir
        )

        output_path = output_dir / dataset_config["output"]

        write_csv_rows(
            output_path=output_path,
            rows=rows,
            fieldnames=MANIFEST_COLUMNS
        )

        print(f"Wrote {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()