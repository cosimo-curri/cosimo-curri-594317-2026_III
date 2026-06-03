from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from bootstrap import add_project_root_to_path


add_project_root_to_path()  # allows imports from src


from src.low_light_enhancement.framework.io import (
    load_config,
    read_csv_rows,
    relative_path,
    write_csv_rows
)


SUMMARY_COLUMNS = [
    "dataset",
    "total_samples",
    "median_original_size",
    "input_luminance",
    "dark_pixels",
    "target_luminance"
]

RESAMPLE_FILTER = Image.Resampling.LANCZOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess images from manifest files and compute "
            "dataset-level statistics."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file."
    )

    return parser.parse_args()


def validate_manifest_rows(
    manifest_path: Path,
    rows: list[dict[str, str]]
) -> None:
    errors: list[str] = []
    split_input_paths: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        split_input_paths[row["split"]].append(row["input_path"])

    for split_name, input_paths in split_input_paths.items():
        duplicated_paths = [
            input_path
            for input_path, count in Counter(input_paths).items()
            if count > 1
        ]

        for input_path in duplicated_paths:
            errors.append(
                f"{manifest_path}: duplicated input path {input_path} "
                f"in split {split_name}."
            )

    split_names = sorted(split_input_paths)

    for index, split_name in enumerate(split_names):
        input_paths = set(split_input_paths[split_name])

        for other_split_name in split_names[index + 1:]:
            other_input_paths = set(split_input_paths[other_split_name])
            overlapping_paths = input_paths & other_input_paths

            for input_path in sorted(overlapping_paths):
                errors.append(
                    f"{manifest_path}: input path {input_path} appears in both "
                    f"{split_name} and {other_split_name}."
                )

    if errors:
        shown_errors = "\n".join(errors[:20])

        raise ValueError(
            f"Validation failed for {manifest_path}\n{shown_errors}."
        )


def infer_dataset_root(rows: list[dict[str, str]]) -> Path:
    image_paths: list[Path] = []

    for row in rows:
        image_paths.append(Path(row["input_path"]).resolve())

        if row["target_path"]:
            image_paths.append(Path(row["target_path"]).resolve())

    common_parts = Path(*Path(image_paths[0]).parts)

    for image_path in image_paths[1:]:
        while common_parts not in image_path.parents:
            common_parts = common_parts.parent

    return common_parts


def build_output_path(
    source_path: Path,
    dataset_id: str,
    output_dir: Path,
    dataset_root: Path,
    output_format: str
) -> Path:
    relative_source_path = source_path.resolve().relative_to(
        dataset_root.resolve()
    )

    return (output_dir / dataset_id / relative_source_path).with_suffix(
        f".{output_format}"
    )


def read_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)  # fix orientation from EXIF metadata
        return image.convert("RGB")


def compute_image_statistics(
    image: Image.Image,
    dark_threshold: float
) -> dict[str, float]:
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    luminance = (
        0.2126 * image_array[:, :, 0]
        + 0.7152 * image_array[:, :, 1]
        + 0.0722 * image_array[:, :, 2]
    )

    return {
        "width": float(image.width),
        "height": float(image.height),
        "mean_luminance": float(luminance.mean()),
        "dark_pixel_ratio": float((luminance < dark_threshold).mean())
    }


def process_image(
    source_path: Path,
    output_path: Path,
    image_size: tuple[int, int],
    output_format: str,
    dark_threshold: float
) -> dict[str, float]:
    image = read_rgb_image(source_path)

    image_statistics = compute_image_statistics(
        image=image,
        dark_threshold=dark_threshold
    )

    resized_image = image.resize(image_size, RESAMPLE_FILTER)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized_image.save(output_path, format=output_format.upper())

    return image_statistics


def process_image_once(
    source_path: Path,
    dataset_id: str,
    output_dir: Path,
    dataset_root: Path,
    image_size: tuple[int, int],
    output_format: str,
    dark_threshold: float,
    processed_images: dict[tuple[str, str], tuple[Path, dict[str, float]]]
) -> tuple[Path, dict[str, float]]:
    image_key = (
        dataset_id,
        source_path.resolve().as_posix()
    )

    if image_key not in processed_images:
        output_path = build_output_path(
            source_path=source_path,
            dataset_id=dataset_id,
            output_dir=output_dir,
            dataset_root=dataset_root,
            output_format=output_format
        )

        image_statistics = process_image(
            source_path=source_path,
            output_path=output_path,
            image_size=image_size,
            output_format=output_format,
            dark_threshold=dark_threshold
        )

        processed_images[image_key] = (output_path, image_statistics)

    return processed_images[image_key]


def make_empty_dataset_statistics() -> dict[str, Any]:
    return {
        "total_samples": 0,
        "widths": [],
        "heights": [],
        "input_luminance": [],
        "dark_pixels": [],
        "target_luminance": []
    }


def update_dataset_statistics(
    dataset_statistics: dict[str, Any],
    dataset_id: str,
    input_statistics: dict[str, float],
    target_statistics: dict[str, float] | None
) -> None:
    statistics = dataset_statistics[dataset_id]

    statistics["total_samples"] += 1
    statistics["widths"].append(input_statistics["width"])
    statistics["heights"].append(input_statistics["height"])
    statistics["input_luminance"].append(input_statistics["mean_luminance"])
    statistics["dark_pixels"].append(input_statistics["dark_pixel_ratio"])

    if target_statistics is not None:
        statistics["target_luminance"].append(
            target_statistics["mean_luminance"]
        )


def format_dimension(value: float) -> str:
    if value.is_integer():
        return str(int(value))

    return f"{value:.1f}"


def format_median_size(
    widths: list[float],
    heights: list[float]
) -> str:
    median_width = float(median(widths))
    median_height = float(median(heights))

    return (
        f"{format_dimension(median_width)}x"
        f"{format_dimension(median_height)}"
    )


def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_std(values: list[float]) -> float:
    mean_value = compute_mean(values)

    variance = sum(
        (value - mean_value) ** 2
        for value in values
    ) / len(values)

    return math.sqrt(variance)


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "—"

    return f"{compute_mean(values):.3f} ± {compute_std(values):.3f}"


def format_percentage(values: list[float]) -> str:
    if not values:
        return "—"

    return f"{compute_mean(values) * 100.0:.1f}%"


def build_summary_rows(
    dataset_statistics: dict[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for dataset_id, statistics in dataset_statistics.items():
        rows.append(
            {
                "dataset": dataset_id,
                "total_samples": str(statistics["total_samples"]),
                "median_original_size": format_median_size(
                    widths=statistics["widths"],
                    heights=statistics["heights"]
                ),
                "input_luminance": format_mean_std(
                    statistics["input_luminance"]
                ),
                "dark_pixels": format_percentage(
                    statistics["dark_pixels"]
                ),
                "target_luminance": format_mean_std(
                    statistics["target_luminance"]
                )
            }
        )

    return rows


def process_manifest(
    manifest_path: Path,
    output_dir: Path,
    image_size: tuple[int, int],
    output_format: str,
    dark_threshold: float,
    processed_images: dict[tuple[str, str], tuple[Path, dict[str, float]]],
    dataset_statistics: dict[str, Any]
) -> None:
    dataset_id = manifest_path.stem

    rows, manifest_columns = read_csv_rows(manifest_path)
    dataset_root = infer_dataset_root(rows)

    validate_manifest_rows(
        manifest_path=manifest_path,
        rows=rows
    )

    processed_rows: list[dict[str, str]] = []

    for row in tqdm(rows, desc=dataset_id, unit="sample"):
        processed_row = dict(row)

        input_output_path, input_statistics = process_image_once(
            source_path=Path(row["input_path"]),
            dataset_id=dataset_id,
            output_dir=output_dir,
            dataset_root=dataset_root,
            image_size=image_size,
            output_format=output_format,
            dark_threshold=dark_threshold,
            processed_images=processed_images
        )

        processed_row["input_path"] = relative_path(input_output_path)

        target_statistics: dict[str, float] | None = None

        if row["target_path"]:
            target_output_path, target_statistics = process_image_once(
                source_path=Path(row["target_path"]),
                dataset_id=dataset_id,
                output_dir=output_dir,
                dataset_root=dataset_root,
                image_size=image_size,
                output_format=output_format,
                dark_threshold=dark_threshold,
                processed_images=processed_images
            )

            processed_row["target_path"] = relative_path(target_output_path)

        update_dataset_statistics(
            dataset_statistics=dataset_statistics,
            dataset_id=dataset_id,
            input_statistics=input_statistics,
            target_statistics=target_statistics
        )

        processed_rows.append(processed_row)

    output_manifest_path = output_dir / manifest_path.name

    write_csv_rows(
        output_path=output_manifest_path,
        rows=processed_rows,
        fieldnames=manifest_columns
    )

    print(f"Wrote {len(processed_rows)} rows to: {output_manifest_path}.")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    preprocessing_config = config["preprocessing"]

    input_manifest_dir = Path(
        preprocessing_config["input_manifest_dir"]
    ).resolve()

    output_dir = Path(
        preprocessing_config["output_dir"]
    ).resolve()

    image_size = tuple(preprocessing_config["image"]["size"])
    output_format = preprocessing_config["image"]["output_format"]
    dark_threshold = preprocessing_config["statistics"]["dark_threshold"]

    processed_images: dict[tuple[str, str], tuple[Path, dict[str, float]]] = {}

    dataset_statistics: dict[str, Any] = defaultdict(
        make_empty_dataset_statistics
    )

    for manifest_path in sorted(input_manifest_dir.glob("*.csv")):
        process_manifest(
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_size=image_size,
            output_format=output_format,
            dark_threshold=dark_threshold,
            processed_images=processed_images,
            dataset_statistics=dataset_statistics
        )

    summary_rows = build_summary_rows(dataset_statistics)
    summary_path = output_dir / "dataset_summary.csv"

    write_csv_rows(
        output_path=summary_path,
        rows=summary_rows,
        fieldnames=SUMMARY_COLUMNS
    )

    print(f"Wrote dataset summary to: {summary_path}.")


if __name__ == "__main__":
    main()