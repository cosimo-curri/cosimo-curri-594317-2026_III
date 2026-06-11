from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.low_light_enhancement.framework.io import relative_path, write_json
from src.low_light_enhancement.parsing.common import (
    ParsedRun,
    build_json_path,
    compute_std,
    load_runs
)


CANDIDATES_PER_FAILURE_TYPE = 10

SAMPLE_METRICS = [
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

PERCENTILE_METRICS = [
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

FAILURE_TYPES = [
    "residual_darkness",
    "over_exposure",
    "color_cast",
    "over_smoothing",
    "residual_noise",
    "halo_artifacts",
    "hallucinated_details",
    "good_enhancement"
]

FAILURE_SCORE_SPECS: dict[str, list[tuple[str, str, float]]] = {
    "residual_darkness": [
        ("dark_pixel_ratio", "high", 0.6),
        ("mean_luminance", "low", 0.4)
    ],
    "over_exposure": [
        ("bright_clipping_ratio", "high", 0.7),
        ("mean_luminance", "high", 0.3)
    ],
    "color_cast": [
        ("rgb_channel_imbalance", "high", 1.0)
    ],
    "over_smoothing": [
        ("laplacian_sharpness", "low", 0.6),
        ("luminance_std", "low", 0.4)
    ],
    "residual_noise": [
        ("niqe", "high", 0.35),
        ("brisque", "high", 0.35),
        ("laplacian_sharpness", "high", 0.30)
    ],
    "halo_artifacts": [
        ("bright_clipping_ratio", "high", 0.35),
        ("laplacian_sharpness", "high", 0.30),
        ("brisque", "high", 0.20),
        ("niqe", "high", 0.15)
    ],
    "hallucinated_details_paired": [
        ("ssim", "low", 0.4),
        ("psnr", "low", 0.3),
        ("mae", "high", 0.3)
    ],
    "hallucinated_details_unpaired": [
        ("niqe", "high", 0.35),
        ("brisque", "high", 0.35),
        ("laplacian_sharpness", "anomaly", 0.30)
    ],
    "good_enhancement": [
        ("ssim", "high", 0.25),
        ("psnr", "high", 0.20),
        ("mae", "low", 0.20),
        ("niqe", "low", 0.10),
        ("brisque", "low", 0.10),
        ("dark_pixel_ratio", "low", 0.05),
        ("bright_clipping_ratio", "low", 0.05),
        ("rgb_channel_imbalance", "low", 0.05)
    ]
}


# Dataclasses keep simple data containers concise (no __init__, etc.)
@dataclass
class SampleRunRecord:
    dataset: str
    input_path: str
    target_path: str
    metadata: dict[str, Any]
    metrics: dict[str, float]


@dataclass
class AggregatedSample:
    dataset: str
    input_path: str
    target_path: str
    metadata: dict[str, Any]
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]


def build_sample_run_records(run: ParsedRun) -> list[SampleRunRecord]:
    records = []

    for event in run.events_by_name.get("test_sample_metrics", []):
        records.append(
            SampleRunRecord(
                dataset=event["dataset"],
                input_path=event["input_path"],
                target_path=event.get("target_path", ""),
                metadata=event.get("metadata", {}),
                metrics=event["metrics"]
            )
        )

    if not records:
        raise RuntimeError(
            f"No test sample metrics found in {run.log_path}."
        )

    return records


def group_sample_run_records(
    records: list[SampleRunRecord]
) -> dict[tuple[str, str], list[SampleRunRecord]]:
    grouped_records: dict[tuple[str, str], list[SampleRunRecord]] = (
        defaultdict(list)
    )

    for record in records:
        key = (record.dataset, record.input_path)
        grouped_records[key].append(record)

    return dict(grouped_records)


def metric_values(
    records: list[SampleRunRecord],
    metric_name: str
) -> list[float]:
    return [
        float(record.metrics[metric_name])
        for record in records
        if metric_name in record.metrics
    ]


def build_aggregated_sample(
    records: list[SampleRunRecord]
) -> AggregatedSample:
    first_record = records[0]
    metrics_mean = {}
    metrics_std = {}

    for metric_name in SAMPLE_METRICS:
        values = metric_values(records, metric_name)

        if values:
            metrics_mean[metric_name] = mean(values)
            metrics_std[metric_name] = compute_std(values)

    return AggregatedSample(
        dataset=first_record.dataset,
        input_path=first_record.input_path,
        target_path=first_record.target_path,
        metadata=first_record.metadata,
        metrics_mean=metrics_mean,
        metrics_std=metrics_std
    )


def build_aggregated_samples(
    records: list[SampleRunRecord]
) -> list[AggregatedSample]:
    grouped_records = group_sample_run_records(records)

    return [
        build_aggregated_sample(records)
        for records in grouped_records.values()
    ]


def group_samples_by_dataset(
    samples: list[AggregatedSample]
) -> dict[str, list[AggregatedSample]]:
    grouped_samples: dict[str, list[AggregatedSample]] = defaultdict(list)

    for sample in samples:
        grouped_samples[sample.dataset].append(sample)

    return dict(grouped_samples)


def percentile_scores(values: list[float]) -> dict[float, float]:
    if not values:
        return {}

    if len(values) == 1 or min(values) == max(values):
        return {value: 0.5 for value in values}

    sorted_values = sorted(values)
    score_by_value = {}

    for value in set(sorted_values):
        ranks = [
            index
            for index, sorted_value in enumerate(sorted_values)
            if sorted_value == value
        ]

        score_by_value[value] = mean(ranks) / (len(sorted_values) - 1)

    return score_by_value


def build_sample_percentile_scores(
    samples: list[AggregatedSample]
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {
        sample.input_path: {}
        for sample in samples
    }

    for metric_name in PERCENTILE_METRICS:
        values = [
            float(sample.metrics_mean[metric_name])
            for sample in samples
            if metric_name in sample.metrics_mean
        ]

        score_by_value = percentile_scores(values)

        for sample in samples:
            if metric_name not in sample.metrics_mean:
                continue

            value = float(sample.metrics_mean[metric_name])
            scores[sample.input_path][metric_name] = score_by_value[value]

    return scores


def build_all_percentile_scores(
    samples: list[AggregatedSample]
) -> dict[str, dict[str, float]]:
    all_scores: dict[str, dict[str, float]] = {}

    for dataset_samples in group_samples_by_dataset(samples).values():
        all_scores.update(build_sample_percentile_scores(dataset_samples))

    return all_scores


def is_paired_sample(sample: AggregatedSample) -> bool:
    return bool(sample.target_path) and all(
        metric_name in sample.metrics_mean
        for metric_name in ["ssim", "psnr", "mae"]
    )


def score_component(
    sample_scores: dict[str, float],
    metric_name: str,
    direction: str
) -> float | None:
    if metric_name not in sample_scores:
        return None

    high_score = sample_scores[metric_name]

    if direction == "high":
        return high_score

    if direction == "low":
        return 1.0 - high_score

    if direction == "anomaly":
        return abs(high_score - 0.5) * 2.0

    raise ValueError(f"Unsupported score direction: {direction!r}.")


def score_spec_for(
    failure_type: str,
    sample: AggregatedSample
) -> list[tuple[str, str, float]] | None:
    if failure_type == "hallucinated_details":
        if is_paired_sample(sample):
            return FAILURE_SCORE_SPECS["hallucinated_details_paired"]

        return FAILURE_SCORE_SPECS["hallucinated_details_unpaired"]

    if failure_type == "good_enhancement" and not is_paired_sample(sample):
        return None

    return FAILURE_SCORE_SPECS[failure_type]


def candidate_score(
    sample: AggregatedSample,
    failure_type: str,
    all_scores: dict[str, dict[str, float]]
) -> tuple[float, dict[str, float]] | None:
    spec = score_spec_for(failure_type, sample)

    if spec is None:
        return None

    sample_scores = all_scores[sample.input_path]
    weighted_sum = 0.0
    weight_sum = 0.0
    components = {}

    for metric_name, direction, weight in spec:
        component_score = score_component(
            sample_scores,
            metric_name,
            direction
        )

        if component_score is None:
            return None

        component_name = f"{direction}_{metric_name}"
        components[component_name] = component_score
        weighted_sum += weight * component_score
        weight_sum += weight

    if weight_sum == 0.0:
        return None

    return weighted_sum / weight_sum, components


def metadata_value(sample: AggregatedSample, key: str) -> Any:
    return sample.metadata.get(key, "")


def mean_metrics_for_json(sample: AggregatedSample) -> dict[str, float]:
    return {
        f"mean_{metric_name}": value
        for metric_name, value in sorted(sample.metrics_mean.items())
    }


def std_metrics_for_json(sample: AggregatedSample) -> dict[str, float]:
    return {
        f"std_{metric_name}": value
        for metric_name, value in sorted(sample.metrics_std.items())
    }


def build_candidate_json(
    *,  # next options are keyword-only
    failure_type: str,
    rank: int,
    sample: AggregatedSample,
    score: float,
    components: dict[str, float]
) -> dict[str, Any]:
    return {
        "failure_type": failure_type,
        "rank": rank,
        "dataset": sample.dataset,
        "input_path": sample.input_path,
        "target_path": sample.target_path,
        "category": metadata_value(sample, "category"),
        "illumination_level": metadata_value(sample, "illumination_level"),
        "candidate_score": score,
        "score_components": components,
        "metrics_mean": mean_metrics_for_json(sample),
        "metrics_std": std_metrics_for_json(sample)
    }


def sort_scored_samples(
    scored_samples: list[tuple[float, dict[str, float], AggregatedSample]]
) -> list[tuple[float, dict[str, float], AggregatedSample]]:
    return sorted(
        scored_samples,
        key=lambda item: (
            -item[0],
            item[2].dataset,
            item[2].input_path
        )
    )


def select_failure_candidates(
    samples: list[AggregatedSample],
    all_scores: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    candidates = []

    for failure_type in FAILURE_TYPES:
        scored_samples = []

        for sample in samples:
            score = candidate_score(sample, failure_type, all_scores)

            if score is None:
                continue

            candidate_score_value, components = score

            scored_samples.append(
                (candidate_score_value, components, sample)
            )

        scored_samples = sort_scored_samples(scored_samples)

        for rank, (score, components, sample) in enumerate(
            scored_samples[:CANDIDATES_PER_FAILURE_TYPE],
            start=1
        ):
            candidates.append(
                build_candidate_json(
                    failure_type=failure_type,
                    rank=rank,
                    sample=sample,
                    score=score,
                    components=components
                )
            )

    return candidates


def run_failure_candidates_parsing(logs_dir: Path, output_name: str) -> None:
    runs = load_runs(logs_dir)

    run_records = [
        record
        for run in runs
        for record in build_sample_run_records(run)
    ]

    samples = build_aggregated_samples(run_records)
    all_scores = build_all_percentile_scores(samples)

    candidates = select_failure_candidates(
        samples=samples,
        all_scores=all_scores
    )

    json_path = build_json_path(output_name)

    write_json(json_path, candidates)

    print(f"Parsed {len(runs)} final evaluation run(s)")
    print(f"Aggregated {len(samples)} sample(s)")
    print(f"Selected {len(candidates)} qualitative candidate(s)")
    print(f"JSON: {relative_path(json_path)}")