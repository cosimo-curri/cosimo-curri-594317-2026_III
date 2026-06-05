from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.low_light_enhancement.framework.logging import (
    JsonlLogger,
    TextLogger,
    format_metrics
)
from src.low_light_enhancement.framework.metrics import (
    MetricComputer,
    aggregate_metrics
)
from src.low_light_enhancement.framework.torch_utils import move_batch_to_device


def build_progress_description(
    event_prefix: str,
    dataset: str,
    epoch: int
) -> str:
    if event_prefix == "validation":
        return f"validation epoch {epoch}"

    if event_prefix == "test":
        return f"test {dataset}"

    if event_prefix == dataset:
        return event_prefix

    return f"{event_prefix}:{dataset}"


def group_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))

    return (1, value)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        wrapper: Any,
        device: torch.device,
        logger: JsonlLogger | None = None,
        text_logger: TextLogger | None = None,
        loss_function: nn.Module | None = None
    ) -> None:
        self.model = model
        self.wrapper = wrapper
        self.device = device
        self.logger = logger
        self.text_logger = text_logger
        self.loss_function = loss_function
        self.metric_computer = MetricComputer(device=device)

    def evaluate(
        self,
        dataloader: DataLoader[dict[str, Any]],
        *,  # next options are keyword-only
        event_prefix: str,
        run_index: int,
        dataset: str,
        epoch: int,
        group_by: str | None = None,
        log_samples: bool = False,
        include_no_reference: bool = True,
        include_diagnostics: bool = True
    ) -> dict[str, float]:
        self.model.eval()

        all_sample_metrics: list[dict[str, float]] = []
        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        global_sample_index = 0
        loss_sum = 0.0
        loss_count = 0

        with torch.no_grad():
            progress = tqdm(
                dataloader,
                desc=build_progress_description(
                    event_prefix=event_prefix,
                    dataset=dataset,
                    epoch=epoch
                ),
                unit="batch",
                file=sys.stdout
            )

            for batch in progress:
                batch = move_batch_to_device(batch, self.device)
                output = self.wrapper.forward(self.model, batch)
                prediction = self.wrapper.get_prediction(output)
                target = batch["target"]

                if self.loss_function is not None and target is not None:
                    loss = self.loss_function(prediction, target)
                    batch_size = prediction.shape[0]
                    loss_sum += float(loss.detach().cpu().item()) * batch_size
                    loss_count += batch_size

                sample_metrics = self.metric_computer.compute_samples(
                    prediction=prediction,
                    target=target,
                    include_no_reference=include_no_reference,
                    include_diagnostics=include_diagnostics
                )

                all_sample_metrics.extend(sample_metrics)

                for sample_index, metrics in enumerate(sample_metrics):
                    if group_by is not None:
                        group_values = batch["metadata"].get(group_by, [""])
                        group_value = group_values[sample_index]
                        groups[group_value].append(metrics)

                    if log_samples:
                        self.log_sample_metrics(
                            event_prefix=event_prefix,
                            run_index=run_index,
                            dataset=dataset,
                            epoch=epoch,
                            batch=batch,
                            sample_index=sample_index,
                            global_sample_index=global_sample_index,
                            metrics=metrics
                        )

                    global_sample_index += 1

        means, stds = aggregate_metrics(all_sample_metrics)

        if loss_count > 0:
            means["loss"] = loss_sum / loss_count

        if self.text_logger is not None:
            self.text_logger.info(
                f"{event_prefix.capitalize()} {dataset} | "
                f"{format_metrics(means)}"
            )

        if self.logger is not None:
            self.logger.log(
                f"{event_prefix}_dataset_metrics",
                {
                    "run_index": run_index,
                    "dataset": dataset,
                    "epoch": epoch,
                    "num_samples": len(all_sample_metrics),
                    "metrics_mean": means,
                    "metrics_std": stds
                }
            )

        if group_by is not None:
            self.log_group_metrics(
                event_prefix=event_prefix,
                run_index=run_index,
                dataset=dataset,
                epoch=epoch,
                group_by=group_by,
                groups=groups
            )

        return means

    def log_sample_metrics(
        self,
        *,
        event_prefix: str,
        run_index: int,
        dataset: str,
        epoch: int,
        batch: dict[str, Any],
        sample_index: int,
        global_sample_index: int,
        metrics: dict[str, float]
    ) -> None:
        if self.logger is None:
            return

        metadata = {
            key: values[sample_index]
            for key, values in batch["metadata"].items()
        }

        self.logger.log(
            f"{event_prefix}_sample_metrics",
            {
                "run_index": run_index,
                "dataset": dataset,
                "epoch": epoch,
                "sample_index": global_sample_index,
                "input_path": batch["input_path"][sample_index],
                "target_path": batch["target_path"][sample_index],
                "metadata": metadata,
                "metrics": metrics
            }
        )

    def log_group_metrics(
        self,
        *,
        event_prefix: str,
        run_index: int,
        dataset: str,
        epoch: int,
        group_by: str,
        groups: dict[str, list[dict[str, float]]]
    ) -> None:
        for group_value, sample_metrics in sorted(
            groups.items(),
            key=lambda item: group_sort_key(item[0])
        ):
            means, stds = aggregate_metrics(sample_metrics)

            if self.text_logger is not None:
                self.text_logger.info(
                    f"{event_prefix.capitalize()} {dataset} | "
                    f"{group_by}={group_value} | {format_metrics(means)}"
                )

            if self.logger is not None:
                self.logger.log(
                    f"{event_prefix}_group_metrics",
                    {
                        "run_index": run_index,
                        "dataset": dataset,
                        "epoch": epoch,
                        "group_by": group_by,
                        "group_value": group_value,
                        "num_samples": len(sample_metrics),
                        "metrics_mean": means,
                        "metrics_std": stds
                    }
                )