from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from tqdm import tqdm

from src.low_light_enhancement.framework.checkpointing import (
    build_checkpoint_path,
    load_checkpoint,
    save_checkpoint
)
from src.low_light_enhancement.framework.data import build_dataloader
from src.low_light_enhancement.framework.evaluator import Evaluator
from src.low_light_enhancement.framework.logging import (
    JsonlLogger,
    TextLogger,
    format_metrics
)
from src.low_light_enhancement.framework.registry import build_model_wrapper
from src.low_light_enhancement.framework.torch_utils import (
    copy_state_dict,
    count_parameters,
    get_device,
    get_random_state,
    move_batch_to_device,
    set_random_seed,
    set_random_state
)


class ExperimentTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        *,  # next options are keyword-only
        run_index: int,
        logger: JsonlLogger,
        text_logger: TextLogger
    ) -> None:
        self.config = config
        self.run_index = run_index
        self.logger = logger
        self.text_logger = text_logger
        self.device = get_device()

        self.experiment_name = config["experiment"]["name"]

        self.checkpoint_path = build_checkpoint_path(
            self.experiment_name,
            run_index
        )

        self.seed = int(config["experiment"]["seed"])
        self.deterministic = bool(config["training"].get("deterministic", False))

        self.mixed_precision = (
            bool(config["training"]["mixed_precision"])
            and self.device.type == "cuda"
        )

        set_random_seed(
            seed=self.seed,
            deterministic=self.deterministic
        )

        self.wrapper = build_model_wrapper(config["model"]["name"])
        self.model = self.wrapper.build_model(config["model"]).to(self.device)
        self.loss_function = self.wrapper.build_loss(config["loss"]).to(self.device)

        self.optimizer = build_optimizer(
            optimizer_config=config["optimizer"],
            model=self.model
        )

        self.scaler = GradScaler(
            self.device.type,
            enabled=self.mixed_precision
        )

        self.start_epoch = 1
        self.global_step = 0
        self.best_epoch = 0
        self.best_monitor: float | None = None
        self.best_tie_breaker: float | None = None
        self.bad_epochs = 0
        self.training_finished = False
        self.best_model_state_dict = copy_state_dict(self.model)
        self.restored_checkpoint_info: dict[str, Any] | None = None

        self.restore_checkpoint_if_needed()

    def run(self) -> None:
        self.logger.log(
            "run_start",
            {
                "run_index": self.run_index,
                "experiment_name": self.experiment_name,
                "seed": self.seed,
                "device": self.device.type,
                "num_parameters": count_parameters(self.model),
                "checkpoint_path": self.checkpoint_path.as_posix(),
                "resolved_config": self.config
            }
        )

        if self.restored_checkpoint_info is not None:
            self.logger.log(
                "checkpoint_restored",
                self.restored_checkpoint_info
            )

        self.text_logger.info(
            f"Starting run {self.run_index}: {self.experiment_name}"
        )

        if not self.training_finished:
            self.train()
        else:
            self.text_logger.info(
                f"Run {self.run_index}: restored checkpoint already finished"
            )

        if bool(self.config["evaluation"]["run_test"]):
            self.evaluate_test_sets()

        self.logger.log(
            "run_end",
            {
                "run_index": self.run_index,
                "best_epoch": self.best_epoch,
                "best_monitor": self.best_monitor,
                "best_tie_breaker": self.best_tie_breaker,
                "training_finished": self.training_finished
            }
        )

    def train(self) -> None:
        training_config = self.config["training"]
        data_config = self.config["data"]

        train_loader = build_dataloader(
            manifest_path=Path(data_config["train_manifest"]),
            split="train",
            batch_size=int(training_config["batch_size"]),
            num_workers=int(training_config["num_workers"]),
            shuffle=True,
            require_target=True
        )

        validation_loader = build_dataloader(
            manifest_path=Path(data_config["train_manifest"]),
            split="validation",
            batch_size=int(training_config["batch_size"]),
            num_workers=int(training_config["num_workers"]),
            shuffle=False,
            require_target=True
        )

        epochs = int(training_config["epochs"])
        validation_rate = int(training_config["validation_rate"])

        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)

            self.logger.log(
                "epoch_metrics",
                {
                    "run_index": self.run_index,
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "metrics": {
                        "train_loss": train_loss
                    }
                }
            )

            self.text_logger.info(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f}"
            )

            if epoch % validation_rate == 0:
                validation_metrics = self.evaluate_validation(
                    dataloader=validation_loader,
                    epoch=epoch
                )

                is_best = self.update_best_model(
                    epoch=epoch,
                    validation_metrics=validation_metrics
                )

                self.logger.log(
                    "validation_metrics",
                    {
                        "run_index": self.run_index,
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "metrics": validation_metrics,
                        "is_best": is_best,
                        "best_epoch": self.best_epoch,
                        "bad_epochs": self.bad_epochs
                    }
                )

                self.text_logger.info(
                    f"Validation epoch {epoch} | "
                    f"{format_metrics(validation_metrics)} | "
                    f"best_epoch={self.best_epoch} | "
                    f"bad_epochs={self.bad_epochs}"
                )

                if self.should_stop_early():
                    self.training_finished = True
                    self.save_checkpoint(epoch)

                    self.logger.log(
                        "early_stopping",
                        {
                            "run_index": self.run_index,
                            "epoch": epoch,
                            "best_epoch": self.best_epoch,
                            "bad_epochs": self.bad_epochs
                        }
                    )

                    self.text_logger.info("Early stopping triggered")
                    break

            self.save_checkpoint(epoch)
        else:
            self.training_finished = True
            self.save_checkpoint(epochs)

    def train_epoch(
        self,
        train_loader: Any,
        epoch: int
    ) -> float:
        self.model.train()
        losses: list[float] = []

        for batch in tqdm(train_loader, desc=f"train:{epoch}", unit="batch"):
            batch = move_batch_to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=self.mixed_precision
            ):
                output = self.wrapper.forward(self.model, batch)
                prediction = self.wrapper.get_prediction(output)
                loss = self.loss_function(prediction, batch["target"])

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(float(loss.detach().cpu().item()))
            self.global_step += 1

        return float(sum(losses) / len(losses))

    def evaluate_validation(
        self,
        dataloader: Any,
        epoch: int
    ) -> dict[str, float]:
        evaluator = Evaluator(
            model=self.model,
            wrapper=self.wrapper,
            device=self.device,
            logger=None,
            text_logger=None,
            loss_function=self.loss_function
        )

        return evaluator.evaluate(
            dataloader=dataloader,
            event_prefix="validation",
            run_index=self.run_index,
            dataset="validation",
            epoch=epoch,
            log_samples=False,
            include_no_reference=False,
            include_diagnostics=False
        )

    def evaluate_test_sets(self) -> None:
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)

        self.text_logger.info("Evaluating best model on test manifests")

        evaluator = Evaluator(
            model=self.model,
            wrapper=self.wrapper,
            device=self.device,
            logger=self.logger,
            text_logger=self.text_logger,
            loss_function=self.loss_function
        )

        training_config = self.config["training"]

        for test_config in self.config["data"].get("test_manifests", []):
            test_manifest = Path(test_config["manifest"])

            test_loader = build_dataloader(
                manifest_path=test_manifest,
                split="test",
                batch_size=int(training_config["batch_size"]),
                num_workers=int(training_config["num_workers"]),
                shuffle=False,
                require_target=False
            )

            dataset = test_loader.dataset.dataset_name()  # type: ignore[attr-defined]

            evaluator.evaluate(
                dataloader=test_loader,
                event_prefix="test",
                run_index=self.run_index,
                dataset=dataset,
                epoch=self.best_epoch,
                group_by=test_config.get("group_by"),
                log_samples=True
            )

    def update_best_model(
        self,
        *,
        epoch: int,
        validation_metrics: dict[str, float]
    ) -> bool:
        early_stopping_config = self.config["early_stopping"]
        monitor_name = early_stopping_config["monitor"]
        tie_breaker_name = early_stopping_config["tie_breaker"]

        monitor_value = validation_metrics[monitor_name]
        tie_breaker_value = validation_metrics[tie_breaker_name]

        is_best = is_improvement(
            value=monitor_value,
            best_value=self.best_monitor,
            mode=early_stopping_config["mode_monitor"],
            min_delta=float(early_stopping_config["min_delta"])
        )

        is_tie_breaker_best = is_tie_breaker_improvement(
            value=monitor_value,
            best_value=self.best_monitor,
            tie_breaker_value=tie_breaker_value,
            best_tie_breaker_value=self.best_tie_breaker,
            monitor_min_delta=float(early_stopping_config["min_delta"]),
            tie_breaker_mode=early_stopping_config["mode_tie_breaker"]
        )

        if is_best or is_tie_breaker_best:
            self.best_monitor = monitor_value
            self.best_tie_breaker = tie_breaker_value
            self.best_epoch = epoch
            self.bad_epochs = 0
            self.best_model_state_dict = copy_state_dict(self.model)

            return True

        self.bad_epochs += 1
        return False

    def should_stop_early(self) -> bool:
        patience = int(self.config["early_stopping"]["patience"])
        return self.bad_epochs >= patience

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            "current_model_state_dict": self.model.state_dict(),
            "best_model_state_dict": self.best_model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_state": {
                "epoch": epoch,
                "global_step": self.global_step,
                "training_finished": self.training_finished
            },
            "early_stopping_state": {
                "best_epoch": self.best_epoch,
                "best_monitor": self.best_monitor,
                "best_tie_breaker": self.best_tie_breaker,
                "bad_epochs": self.bad_epochs
            },
            "random_state": get_random_state(),
            "resolved_config": self.config
        }

        save_checkpoint(self.checkpoint_path, checkpoint)

    def restore_checkpoint_if_needed(self) -> None:
        restore_checkpoint = self.config["experiment"].get("restore_checkpoint")

        if not restore_checkpoint:
            return

        checkpoint = load_checkpoint(
            Path(restore_checkpoint),
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint["current_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        training_state = checkpoint["training_state"]
        early_stopping_state = checkpoint["early_stopping_state"]

        self.global_step = int(training_state["global_step"])
        self.start_epoch = int(training_state["epoch"]) + 1
        self.training_finished = bool(training_state["training_finished"])

        self.best_epoch = int(early_stopping_state["best_epoch"])
        self.best_monitor = early_stopping_state["best_monitor"]
        self.best_tie_breaker = early_stopping_state["best_tie_breaker"]
        self.bad_epochs = int(early_stopping_state["bad_epochs"])
        self.best_model_state_dict = checkpoint["best_model_state_dict"]

        set_random_state(checkpoint["random_state"])

        self.restored_checkpoint_info = {
            "run_index": self.run_index,
            "restore_checkpoint": str(restore_checkpoint),
            "start_epoch": self.start_epoch,
            "training_finished": self.training_finished
        }


def build_optimizer(
    optimizer_config: dict[str, Any],
    model: nn.Module
) -> Optimizer:
    lr = float(optimizer_config["lr"])
    weight_decay = float(optimizer_config["weight_decay"])

    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


def is_improvement(
    *,
    value: float,
    best_value: float | None,
    mode: str,
    min_delta: float
) -> bool:
    if best_value is None:
        return True

    if mode == "max":
        return value > best_value + min_delta

    if mode == "min":
        return value < best_value - min_delta

    raise ValueError(f"Unsupported mode: {mode!r}.")


def is_tie_breaker_improvement(
    *,
    value: float,
    best_value: float | None,
    tie_breaker_value: float,
    best_tie_breaker_value: float | None,
    monitor_min_delta: float,
    tie_breaker_mode: str
) -> bool:
    if best_value is None or best_tie_breaker_value is None:
        return False

    if abs(value - best_value) > monitor_min_delta:
        return False

    if tie_breaker_mode == "max":
        return tie_breaker_value > best_tie_breaker_value

    if tie_breaker_mode == "min":
        return tie_breaker_value < best_tie_breaker_value

    raise ValueError(f"Unsupported mode: {tie_breaker_mode!r}.")