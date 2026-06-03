from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import add_project_root_to_path


add_project_root_to_path()  # allows imports from src


from src.low_light_enhancement.framework.config import (
    build_resolved_configs,
    load_experiment_config
)
from src.low_light_enhancement.framework.logging import JsonlLogger, TextLogger
from src.low_light_enhancement.framework.trainer import ExperimentTrainer


def print_ascii_art() -> None:
    print(
        """\
 _      _      _____  ______
| |    | |    |_   _||  ____|
| |    | |      | |  | |__
| |    | |      | |  |  __|
| |____| |____ _| |_ | |____
|______|______|_____||______|

Low-Light Image Enhancement with Cross-Dataset Generalization
"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start experiments from a YAML configuration file."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_ascii_art()

    config = load_experiment_config(args.config)
    resolved_configs = build_resolved_configs(config)
    experiment_name = config["experiment"]["name"]
    log_dir = Path("logs") / experiment_name
    text_logger = TextLogger()

    text_logger.info(
        f"Starting {len(resolved_configs)} run(s) for experiment: {experiment_name}"
    )

    for run_index, resolved_config in enumerate(resolved_configs, start=1):
        log_path = log_dir / f"{run_index}.jsonl"
        logger = JsonlLogger(log_path)

        try:
            trainer = ExperimentTrainer(
                config=resolved_config,
                run_index=run_index,
                logger=logger,
                text_logger=text_logger
            )

            trainer.run()
        finally:
            logger.close()

    text_logger.info("All experiments completed")


if __name__ == "__main__":
    main()