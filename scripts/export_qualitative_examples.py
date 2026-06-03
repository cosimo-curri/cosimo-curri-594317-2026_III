from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import add_project_root_to_path


add_project_root_to_path()  # allows imports from src


from src.low_light_enhancement.framework.qualitative import QualitativeExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export selected qualitative examples from a saved checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file."
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to a JSON file with selected input_path values."
    )

    return parser.parse_args()


def build_output_dir(checkpoint_path: Path) -> Path:
    experiment_name = checkpoint_path.parent.name
    run_index = checkpoint_path.stem

    return Path("qualitative") / experiment_name / run_index


def main() -> None:
    args = parse_args()

    output_dir = build_output_dir(args.checkpoint)

    exporter = QualitativeExporter(
        checkpoint_path=args.checkpoint,
        output_dir=output_dir
    )

    exporter.export(args.candidates)


if __name__ == "__main__":
    main()