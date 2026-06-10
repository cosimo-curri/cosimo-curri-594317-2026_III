from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.low_light_enhancement.parsing.multi_seed import run_multi_seed_parsing
from src.low_light_enhancement.parsing.screening import run_screening_parsing


# Each parsing function receives the logs directory and the output CSV name
ParsingFunction = Callable[[Path, str], None]

PARSING_MODES: dict[str, ParsingFunction] = {
    "multi_seed": run_multi_seed_parsing,
    "screening": run_screening_parsing
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse JSONL experiment logs and export a CSV file."
    )

    parser.add_argument(
        "--parsing-mode",
        choices=sorted(PARSING_MODES),
        required=True,
        help="Parsing mode to execute."
    )

    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Directory containing JSONL logs."
    )

    parser.add_argument(
        "--csv-name",
        default="parsed_logs.csv",
        help="Name of the CSV file written inside parsed_logs/."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsing_function = PARSING_MODES[args.parsing_mode]

    parsing_function(
        logs_dir=args.logs_dir,
        csv_name=args.csv_name
    )


if __name__ == "__main__":
    main()