from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.low_light_enhancement.parsing.failure_candidates import (
    run_failure_candidates_parsing
)
from src.low_light_enhancement.parsing.final_eval import (
    run_final_eval_parsing
)
from src.low_light_enhancement.parsing.multi_seed import (
    run_multi_seed_parsing
)
from src.low_light_enhancement.parsing.screening import (
    run_screening_parsing
)


# Each parsing function receives the logs directory and the output file name
ParsingFunction = Callable[[Path, str], None]

PARSING_MODES: dict[str, ParsingFunction] = {
    "failure_candidates": run_failure_candidates_parsing,
    "final_eval": run_final_eval_parsing,
    "multi_seed": run_multi_seed_parsing,
    "screening": run_screening_parsing
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse JSONL experiment logs and export parsed artifacts."
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
        "--output-name",
        required=True,
        help="Name of the output file written inside parsed_logs/."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsing_function = PARSING_MODES[args.parsing_mode]

    parsing_function(args.logs_dir, args.output_name)


if __name__ == "__main__":
    main()