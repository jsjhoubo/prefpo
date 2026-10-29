"""Command-line entry point for local Lattice research experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Sequence

from prefpo.lattice.config import ExperimentConfig
from prefpo.lattice.runner import LatticeExperiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prefpo.lattice.cli",
        description="Run a local or model-backed Lattice-inspired research experiment.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the run output directory from the experiment config",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--print-result",
        action="store_true",
        help="Print the complete JSON result after writing experiment artifacts",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    result = await LatticeExperiment(config).run()
    if args.print_result:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(
            f"Completed {config.name}: run_id={result.run_id} "
            f"validation_f1={result.construction.best_report.overall.f1:.3f} "
            f"artifacts={result.run_directory}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
