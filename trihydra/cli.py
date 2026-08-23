"""Command-line interface for validated, TOML-driven TriHydrA runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from trihydra.batch import run_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trihydra")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run TriHydrA from a TOML file")
    run.add_argument(
        "--config", type=Path, default=Path("trihydra.toml"),
        help="configuration file (default: trihydra.toml)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = run_batch(args.config).manifest
    # An empty run is not a successful assessment: no station was processed
    # and no scientific result can be delivered to the caller/HPC scheduler.
    if manifest.empty:
        return 1
    return 0 if (manifest["status"] == "completed").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
