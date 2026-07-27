"""Safe migration utilities; this is not yet the production pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence, TextIO

from .comparison import compare_output_trees
from .config import ConfigError, default_config_path, load_config
from .layout import LEGACY_OUTPUT_DIRECTORY_NAMES, ProjectLayout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migration_tools.py",
        description="Safe tools for the ProcessSqlData-TS migration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-config",
        help="Validate the local TOML file without connecting to PostgreSQL.",
    )
    validate.add_argument(
        "--config",
        type=Path,
        help="Defaults to config.local.toml beside the repository entrypoint.",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Compare a frozen legacy output tree with migrated output.",
    )
    compare.add_argument("--expected", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--relative-tolerance", type=float, default=1e-9)
    compare.add_argument("--absolute-tolerance", type=float, default=1e-12)

    subparsers.add_parser(
        "show-layout",
        help="Print script-anchored legacy output paths without creating them.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    entrypoint: Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    parser = build_parser()
    arguments = parser.parse_args(argv)
    effective_entrypoint = (
        entrypoint.resolve()
        if entrypoint is not None
        else (Path(__file__).resolve().parents[2] / "ProcessSqlData Final.py")
    )
    layout = ProjectLayout.from_entrypoint(effective_entrypoint)

    if arguments.command == "validate-config":
        config_path = arguments.config or default_config_path(
            layout.repository_root
        )
        try:
            config = load_config(config_path)
        except ConfigError as exc:
            print(f"Configuration invalid: {exc}", file=error_output)
            return 2
        print(
            f"Configuration valid: {config.source_path} "
            "(database session will be forced read-only)",
            file=output,
        )
        return 0

    if arguments.command == "compare":
        result = compare_output_trees(
            arguments.expected,
            arguments.actual,
            relative_tolerance=arguments.relative_tolerance,
            absolute_tolerance=arguments.absolute_tolerance,
        )
        print(result.summary(), file=output if result.matches else error_output)
        return 0 if result.matches else 1

    if arguments.command == "show-layout":
        for name in LEGACY_OUTPUT_DIRECTORY_NAMES:
            print(f"{name}={layout.output_path(name)}", file=output)
        return 0

    parser.error(f"Unknown command: {arguments.command}")
    return 2
