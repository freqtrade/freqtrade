#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.freqai_checks import validate_freqai_strategy_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Bot Factory FreqAI feature, label, and lookahead conventions."
    )
    parser.add_argument("paths", nargs="+", help="Strategy Python files or directories to scan.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path. Defaults to registry/strategies/checks/.",
    )
    parser.add_argument("--no-fail", action="store_true", help="Always exit with status 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_freqai_strategy_paths([Path(path) for path in args.paths])
    output = Path(args.output) if args.output else _default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json(), encoding="utf-8")

    print(report.to_json())
    print(f"FreqAI validation report written: {output}")

    if not report.ok and not args.no_fail:
        return 1
    return 0


def _default_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_freqai_validation.json"


if __name__ == "__main__":
    sys.exit(main())
