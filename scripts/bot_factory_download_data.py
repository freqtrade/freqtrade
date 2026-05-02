#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.data_quality import (
    check_ohlcv_parquet,
    default_quality_output_path,
    ohlcv_data_path,
    write_quality_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe wrapper for freqtrade download-data.")
    parser.add_argument("--config", default="user_data/config.json")
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--timeframes", nargs="+", default=["5m"])
    parser.add_argument("--timerange", default=None)
    parser.add_argument("--trading-mode", choices=["spot", "margin", "futures"], default=None)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--userdir", default="user_data")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--skip-data-quality-check",
        action="store_true",
        help="Skip the post-download OHLCV parquet quality check.",
    )
    parser.add_argument(
        "--quality-output",
        default=None,
        help="Optional JSON path for the OHLCV quality report.",
    )
    parser.add_argument(
        "--enable-freqai",
        action="store_true",
        help="Use FreqAI settings from config. Disabled by default for Phase 1 OHLCV checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_file(args.config, "config")

    overlay_path = None if args.enable_freqai else _write_freqai_disabled_overlay()
    config_args = ["-c", args.config]
    if overlay_path:
        config_args.extend(["-c", str(overlay_path)])

    cmd = [
        args.python,
        "-m",
        "freqtrade_ext.bot_factory.freqtrade_cli",
        "download-data",
        *config_args,
        "--userdir",
        args.userdir,
        "--pairs",
        *args.pairs,
        "--timeframes",
        *args.timeframes,
        "--data-format-ohlcv",
        "parquet",
    ]
    if args.timerange:
        cmd.extend(["--timerange", args.timerange])
    if args.trading_mode:
        cmd.extend(["--trading-mode", args.trading_mode])
    if args.datadir:
        cmd.extend(["--datadir", args.datadir])

    print("Running:", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, text=True)
        if completed.returncode != 0:
            return int(completed.returncode)
        if args.skip_data_quality_check:
            return 0
        return _run_quality_checks(args)
    finally:
        if overlay_path:
            overlay_path.unlink(missing_ok=True)


def _require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise SystemExit(f"{label} file not found: {path}")


def _write_freqai_disabled_overlay() -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="bot_factory_freqai_disabled_", delete=False
    ) as handle:
        json.dump({"freqai": {"enabled": False}}, handle)
        return Path(handle.name)


def _run_quality_checks(args: argparse.Namespace) -> int:
    reports = []
    for pair in args.pairs:
        for timeframe in args.timeframes:
            path = ohlcv_data_path(
                config_path=Path(args.config),
                userdir=Path(args.userdir),
                pair=pair,
                timeframe=timeframe,
                trading_mode=args.trading_mode,
                datadir=Path(args.datadir) if args.datadir else None,
            )
            reports.append(check_ohlcv_parquet(path, timeframe))

    output = Path(args.quality_output) if args.quality_output else default_quality_output_path()
    write_quality_reports(reports, output)
    print(f"OHLCV quality report written: {output}")

    failed = [report for report in reports if not report.ok]
    if failed:
        for report in failed:
            print(report.to_json())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
