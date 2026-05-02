from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass
class DataQualityFinding:
    path: str
    rule: str
    severity: str
    message: str


@dataclass
class OHLCVQualityReport:
    path: str
    ok: bool
    rows: int
    columns: list[str]
    start: str | None
    end: str | None
    expected_timeframe: str | None
    expected_interval_seconds: int | None
    duplicate_timestamps: int
    missing_intervals: int
    findings: list[DataQualityFinding]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "rows": self.rows,
            "columns": self.columns,
            "start": self.start,
            "end": self.end,
            "expected_timeframe": self.expected_timeframe,
            "expected_interval_seconds": self.expected_interval_seconds,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_intervals": self.missing_intervals,
            "findings": [asdict(f) for f in self.findings],
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def check_ohlcv_parquet(path: Path, expected_timeframe: str | None = None) -> OHLCVQualityReport:
    findings: list[DataQualityFinding] = []
    interval_seconds = timeframe_to_seconds(expected_timeframe) if expected_timeframe else None

    if not path.is_file():
        findings.append(
            _finding(path, "file_exists", "error", f"OHLCV parquet file does not exist: {path}")
        )
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        findings.append(_finding(path, "read_parquet", "error", f"Could not read parquet: {exc}"))
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    columns = list(dataframe.columns)
    rows = len(dataframe)
    missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        findings.append(
            _finding(
                path,
                "required_columns",
                "error",
                f"Missing required OHLCV columns: {', '.join(missing_columns)}",
            )
        )
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    if rows == 0:
        findings.append(_finding(path, "non_empty", "error", "OHLCV file contains no rows."))
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
    start = _timestamp_to_str(dataframe["date"].min())
    end = _timestamp_to_str(dataframe["date"].max())

    null_dates = int(dataframe["date"].isna().sum())
    if null_dates:
        findings.append(
            _finding(path, "date_not_null", "error", f"Found {null_dates} rows with invalid dates.")
        )

    null_ohlcv = dataframe[list(REQUIRED_OHLCV_COLUMNS)].isna().sum()
    for column, count in null_ohlcv.items():
        if int(count) > 0:
            findings.append(
                _finding(path, f"{column}_not_null", "error", f"Column '{column}' has {int(count)} null values.")
            )

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    if duplicate_timestamps:
        findings.append(
            _finding(
                path,
                "no_duplicate_timestamps",
                "error",
                f"Found {duplicate_timestamps} duplicate OHLCV timestamps.",
            )
        )

    if not dataframe["date"].is_monotonic_increasing:
        findings.append(
            _finding(path, "timestamps_sorted", "error", "OHLCV timestamps are not sorted ascending.")
        )

    _check_numeric_columns(path, dataframe, findings)
    _check_price_bounds(path, dataframe, findings)

    missing_intervals = 0
    if interval_seconds:
        missing_intervals = _check_time_intervals(path, dataframe, interval_seconds, findings)

    ok = not any(finding.severity == "error" for finding in findings)
    return _report(
        path,
        ok,
        rows,
        columns,
        start,
        end,
        expected_timeframe,
        interval_seconds,
        duplicate_timestamps,
        missing_intervals,
        findings,
    )


def timeframe_to_seconds(timeframe: str | None) -> int | None:
    if not timeframe:
        return None
    if len(timeframe) < 2:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    multipliers = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return amount * multipliers[unit]


def write_quality_reports(reports: list[OHLCVQualityReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": all(report.ok for report in reports),
        "reports": [report.to_dict() for report in reports],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_quality_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_ohlcv_quality.json"


def ohlcv_data_path(
    *,
    config_path: Path,
    userdir: Path,
    pair: str,
    timeframe: str,
    trading_mode: str | None = None,
    datadir: Path | None = None,
) -> Path:
    exchange = _exchange_name(config_path)
    mode = trading_mode or _trading_mode(config_path)
    root = datadir if datadir else userdir / "data"
    filename = pair_to_ohlcv_filename(pair, timeframe, mode)
    return root / exchange / mode / filename if mode != "spot" else root / exchange / filename


def pair_to_ohlcv_filename(pair: str, timeframe: str, trading_mode: str | None = None) -> str:
    pair_name = pair.replace("/", "_").replace(":", "_")
    mode = trading_mode or "spot"
    suffix = f"-{timeframe}"
    if mode != "spot":
        suffix = f"{suffix}-{mode}"
    return f"{pair_name}{suffix}.parquet"


def _check_numeric_columns(
    path: Path, dataframe: pd.DataFrame, findings: list[DataQualityFinding]
) -> None:
    for column in REQUIRED_OHLCV_COLUMNS[1:]:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        invalid = int(numeric.isna().sum())
        if invalid:
            findings.append(
                _finding(path, f"{column}_numeric", "error", f"Column '{column}' has {invalid} non-numeric values.")
            )

    for column in PRICE_COLUMNS:
        non_positive = int((pd.to_numeric(dataframe[column], errors="coerce") <= 0).sum())
        if non_positive:
            findings.append(
                _finding(
                    path,
                    f"{column}_positive",
                    "error",
                    f"Column '{column}' has {non_positive} non-positive values.",
                )
            )

    negative_volume = int((pd.to_numeric(dataframe["volume"], errors="coerce") < 0).sum())
    if negative_volume:
        findings.append(
            _finding(path, "volume_non_negative", "error", f"Volume has {negative_volume} negative values.")
        )


def _check_price_bounds(path: Path, dataframe: pd.DataFrame, findings: list[DataQualityFinding]) -> None:
    prices = dataframe[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    high_too_low = int((prices["high"] < prices[["open", "low", "close"]].max(axis=1)).sum())
    low_too_high = int((prices["low"] > prices[["open", "high", "close"]].min(axis=1)).sum())

    if high_too_low:
        findings.append(
            _finding(path, "high_bounds", "error", f"High is below open/low/close in {high_too_low} rows.")
        )
    if low_too_high:
        findings.append(
            _finding(path, "low_bounds", "error", f"Low is above open/high/close in {low_too_high} rows.")
        )


def _check_time_intervals(
    path: Path, dataframe: pd.DataFrame, interval_seconds: int, findings: list[DataQualityFinding]
) -> int:
    dates = dataframe["date"].dropna()
    if len(dates) < 2:
        return 0

    diffs = dates.diff().dropna().dt.total_seconds().astype(int)
    wrong = diffs[diffs != interval_seconds]
    if wrong.empty:
        return 0

    missing = int(((wrong[wrong > interval_seconds] / interval_seconds) - 1).sum())
    findings.append(
        _finding(
            path,
            "expected_time_interval",
            "warning",
            f"Found {len(wrong)} irregular intervals; estimated missing candles: {missing}.",
        )
    )
    return missing


def _exchange_name(config_path: Path) -> str:
    config = _load_json(config_path)
    exchange = config.get("exchange", {})
    name = exchange.get("name") if isinstance(exchange, dict) else None
    if not name:
        raise ValueError(f"Could not determine exchange.name from config: {config_path}")
    return str(name)


def _trading_mode(config_path: Path) -> str:
    config = _load_json(config_path)
    mode = config.get("trading_mode", "spot")
    return str(mode or "spot")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report(
    path: Path,
    ok: bool,
    rows: int,
    columns: list[str],
    start: str | None,
    end: str | None,
    expected_timeframe: str | None,
    expected_interval_seconds: int | None,
    duplicate_timestamps: int,
    missing_intervals: int,
    findings: list[DataQualityFinding],
) -> OHLCVQualityReport:
    return OHLCVQualityReport(
        path=str(path),
        ok=ok,
        rows=rows,
        columns=columns,
        start=start,
        end=end,
        expected_timeframe=expected_timeframe,
        expected_interval_seconds=expected_interval_seconds,
        duplicate_timestamps=duplicate_timestamps,
        missing_intervals=missing_intervals,
        findings=findings,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _finding(path: Path, rule: str, severity: str, message: str) -> DataQualityFinding:
    return DataQualityFinding(path=str(path), rule=rule, severity=severity, message=message)


def _timestamp_to_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return value.isoformat()
