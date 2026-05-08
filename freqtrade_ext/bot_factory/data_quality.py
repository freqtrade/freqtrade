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


def check_funding_rate_parquet(
    path: Path, expected_timeframe: str | None = None
) -> OHLCVQualityReport:
    findings: list[DataQualityFinding] = []
    interval_seconds = timeframe_to_seconds(expected_timeframe) if expected_timeframe else None

    if not path.is_file():
        findings.append(
            _finding(path, "file_exists", "error", f"Funding-rate parquet file does not exist: {path}")
        )
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        findings.append(_finding(path, "read_parquet", "error", f"Could not read parquet: {exc}"))
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    columns = list(dataframe.columns)
    rows = len(dataframe)
    required_columns = ("date", "open")
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        findings.append(
            _finding(
                path,
                "required_columns",
                "error",
                f"Missing required funding-rate columns: {', '.join(missing_columns)}",
            )
        )
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    if rows == 0:
        findings.append(_finding(path, "non_empty", "error", "Funding-rate file contains no rows."))
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

    rates = pd.to_numeric(dataframe["open"], errors="coerce")
    invalid_rates = int(rates.isna().sum())
    if invalid_rates:
        findings.append(
            _finding(path, "funding_rate_numeric", "error", f"Funding-rate column has {invalid_rates} non-numeric values.")
        )

    extreme_rates = int((rates.abs() > 0.05).sum())
    if extreme_rates:
        findings.append(
            _finding(path, "funding_rate_abs_reasonable", "warning", f"Funding-rate column has {extreme_rates} values with absolute rate above 5%.")
        )

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    if duplicate_timestamps:
        findings.append(
            _finding(
                path,
                "no_duplicate_timestamps",
                "error",
                f"Found {duplicate_timestamps} duplicate funding-rate timestamps.",
            )
        )

    if not dataframe["date"].is_monotonic_increasing:
        findings.append(
            _finding(path, "timestamps_sorted", "error", "Funding-rate timestamps are not sorted ascending.")
        )

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


def check_open_interest_parquet(
    path: Path, expected_timeframe: str | None = None
) -> OHLCVQualityReport:
    findings: list[DataQualityFinding] = []
    interval_seconds = timeframe_to_seconds(expected_timeframe) if expected_timeframe else None

    if not path.is_file():
        findings.append(
            _finding(path, "file_exists", "error", f"Open-interest parquet file does not exist: {path}")
        )
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        findings.append(_finding(path, "read_parquet", "error", f"Could not read parquet: {exc}"))
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    columns = list(dataframe.columns)
    rows = len(dataframe)
    if "date" not in dataframe.columns:
        findings.append(_finding(path, "required_columns", "error", "Missing required open-interest column: date"))
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    interest_column = _open_interest_value_column(dataframe)
    if interest_column is None:
        findings.append(
            _finding(
                path,
                "required_columns",
                "error",
                "Missing open-interest value column; expected one of open_interest, open, close.",
            )
        )
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    if rows == 0:
        findings.append(_finding(path, "non_empty", "error", "Open-interest file contains no rows."))
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

    interest = pd.to_numeric(dataframe[interest_column], errors="coerce")
    invalid_interest = int(interest.isna().sum())
    if invalid_interest:
        findings.append(
            _finding(
                path,
                "open_interest_numeric",
                "error",
                f"Open-interest column has {invalid_interest} non-numeric values.",
            )
        )

    negative_interest = int((interest < 0).sum())
    if negative_interest:
        findings.append(
            _finding(
                path,
                "open_interest_non_negative",
                "error",
                f"Open-interest column has {negative_interest} negative values.",
            )
        )

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    if duplicate_timestamps:
        findings.append(
            _finding(
                path,
                "no_duplicate_timestamps",
                "error",
                f"Found {duplicate_timestamps} duplicate open-interest timestamps.",
            )
        )

    if not dataframe["date"].is_monotonic_increasing:
        findings.append(
            _finding(path, "timestamps_sorted", "error", "Open-interest timestamps are not sorted ascending.")
        )

    missing_intervals = 0
    if interval_seconds:
        missing_intervals = _check_time_intervals(path, dataframe, interval_seconds, findings)

    return _report(
        path,
        not any(f.severity == "error" for f in findings),
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


def check_long_short_ratio_parquet(
    path: Path, expected_timeframe: str | None = None
) -> OHLCVQualityReport:
    findings: list[DataQualityFinding] = []
    interval_seconds = timeframe_to_seconds(expected_timeframe) if expected_timeframe else None

    if not path.is_file():
        findings.append(
            _finding(path, "file_exists", "error", f"Long/short-ratio parquet file does not exist: {path}")
        )
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        findings.append(_finding(path, "read_parquet", "error", f"Could not read parquet: {exc}"))
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    columns = list(dataframe.columns)
    rows = len(dataframe)
    if "date" not in dataframe.columns:
        findings.append(_finding(path, "required_columns", "error", "Missing required long/short-ratio column: date"))
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    ratio_columns = _long_short_ratio_value_columns(dataframe)
    if ratio_columns is None:
        findings.append(
            _finding(
                path,
                "required_columns",
                "error",
                (
                    "Missing long/short-ratio value columns; expected long_account_ratio/buyRatio "
                    "and short_account_ratio/sellRatio."
                ),
            )
        )
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    if rows == 0:
        findings.append(_finding(path, "non_empty", "error", "Long/short-ratio file contains no rows."))
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    long_column, short_column = ratio_columns
    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
    start = _timestamp_to_str(dataframe["date"].min())
    end = _timestamp_to_str(dataframe["date"].max())

    null_dates = int(dataframe["date"].isna().sum())
    if null_dates:
        findings.append(
            _finding(path, "date_not_null", "error", f"Found {null_dates} rows with invalid dates.")
        )

    long_ratio = pd.to_numeric(dataframe[long_column], errors="coerce")
    short_ratio = pd.to_numeric(dataframe[short_column], errors="coerce")
    invalid_long = int(long_ratio.isna().sum())
    invalid_short = int(short_ratio.isna().sum())
    if invalid_long:
        findings.append(
            _finding(
                path,
                "long_account_ratio_numeric",
                "error",
                f"Long-account ratio column has {invalid_long} non-numeric values.",
            )
        )
    if invalid_short:
        findings.append(
            _finding(
                path,
                "short_account_ratio_numeric",
                "error",
                f"Short-account ratio column has {invalid_short} non-numeric values.",
            )
        )

    long_out_of_bounds = int(((long_ratio < 0.0) | (long_ratio > 1.0)).sum())
    short_out_of_bounds = int(((short_ratio < 0.0) | (short_ratio > 1.0)).sum())
    if long_out_of_bounds:
        findings.append(
            _finding(
                path,
                "long_account_ratio_between_zero_and_one",
                "error",
                f"Long-account ratio has {long_out_of_bounds} values outside [0, 1].",
            )
        )
    if short_out_of_bounds:
        findings.append(
            _finding(
                path,
                "short_account_ratio_between_zero_and_one",
                "error",
                f"Short-account ratio has {short_out_of_bounds} values outside [0, 1].",
            )
        )

    ratio_sum_drift = int(((long_ratio + short_ratio - 1.0).abs() > 0.02).sum())
    if ratio_sum_drift:
        findings.append(
            _finding(
                path,
                "long_short_account_ratio_sum",
                "warning",
                f"Long/short account ratios do not sum near 1.0 in {ratio_sum_drift} rows.",
            )
        )

    long_short_column = _long_short_ratio_column(dataframe)
    if long_short_column is not None:
        long_short_ratio = pd.to_numeric(dataframe[long_short_column], errors="coerce")
        invalid_long_short = int(long_short_ratio.isna().sum())
        if invalid_long_short:
            findings.append(
                _finding(
                    path,
                    "long_short_ratio_numeric",
                    "error",
                    f"Long/short ratio column has {invalid_long_short} non-numeric values.",
                )
            )
        negative_long_short = int((long_short_ratio < 0.0).sum())
        if negative_long_short:
            findings.append(
                _finding(
                    path,
                    "long_short_ratio_non_negative",
                    "error",
                    f"Long/short ratio column has {negative_long_short} negative values.",
                )
            )
        expected = long_ratio / short_ratio.where(short_ratio > 0.0)
        mismatch = int(((long_short_ratio - expected).abs() > 0.05).sum())
        if mismatch:
            findings.append(
                _finding(
                    path,
                    "long_short_ratio_consistency",
                    "warning",
                    f"Long/short ratio differs from long/short account ratios in {mismatch} rows.",
                )
            )

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    if duplicate_timestamps:
        findings.append(
            _finding(
                path,
                "no_duplicate_timestamps",
                "error",
                f"Found {duplicate_timestamps} duplicate long/short-ratio timestamps.",
            )
        )

    if not dataframe["date"].is_monotonic_increasing:
        findings.append(
            _finding(path, "timestamps_sorted", "error", "Long/short-ratio timestamps are not sorted ascending.")
        )

    missing_intervals = 0
    if interval_seconds:
        missing_intervals = _check_time_intervals(path, dataframe, interval_seconds, findings)

    return _report(
        path,
        not any(f.severity == "error" for f in findings),
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


def check_order_book_parquet(
    path: Path, expected_timeframe: str | None = None
) -> OHLCVQualityReport:
    findings: list[DataQualityFinding] = []
    interval_seconds = timeframe_to_seconds(expected_timeframe) if expected_timeframe else None

    if not path.is_file():
        findings.append(
            _finding(path, "file_exists", "error", f"Order-book parquet file does not exist: {path}")
        )
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        findings.append(_finding(path, "read_parquet", "error", f"Could not read parquet: {exc}"))
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    columns = list(dataframe.columns)
    rows = len(dataframe)
    if "date" not in dataframe.columns:
        findings.append(_finding(path, "required_columns", "error", "Missing required order-book column: date"))
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    bid_price_column = _order_book_column(dataframe, ("best_bid", "bid_price", "bidPrice", "bid1Price", "bid"))
    ask_price_column = _order_book_column(dataframe, ("best_ask", "ask_price", "askPrice", "ask1Price", "ask"))
    bid_size_column = _order_book_column(dataframe, ("bid_size", "bidSize", "bid1Size", "bid_qty", "bidQty"))
    ask_size_column = _order_book_column(dataframe, ("ask_size", "askSize", "ask1Size", "ask_qty", "askQty"))
    missing_book_columns = [
        name
        for name, column in (
            ("best bid price", bid_price_column),
            ("best ask price", ask_price_column),
            ("bid size", bid_size_column),
            ("ask size", ask_size_column),
        )
        if column is None
    ]
    if missing_book_columns:
        findings.append(
            _finding(
                path,
                "required_columns",
                "error",
                f"Missing normalized order-book columns: {', '.join(missing_book_columns)}.",
            )
        )
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    if rows == 0:
        findings.append(_finding(path, "non_empty", "error", "Order-book file contains no rows."))
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

    bid_price = pd.to_numeric(dataframe[bid_price_column], errors="coerce")
    ask_price = pd.to_numeric(dataframe[ask_price_column], errors="coerce")
    bid_size = pd.to_numeric(dataframe[bid_size_column], errors="coerce")
    ask_size = pd.to_numeric(dataframe[ask_size_column], errors="coerce")
    for rule, series, label in (
        ("best_bid_numeric", bid_price, "Best bid price"),
        ("best_ask_numeric", ask_price, "Best ask price"),
        ("bid_size_numeric", bid_size, "Bid size"),
        ("ask_size_numeric", ask_size, "Ask size"),
    ):
        invalid = int(series.isna().sum())
        if invalid:
            findings.append(_finding(path, rule, "error", f"{label} has {invalid} non-numeric values."))

    for rule, series, label in (
        ("best_bid_positive", bid_price, "Best bid price"),
        ("best_ask_positive", ask_price, "Best ask price"),
    ):
        non_positive = int((series <= 0.0).sum())
        if non_positive:
            findings.append(_finding(path, rule, "error", f"{label} has {non_positive} non-positive values."))

    for rule, series, label in (
        ("bid_size_positive", bid_size, "Bid size"),
        ("ask_size_positive", ask_size, "Ask size"),
    ):
        non_positive = int((series <= 0.0).sum())
        if non_positive:
            findings.append(_finding(path, rule, "error", f"{label} has {non_positive} non-positive values."))

    crossed = int((bid_price > ask_price).sum())
    if crossed:
        findings.append(
            _finding(path, "best_bid_not_above_best_ask", "error", f"Found {crossed} crossed-book rows.")
        )

    midpoint = (bid_price + ask_price) / 2.0
    spread_bps = ((ask_price - bid_price) / midpoint) * 10000.0
    wide_spread = int((spread_bps > 500.0).sum())
    if wide_spread:
        findings.append(
            _finding(path, "spread_bps_reasonable", "warning", f"Found {wide_spread} rows with spread above 500 bps.")
        )

    imbalance_column = _order_book_column(dataframe, ("depth_imbalance", "book_imbalance", "top_of_book_imbalance"))
    if imbalance_column:
        imbalance = pd.to_numeric(dataframe[imbalance_column], errors="coerce")
        invalid_imbalance = int(imbalance.isna().sum())
        if invalid_imbalance:
            findings.append(
                _finding(
                    path,
                    "depth_imbalance_numeric",
                    "error",
                    f"Depth imbalance has {invalid_imbalance} non-numeric values.",
                )
            )
        out_of_bounds = int(((imbalance < -1.0) | (imbalance > 1.0)).sum())
        if out_of_bounds:
            findings.append(
                _finding(
                    path,
                    "depth_imbalance_bounds",
                    "error",
                    f"Depth imbalance has {out_of_bounds} values outside [-1, 1].",
                )
            )

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    if duplicate_timestamps:
        findings.append(
            _finding(
                path,
                "no_duplicate_timestamps",
                "error",
                f"Found {duplicate_timestamps} duplicate order-book timestamps.",
            )
        )

    if not dataframe["date"].is_monotonic_increasing:
        findings.append(
            _finding(path, "timestamps_sorted", "error", "Order-book timestamps are not sorted ascending.")
        )

    missing_intervals = 0
    if interval_seconds:
        missing_intervals = _check_time_intervals(path, dataframe, interval_seconds, findings)

    return _report(
        path,
        not any(f.severity == "error" for f in findings),
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


def check_liquidation_parquet(path: Path) -> OHLCVQualityReport:
    findings: list[DataQualityFinding] = []

    if not path.is_file():
        findings.append(
            _finding(path, "file_exists", "error", f"Liquidation parquet file does not exist: {path}")
        )
        return _report(path, False, 0, [], None, None, None, None, 0, 0, findings)

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        findings.append(_finding(path, "read_parquet", "error", f"Could not read parquet: {exc}"))
        return _report(path, False, 0, [], None, None, None, None, 0, 0, findings)

    columns = list(dataframe.columns)
    rows = len(dataframe)
    timestamp_column = _liquidation_column(dataframe, ("date", "T", "updatedTime", "updateTime", "timestamp", "ts"))
    side_column = _liquidation_column(dataframe, ("side", "S"))
    size_column = _liquidation_column(dataframe, ("size", "quantity", "qty", "v"))
    price_column = _liquidation_column(dataframe, ("price", "bankruptcy_price", "p"))
    missing_columns = [
        name
        for name, column in (
            ("timestamp", timestamp_column),
            ("side", side_column),
            ("size", size_column),
            ("bankruptcy price", price_column),
        )
        if column is None
    ]
    if missing_columns:
        findings.append(
            _finding(
                path,
                "required_columns",
                "error",
                f"Missing liquidation columns: {', '.join(missing_columns)}.",
            )
        )
        return _report(path, False, rows, columns, None, None, None, None, 0, 0, findings)

    if rows == 0:
        findings.append(_finding(path, "non_empty", "error", "Liquidation file contains no rows."))
        return _report(path, False, rows, columns, None, None, None, None, 0, 0, findings)

    dataframe = dataframe.copy()
    dataframe["date"] = _coerce_liquidation_timestamp(dataframe[timestamp_column])
    start = _timestamp_to_str(dataframe["date"].min())
    end = _timestamp_to_str(dataframe["date"].max())

    null_dates = int(dataframe["date"].isna().sum())
    if null_dates:
        findings.append(
            _finding(path, "date_not_null", "error", f"Found {null_dates} rows with invalid liquidation timestamps.")
        )

    side = dataframe[side_column].astype(str).str.strip().str.upper()
    invalid_side = int((~side.isin(["BUY", "SELL"])).sum())
    if invalid_side:
        findings.append(
            _finding(
                path,
                "side_buy_or_sell",
                "error",
                f"Liquidation side has {invalid_side} values outside Buy/Sell.",
            )
        )

    size = pd.to_numeric(dataframe[size_column], errors="coerce")
    invalid_size = int(size.isna().sum())
    if invalid_size:
        findings.append(
            _finding(path, "liquidation_size_numeric", "error", f"Liquidation size has {invalid_size} non-numeric values.")
        )
    non_positive_size = int((size <= 0.0).sum())
    if non_positive_size:
        findings.append(
            _finding(path, "liquidation_size_positive", "error", f"Liquidation size has {non_positive_size} non-positive values.")
        )

    price = pd.to_numeric(dataframe[price_column], errors="coerce")
    invalid_price = int(price.isna().sum())
    if invalid_price:
        findings.append(
            _finding(path, "liquidation_price_numeric", "error", f"Liquidation price has {invalid_price} non-numeric values.")
        )
    non_positive_price = int((price <= 0.0).sum())
    if non_positive_price:
        findings.append(
            _finding(path, "liquidation_price_positive", "error", f"Liquidation price has {non_positive_price} non-positive values.")
        )

    duplicate_events = int(
        dataframe[["date", side_column, size_column, price_column]]
        .astype(str)
        .duplicated()
        .sum()
    )
    if duplicate_events:
        findings.append(
            _finding(
                path,
                "no_duplicate_liquidation_events",
                "warning",
                f"Found {duplicate_events} duplicate liquidation event rows.",
            )
        )

    if not dataframe["date"].is_monotonic_increasing:
        findings.append(
            _finding(path, "timestamps_sorted", "error", "Liquidation timestamps are not sorted ascending.")
        )

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    return _report(
        path,
        not any(f.severity == "error" for f in findings),
        rows,
        columns,
        start,
        end,
        None,
        None,
        duplicate_timestamps,
        0,
        findings,
    )


def check_mark_price_parquet(
    path: Path, expected_timeframe: str | None = None
) -> OHLCVQualityReport:
    findings: list[DataQualityFinding] = []
    interval_seconds = timeframe_to_seconds(expected_timeframe) if expected_timeframe else None

    if not path.is_file():
        findings.append(
            _finding(path, "file_exists", "error", f"Mark-price parquet file does not exist: {path}")
        )
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        findings.append(_finding(path, "read_parquet", "error", f"Could not read parquet: {exc}"))
        return _report(path, False, 0, [], None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    columns = list(dataframe.columns)
    rows = len(dataframe)
    required_columns = ("date", *PRICE_COLUMNS)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        findings.append(
            _finding(
                path,
                "required_columns",
                "error",
                f"Missing required mark-price columns: {', '.join(missing_columns)}",
            )
        )
        return _report(path, False, rows, columns, None, None, expected_timeframe, interval_seconds, 0, 0, findings)

    if rows == 0:
        findings.append(_finding(path, "non_empty", "error", "Mark-price file contains no rows."))
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

    null_prices = dataframe[list(PRICE_COLUMNS)].isna().sum()
    for column, count in null_prices.items():
        if int(count) > 0:
            findings.append(
                _finding(path, f"{column}_not_null", "error", f"Column '{column}' has {int(count)} null values.")
            )

    for column in PRICE_COLUMNS:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        invalid = int(numeric.isna().sum())
        if invalid:
            findings.append(
                _finding(path, f"{column}_numeric", "error", f"Column '{column}' has {invalid} non-numeric values.")
            )
        non_positive = int((numeric <= 0).sum())
        if non_positive:
            findings.append(
                _finding(path, f"{column}_positive", "error", f"Column '{column}' has {non_positive} non-positive values.")
            )

    duplicate_timestamps = int(dataframe["date"].duplicated().sum())
    if duplicate_timestamps:
        findings.append(
            _finding(
                path,
                "no_duplicate_timestamps",
                "error",
                f"Found {duplicate_timestamps} duplicate mark-price timestamps.",
            )
        )

    if not dataframe["date"].is_monotonic_increasing:
        findings.append(
            _finding(path, "timestamps_sorted", "error", "Mark-price timestamps are not sorted ascending.")
        )

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


def default_funding_rate_quality_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_funding_rate_quality.json"


def default_mark_price_quality_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_mark_price_quality.json"


def default_open_interest_quality_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_open_interest_quality.json"


def default_long_short_ratio_quality_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_long_short_ratio_quality.json"


def default_order_book_quality_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_order_book_quality.json"


def default_liquidation_quality_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_liquidation_quality.json"


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


def _open_interest_value_column(dataframe: pd.DataFrame) -> str | None:
    for column in ("open_interest", "open", "close"):
        if column in dataframe.columns:
            return column
    return None


def _long_short_ratio_value_columns(dataframe: pd.DataFrame) -> tuple[str, str] | None:
    long_column = next(
        (
            column
            for column in ("long_account_ratio", "buy_ratio", "buyRatio")
            if column in dataframe.columns
        ),
        None,
    )
    short_column = next(
        (
            column
            for column in ("short_account_ratio", "sell_ratio", "sellRatio")
            if column in dataframe.columns
        ),
        None,
    )
    if not long_column or not short_column:
        return None
    return long_column, short_column


def _long_short_ratio_column(dataframe: pd.DataFrame) -> str | None:
    for column in ("long_short_ratio", "account_long_short_ratio"):
        if column in dataframe.columns:
            return column
    return None


def _order_book_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _liquidation_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _coerce_liquidation_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        median = numeric.dropna().abs().median()
        if median > 100000000000000:
            return pd.to_datetime(numeric, unit="ns", utc=True, errors="coerce")
        if median > 10000000000:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if median > 1000000000:
            return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


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
