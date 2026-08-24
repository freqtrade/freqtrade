# research/tests/test_regime.py
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.history import get_timerange
from research.regime import classify_regimes, regime_report
from research.walkforward import Window, WindowResult


TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
PAIR = "UNITTEST/BTC"
TIMEFRAME = "5m"


def _split_into_n_windows(min_date, max_date, n):
    """N equal-duration, contiguous windows spanning [min_date, max_date). Only
    test_start/test_end matter to classify_regimes, so train_start/train_end are set
    equal to test_start -- unused, but Window requires all four fields."""
    step = (max_date - min_date) / n
    windows = []
    for i in range(n):
        test_start = min_date + step * i
        test_end = min_date + step * (i + 1)
        windows.append(
            Window(
                train_start=test_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return windows


def _real_return_and_vol(pair, timeframe, datadir, window):
    """Independently recompute a window's total_return/realized_vol via a fresh
    history.load_data call -- a separate execution path from classify_regimes' own
    internals, used to derive the expected label from real data rather than a
    hand-picked one."""
    timerange = TimeRange(
        "date", "date", int(window.test_start.timestamp()), int(window.test_end.timestamp())
    )
    data = history.load_data(
        datadir=datadir, timeframe=timeframe, pairs=[pair], timerange=timerange
    )
    close = data[pair]["close"]
    if len(close) < 2:
        return 0.0, 0.0
    return float(close.iloc[-1] / close.iloc[0] - 1), float(close.pct_change().dropna().std())


def test_classify_regimes_trend_axis_matches_threshold_on_real_data():
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    min_date, max_date = get_timerange(full_data)
    windows = _split_into_n_windows(min_date, max_date, n=5)
    trend_threshold = 0.05

    labels = classify_regimes(
        PAIR, TIMEFRAME, TESTDATADIR, windows, trend_threshold=trend_threshold
    )

    assert len(labels) == len(windows)
    for window, label in zip(windows, labels, strict=True):
        total_return, _ = _real_return_and_vol(PAIR, TIMEFRAME, TESTDATADIR, window)
        if total_return > trend_threshold:
            expected_trend = "Bull"
        elif total_return < -trend_threshold:
            expected_trend = "Bear"
        else:
            expected_trend = "Sideways"
        assert label.split("/")[0] == expected_trend


def test_classify_regimes_volatility_axis_ranks_relative_to_median_on_real_data():
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    min_date, max_date = get_timerange(full_data)
    windows = _split_into_n_windows(min_date, max_date, n=5)

    labels = classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, windows)

    vols = [_real_return_and_vol(PAIR, TIMEFRAME, TESTDATADIR, w)[1] for w in windows]
    median_vol = float(np.median(vols))
    for vol, label in zip(vols, labels, strict=True):
        expected_volatility = "High" if vol > median_vol else "Low"
        assert label.split("/")[1] == expected_volatility


def test_classify_regimes_degenerate_window_is_sideways_and_does_not_raise():
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    min_date, _ = get_timerange(full_data)
    tiny_end = min_date + timedelta(seconds=1)  # far shorter than one 5m candle
    window = Window(
        train_start=min_date, train_end=min_date, test_start=min_date, test_end=tiny_end
    )

    labels = classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, [window])

    # single-window input: median vol is that window's own (degenerate) vol of 0.0,
    # so 0.0 > 0.0 is False -- deterministically "Low", not just "does not crash".
    assert labels == ["Sideways/Low"]


def test_classify_regimes_zero_candle_window_is_sideways_and_does_not_raise():
    """Regression test: a window whose test period loads ZERO candles (distinct from
    the 1-candle degenerate case above). freqtrade's history.load_data only inserts a
    pair into its returned dict when the loaded frame is non-empty, so a window placed
    entirely after the fixture data's max_date loads no candles at all and `pair` is
    absent from the dict -- indexing into it unguarded raises KeyError before the
    `len(close) < 2` fail-closed check ever runs.
    """
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    _, max_date = get_timerange(full_data)
    window = Window(
        train_start=max_date,
        train_end=max_date,
        test_start=max_date + timedelta(days=1),
        test_end=max_date + timedelta(days=2),
    )

    labels = classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, [window])

    # single-window input: median vol is that window's own (degenerate) vol of 0.0,
    # so 0.0 > 0.0 is False -- deterministically "Low", same as the 1-candle case.
    assert labels == ["Sideways/Low"]


def test_classify_regimes_raises_on_empty_windows():
    with pytest.raises(ValueError, match="windows"):
        classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, [])


def test_classify_regimes_handles_exactly_2_candle_window_without_nan_poisoning():
    """Regression test: exactly 2 candles produce 1 pct_change, which has undefined
    sample std (NaN with ddof=1). This NaN must not poison np.median(vols) for the
    entire call.
    """
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    min_date, max_date = get_timerange(full_data)

    # Create a window that spans exactly 2 candles:
    # history.load_data's timerange is inclusive on both ends, so to get exactly 2 candles
    # (at t=0 and t=5min), test_end must be < t=10min. Using t=5min+1sec ensures exactly 2.
    two_candle_window = Window(
        train_start=min_date,
        train_end=min_date,
        test_start=min_date,
        test_end=min_date + timedelta(minutes=5, seconds=1),  # spans exactly 2 candles
    )

    # Also create a normal multi-candle window to check it's not affected
    # This spans from after the 2-candle window to the end
    normal_window = Window(
        train_start=min_date + timedelta(minutes=5, seconds=1),
        train_end=min_date + timedelta(minutes=5, seconds=1),
        test_start=min_date + timedelta(minutes=5, seconds=1),
        test_end=max_date,
    )

    # Classify both windows together
    labels = classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, [two_candle_window, normal_window])

    # 2-candle window should have "Low" volatility (fail-closed)
    assert labels[0].endswith("/Low"), f"2-candle window volatility should be Low, got {labels[0]}"

    # The normal window must NOT be forced to "Low" by a poisoned NaN median.
    # With vols = [0.0, v] for v > 0, np.median gives median_vol = v/2 (not 0.0 --
    # np.median averages the two middle values of an even-length list). The real-vol
    # window's v > v/2 is True -> "High"; the degenerate window's 0.0 > v/2 is False ->
    # "Low".
    # The buggy code (unguarded .std() on 1 pct_change = NaN) forces labels[1] to
    # "Low", while the fixed code correctly labels it "High".
    assert labels[1].endswith("/High"), (
        f"Normal window should be High volatility (not NaN-poisoned), got {labels[1]}"
    )

    # Verify no exception was raised
    assert len(labels) == 2


_DUMMY_WINDOW = Window(
    train_start=datetime(2020, 1, 1, tzinfo=UTC),
    train_end=datetime(2020, 1, 8, tzinfo=UTC),
    test_start=datetime(2020, 1, 8, tzinfo=UTC),
    test_end=datetime(2020, 1, 15, tzinfo=UTC),
)


def test_regime_report_aggregates_by_label():
    wr_a1 = WindowResult(
        window=_DUMMY_WINDOW,
        variant_returns={},
        best_params={},
        train_sharpe=0.0,
        test_sharpe=1.0,
        test_n_trades=3,
        test_returns=[0.01, 0.02, -0.01],
    )
    wr_a2 = WindowResult(
        window=_DUMMY_WINDOW,
        variant_returns={},
        best_params={},
        train_sharpe=0.0,
        test_sharpe=2.0,
        test_n_trades=2,
        test_returns=[0.03, 0.04],
    )
    wr_b1 = WindowResult(
        window=_DUMMY_WINDOW,
        variant_returns={},
        best_params={},
        train_sharpe=0.0,
        test_sharpe=-1.0,
        test_n_trades=5,
        test_returns=[-0.05, -0.02, 0.01, -0.03, -0.01],
    )

    report = regime_report([wr_a1, wr_a2, wr_b1], ["Bull/High", "Bull/High", "Bear/Low"])

    assert set(report) == {"Bull/High", "Bear/Low"}
    assert report["Bull/High"]["n_windows"] == 2
    assert report["Bull/High"]["n_trades"] == 5
    assert report["Bull/High"]["mean_test_sharpe"] == pytest.approx(1.5)
    assert report["Bull/High"]["total_return"] == pytest.approx(0.01 + 0.02 - 0.01 + 0.03 + 0.04)
    assert report["Bear/Low"]["n_windows"] == 1
    assert report["Bear/Low"]["n_trades"] == 5
    assert report["Bear/Low"]["mean_test_sharpe"] == pytest.approx(-1.0)
    assert report["Bear/Low"]["total_return"] == pytest.approx(-0.05 - 0.02 + 0.01 - 0.03 - 0.01)


def test_regime_report_raises_on_mismatched_lengths():
    wr = WindowResult(
        window=_DUMMY_WINDOW,
        variant_returns={},
        best_params={},
        train_sharpe=0.0,
        test_sharpe=1.0,
        test_n_trades=1,
        test_returns=[0.01],
    )
    with pytest.raises(ValueError, match="same length"):
        regime_report([wr], ["Bull/High", "Bear/Low"])
