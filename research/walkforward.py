# research/walkforward.py
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.metrics import calculate_sharpe
from freqtrade.optimize.backtesting import Backtesting


@dataclass
class Window:
    """One walk-forward train/test period. `test_start` always equals
    `train_end`; see `generate_windows` for how consecutive windows relate."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class WindowResult:
    """Outcome of running one `Window` through `WalkForwardRunner.run_window`:
    the train-period Sharpe of every param variant (`variant_returns`, keyed by
    `variant_key`, feeding `research.pbo`), the train-selected best params, and
    the resulting out-of-sample (test-period) performance."""

    window: Window
    variant_returns: dict[str, float]
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_n_trades: int
    test_returns: list[float]


def variant_key(params: dict) -> str:
    """Canonical, order-independent string key for a param-variant dict, used
    to identify the same variant across windows (e.g. in `variant_returns`)."""
    return json.dumps(params, sort_keys=True)


def generate_windows(
    start: datetime, end: datetime, train_days: int, test_days: int
) -> list[Window]:
    """Rolling, contiguous windows: each window's test period starts exactly
    where its train period ends, and test periods are gapless across windows
    (the next window's test period starts exactly where the previous
    window's test period ended). Cursor advances by test_days each step, so
    every day in [start, end) is covered by at most one window's OOS test
    period, maximizing out-of-sample statistical power for Task 5's
    DSR/permutation-test/PBO evaluation."""
    windows: list[Window] = []
    cursor = start
    while True:
        train_end = cursor + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > end:
            break
        windows.append(Window(cursor, train_end, train_end, test_end))
        cursor = cursor + timedelta(days=test_days)
    return windows


class WalkForwardRunner:
    """Runs a strategy through freqtrade's `Backtesting` engine across a series
    of walk-forward windows, selecting the best param variant on each window's
    train period and evaluating it on that window's held-out test period."""

    def __init__(self, config: dict, pairs: list[str], timeframe: str, datadir: Path):
        self.config = config
        self.pairs = pairs
        self.timeframe = timeframe
        self.datadir = datadir

    def run_window(self, window: Window, param_grid: list[dict]) -> WindowResult:
        """Backtest every variant in `param_grid` on `window`'s train period,
        pick the highest-train-Sharpe variant, then backtest ONLY that variant
        on the test period.

        This ordering is the load-bearing invariant: parameter selection is
        strictly train-only. Test-period data is never touched until after
        `best_params` is fixed, so no parameter choice can be informed by data
        it will later be scored against -- the test-period Sharpe this returns
        is a genuine out-of-sample estimate, not a look-ahead-contaminated one.

        Indicators are (re)computed per param variant over the full
        [train_start, test_end] span before either backtest call (freqtrade's
        own convention), which is safe for backward-looking, row-local
        indicators but can leak test-period information into train-period
        selection for non-causal or DataFrame-wide-normalized indicators --
        see the inline note below.
        """
        backtesting = Backtesting(self.config)
        backtesting._set_strategy(backtesting.strategylist[0])

        timerange = TimeRange(
            "date", "date", int(window.train_start.timestamp()), int(window.test_end.timestamp())
        )
        data = history.load_data(
            datadir=self.datadir,
            timeframe=self.timeframe,
            pairs=self.pairs,
            timerange=timerange,
            startup_candles=backtesting.required_startup,
        )

        if not param_grid:
            raise ValueError("param_grid must not be empty")

        variant_returns: dict[str, float] = {}
        best_sharpe = -np.inf
        best_params: dict | None = None
        best_processed: dict | None = None

        for params in param_grid:
            for name, value in params.items():
                getattr(backtesting.strategy, name).value = value

            # ponytail: indicators are (re)computed per param variant over
            # [train_start, test_end], per freqtrade's own convention (§3 of the
            # architecture doc). Safe for backward-looking, row-local indicators
            # (this strategy's RSI); two families of indicator would leak
            # test-period information into the train-period parameter selection
            # done below even though they're "backward-looking" in the naive
            # sense: (1) a non-causal indicator (centered rolling, .shift(-n)),
            # and (2) any indicator normalized against a DataFrame-wide
            # statistic (global z-score, min/max scaling, global percentile
            # rank) computed over the whole [train_start, test_end] span, since
            # that statistic itself is contaminated by test-period rows. Run
            # freqtrade's lookahead-analysis (and audit for global
            # normalization) on any strategy before trusting this runner's
            # results for it.
            processed = backtesting.strategy.advise_all_indicators(data)
            processed = trim_dataframes(processed, timerange, backtesting.required_startup)

            train_result = backtesting.backtest(
                processed=deepcopy(processed),
                start_date=window.train_start,
                end_date=window.train_end,
            )
            train_trades = train_result["results"]
            sharpe = calculate_sharpe(
                train_trades, window.train_start, window.train_end, self.config["dry_run_wallet"]
            )
            key = variant_key(params)
            variant_returns[key] = (
                float((train_trades["profit_abs"] / self.config["dry_run_wallet"]).mean())
                if len(train_trades)
                else 0.0
            )

            if sharpe > best_sharpe:
                best_sharpe, best_params, best_processed = sharpe, params, processed

        if best_params is None or best_processed is None:
            raise RuntimeError(
                "no param variant produced a result"
            )  # unreachable: param_grid non-empty
        for name, value in best_params.items():
            getattr(backtesting.strategy, name).value = value
        test_result = backtesting.backtest(
            processed=deepcopy(best_processed),
            start_date=window.test_start,
            end_date=window.test_end,
        )
        test_trades = test_result["results"]
        test_returns = (test_trades["profit_abs"] / self.config["dry_run_wallet"]).tolist()
        test_sharpe = calculate_sharpe(
            test_trades, window.test_start, window.test_end, self.config["dry_run_wallet"]
        )

        return WindowResult(
            window=window,
            variant_returns=variant_returns,
            best_params=best_params,
            train_sharpe=best_sharpe,
            test_sharpe=test_sharpe,
            test_n_trades=len(test_trades),
            test_returns=test_returns,
        )

    def run(self, windows: list[Window], param_grid: list[dict]) -> list[WindowResult]:
        """Run `run_window` over every window in sequence and collect results."""
        return [self.run_window(w, param_grid) for w in windows]
