from __future__ import annotations

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


BOT_FACTORY_CANDIDATE_IDENTITY = {
    "candidate_id": "strong-uptrend-historical-ohlcv-candidate",
    "strategy_id": "long_only_strong_uptrend_momentum",
    "strategy_class_name": "DonchianTrendBullStrategy",
    "strategy_source_path": "user_data/strategies/DonchianTrendBullStrategy.py",
    "strategy_version": "strong_uptrend_momentum_v1",
    "signal_version": "strong_uptrend_signal_v1",
    "risk_policy_version": "long_only_risk_v1",
    "regime_classifier_version": "regime_classifier_v1",
    "cost_model_id": "cost_model_v1",
    "allowed_pairs": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
    "allowed_timeframes": ["5m"],
    "created_at": "2026-05-21T00:00:00+09:00",
    "source_artifacts": {
        "strategy_source": "user_data/strategies/DonchianTrendBullStrategy.py",
        "historical_backtest_metrics": (
            "data/backtests/DonchianTrendBullStrategy/"
            "historical_uptrend_20240202_20240305_v3/metrics.json"
        ),
        "historical_backtest_trades": (
            "data/backtests/DonchianTrendBullStrategy/"
            "historical_uptrend_20240202_20240305_v3/trades.csv"
        ),
        "selector_replay": (
            "data/regime_selector_replays/"
            "historical_uptrend_20240202_20240304/selector_replay.json"
        ),
    },
}


class DonchianTrendBullStrategy(IStrategy):
    """Long-only Donchian breakout strategy for historical bull-regime tests."""

    bot_factory_candidate_identity = BOT_FACTORY_CANDIDATE_IDENTITY

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count: int = 864
    process_only_new_candles = True

    minimal_roi = {"0": 10.0}
    stoploss = -0.12

    trailing_stop = False
    trailing_stop_positive = 0.0
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe["close"]
        high = dataframe["high"]
        low = dataframe["low"]

        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=72)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=288)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        dataframe["donchian_entry_high"] = high.rolling(144).max().shift(1)
        dataframe["donchian_exit_low"] = low.rolling(576).min().shift(1)
        dataframe["volume_mean"] = dataframe["volume"].rolling(288).mean()

        dataframe["ema_fast_slope"] = dataframe["ema_fast"] / dataframe["ema_fast"].shift(36) - 1.0
        dataframe["atr_pct"] = dataframe["atr"] / close

        directional_move = (close - close.shift(288)).abs()
        path_distance = close.diff().abs().rolling(288).sum()
        dataframe["range_efficiency"] = directional_move / path_distance

        dataframe["bull_regime"] = (
            (close > dataframe["ema_slow"])
            & (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_fast_slope"] > 0.0005)
            & (dataframe["adx"] > 14)
            & (dataframe["range_efficiency"] > 0.020)
            & (dataframe["atr_pct"] > 0.001)
            & (dataframe["volume"] > dataframe["volume_mean"] * 0.60)
        )
        dataframe["breakout_long"] = close > dataframe["donchian_entry_high"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        enter_long = dataframe["bull_regime"] & dataframe["breakout_long"]
        dataframe.loc[enter_long, ["enter_long", "enter_tag"]] = (
            1,
            "donchian_bull_breakout",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe["close"]
        exit_long = (
            (close < dataframe["donchian_exit_low"])
            & (close < dataframe["ema_slow"])
            & (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["ema_fast_slope"] < 0)
        )
        dataframe.loc[exit_long, ["exit_long", "exit_tag"]] = (
            1,
            "trend_break_or_donchian_exit",
        )
        return dataframe
