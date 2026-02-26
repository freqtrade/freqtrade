# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)
from freqtrade.persistence import Trade as TradeModel

import talib.abstract as ta
from technical import qtpylib

logger = logging.getLogger(__name__)


def _get_pg_conn():
    """Get PostgreSQL connection if DATABASE_URL is set."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return None
    try:
        import psycopg2
        return psycopg2.connect(url.replace("postgres://", "postgresql://"))
    except Exception as e:
        logger.warning(f"Postgres connect failed: {e}")
        return None


def _log_trade_to_pg(trade: Trade, profit: float, profit_pct: float,
                     exit_reason: str, balance: float):
    """Write completed trade to PostgreSQL."""
    conn = _get_pg_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()
        dur = int((datetime.now(timezone.utc) - trade.open_date_utc).total_seconds() / 60)
        cur.execute("""
            INSERT INTO trade_log
                (trade_id, pair, side, open_time, close_time, open_rate,
                 close_rate, stake_amount, profit_amount, profit_pct,
                 exit_reason, entry_tag, duration_minutes, balance_after)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            trade.id, trade.pair, "long",
            trade.open_date_utc, datetime.now(timezone.utc),
            float(trade.open_rate), float(trade.close_rate or 0),
            float(trade.stake_amount), round(profit, 6),
            round(profit_pct * 100, 2), exit_reason,
            trade.enter_tag, dur, round(balance, 4)
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Postgres write failed: {e}")
        try:
            conn.close()
        except Exception:
            pass


def _log_status_to_pg(status: str, balance: float, open_trades: int,
                      total_trades: int, total_profit: float, total_pct: float):
    """Write bot status snapshot to PostgreSQL."""
    conn = _get_pg_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO bot_status
                (status, balance, open_trades, total_trades, total_profit, total_profit_pct)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (status, round(balance, 4), open_trades, total_trades,
              round(total_profit, 6), round(total_pct, 2)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Postgres status write failed: {e}")
        try:
            conn.close()
        except Exception:
            pass


class TrendRiderStrategy(IStrategy):
    """
    TrendRider v8 — MEXC spot, PostgreSQL tracking, clean Telegram.
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    minimal_roi = {
        "0": 0.06,
        "240": 0.035,
        "600": 0.02,
        "1200": 0.008,
    }

    stoploss = -0.018
    use_custom_stoploss = False

    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.030
    trailing_only_offset_is_reached = True

    timeframe = "1h"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 210

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 48},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 1,
                "stop_duration_candles": 48,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 168,
                "max_allowed_drawdown": 0.05,
                "trade_limit": 1,
                "stop_duration_candles": 72,
            },
        ]

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema21": {"color": "#3af0f0"},
            "ema50": {"color": "#f5a623"},
            "ema200": {"color": "#d63384"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "#7b68ee"}},
            "Regime": {"bull_regime": {"color": "#00ff7f"}},
        },
    }

    # ─── Custom Telegram Messages ───────────────────────────────

    def custom_entry_notify(self, trade: Trade, order, current_time, **kwargs) -> str:
        """Clean, simple entry notification."""
        return (
            f"📈 *BUY {trade.pair}*\n"
            f"💰 {trade.stake_amount:.2f} USDT @ {trade.open_rate:.4f}\n"
            f"🛡 Stop: {self.stoploss*100:.1f}% | Trail: {self.trailing_stop_positive_offset*100:.1f}%"
        )

    def custom_exit_notify(self, trade: Trade, order, current_time, **kwargs) -> str:
        """Clean, simple exit notification with P&L."""
        profit = trade.calc_profit_ratio(trade.close_rate) if trade.close_rate else 0
        profit_amt = trade.calc_profit(trade.close_rate) if trade.close_rate else 0
        emoji = "✅" if profit >= 0 else "❌"
        dur = current_time - trade.open_date_utc
        hours = int(dur.total_seconds() // 3600)
        mins = int((dur.total_seconds() % 3600) // 60)

        # Log to Postgres
        total_bal = trade.stake_amount + profit_amt
        _log_trade_to_pg(trade, profit_amt, profit, str(trade.exit_reason), total_bal)

        return (
            f"{emoji} *SELL {trade.pair}*\n"
            f"P&L: {profit*100:+.2f}% ({profit_amt:+.4f} USDT)\n"
            f"Held: {hours}h {mins}m | Reason: {trade.exit_reason}"
        )

    # ─── Indicators ─────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macdhist"] = macd["macdhist"]

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100

        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()

        dataframe["ema200_slope"] = (
            (dataframe["ema200"] - dataframe["ema200"].shift(10))
            / dataframe["ema200"].shift(10) * 100
        )
        dataframe["ema50_slope"] = (
            (dataframe["ema50"] - dataframe["ema50"].shift(5))
            / dataframe["ema50"].shift(5) * 100
        )

        dataframe["bull_regime"] = np.where(
            (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema200_slope"] > -0.1),
            1.0, 0.0,
        )

        dataframe["dist_to_ema21"] = (
            (dataframe["close"] - dataframe["ema21"]) / dataframe["ema21"] * 100
        )

        dataframe["rsi_rising"] = np.where(
            (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["rsi"] > dataframe["rsi"].shift(2)),
            1.0, 0.0,
        )

        dataframe["higher_low"] = np.where(
            dataframe["low"] > dataframe["low"].shift(2), 1.0, 0.0,
        )

        dataframe["green_candle"] = np.where(
            dataframe["close"] > dataframe["open"], 1.0, 0.0,
        )
        dataframe["prev_green"] = dataframe["green_candle"].shift(1)

        ema_above = (dataframe["ema21"] > dataframe["ema50"]).astype(int)
        streak = ema_above * 0
        for i in range(1, len(ema_above)):
            if ema_above.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i - 1] + 1
            else:
                streak.iloc[i] = 0
        dataframe["trend_age"] = streak

        return dataframe

    # ─── Entry / Exit ───────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bull_regime"] == 1)
                & (dataframe["ema21"] > dataframe["ema50"])
                & (dataframe["trend_age"] >= 5)
                & (dataframe["close"] > dataframe["ema21"])
                & (dataframe["ema50_slope"] > 0)
                & (dataframe["dist_to_ema21"] < 1.5)
                & (dataframe["dist_to_ema21"] > -0.3)
                & (dataframe["adx"] > 22)
                & (dataframe["plus_di"] > dataframe["minus_di"])
                & (dataframe["rsi"] > 45)
                & (dataframe["rsi"] < 62)
                & (dataframe["rsi_rising"] == 1)
                & (dataframe["macdhist"] > 0)
                & (dataframe["higher_low"] == 1)
                & (dataframe["green_candle"] == 1)
                & (dataframe["prev_green"] == 1)
                & (dataframe["volume"] > dataframe["volume_sma"] * 1.2)
                & (dataframe["atr_pct"] < 3.0)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["ema21"] < dataframe["ema50"])
                    | (dataframe["rsi"] > 72)
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
