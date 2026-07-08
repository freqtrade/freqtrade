# pragma pylint: disable=missing-docstring, invalid-name
import json
from datetime import date, timedelta
from typing import Set
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class FxMacroDataEventRiskStrategy(IStrategy):
    """
    Example strategy template that suppresses new entries on top-tier USD macro
    release dates from FXMacroData's public release-calendar endpoint.
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 50
    stoploss = -0.10
    minimal_roi = {"0": 0.02}

    macro_blackout_dates: Set[str] = set()

    def bot_start(self, **kwargs) -> None:
        self.macro_blackout_dates = self._fetch_top_tier_dates("USD")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sma_fast"] = dataframe["close"].rolling(12).mean()
        dataframe["sma_slow"] = dataframe["close"].rolling(48).mean()
        dataframe["macro_blackout"] = dataframe["date"].dt.strftime("%Y-%m-%d").isin(
            self.macro_blackout_dates
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["sma_fast"] > dataframe["sma_slow"])
                & (~dataframe["macro_blackout"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["sma_fast"] < dataframe["sma_slow"])
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe

    @staticmethod
    def _fetch_top_tier_dates(currency: str, days: int = 30) -> Set[str]:
        start = date.today()
        end = start + timedelta(days=days)
        query = urlencode({"start_date": start.isoformat(), "end_date": end.isoformat()})
        url = f"https://fxmacrodata.com/api/v1/calendar/{currency}?{query}"

        with urlopen(url, timeout=20) as response:
            payload = json.load(response)

        events = pd.DataFrame(payload.get("data", []))
        if events.empty:
            return set()

        top_tier = events[
            (events["top_tier_for_currency"] == True)  # noqa: E712
            | (events["market_tier"] == 1)
        ].copy()
        top_tier["release_date"] = top_tier["announcement_datetime_utc"].fillna(
            top_tier["date"]
        ).str[:10]
        return set(top_tier["release_date"].dropna())
