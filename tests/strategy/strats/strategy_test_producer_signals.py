# pragma pylint: disable=missing-docstring, invalid-name

from pandas import DataFrame

from freqtrade.enums import CandleType
from freqtrade.strategy import IStrategy


class StrategyTestProducerSignals(IStrategy):
    """Strategy used by tests to validate upstream signal replay."""

    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count: int = 1

    minimal_roi = {"0": 0.0}
    stoploss = -0.99

    process_only_new_candles = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = str(metadata["pair"])
        candle_type = self.config.get("candle_type_def", CandleType.SPOT)

        producer_df, _la = self.dp.get_producer_df(
            pair,
            timeframe=self.timeframe,
            candle_type=candle_type,
            producer_name="default",
        )

        if not producer_df.empty and "date" in producer_df.columns:
            cols = [
                "enter_long",
                "exit_long",
                "enter_short",
                "exit_short",
                "enter_tag",
                "exit_tag",
            ]
            available = [c for c in cols if c in producer_df.columns]
            if available:
                p_df = producer_df[["date", *available]].copy()
                p_df.rename(columns={c: f"{c}_default" for c in available}, inplace=True)
                dataframe = dataframe.merge(p_df, on="date", how="left")

        for col in [
            "enter_long_default",
            "exit_long_default",
            "enter_short_default",
            "exit_short_default",
        ]:
            if col not in dataframe.columns:
                dataframe[col] = 0
            dataframe[col] = dataframe[col].fillna(0).astype(int)

        for col in ["enter_tag_default", "exit_tag_default"]:
            if col not in dataframe.columns:
                dataframe[col] = None

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["enter_long_default"] == 1, "enter_long"] = 1
        dataframe.loc[dataframe["enter_short_default"] == 1, "enter_short"] = 1
        dataframe.loc[dataframe["enter_long_default"] == 1, "enter_tag"] = dataframe[
            "enter_tag_default"
        ]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["exit_long_default"] == 1, "exit_long"] = 1
        dataframe.loc[dataframe["exit_short_default"] == 1, "exit_short"] = 1
        dataframe.loc[dataframe["exit_long_default"] == 1, "exit_tag"] = dataframe["exit_tag_default"]
        return dataframe
