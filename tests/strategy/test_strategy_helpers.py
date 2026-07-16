from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import freqtrade.strategy.informative_decorator as informative_decorator_module
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, RunMode
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.strategy import (
    IStrategy,
    informative,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)
from tests.conftest import generate_test_data, get_patched_exchange


class SharedInformativeCallCountStrategy(IStrategy):
    timeframe = "5m"
    stoploss = -0.10
    minimal_roi = {}
    process_only_new_candles = True

    def __init__(self, config) -> None:
        self.informative_calls: list[tuple[str, str, str]] = []
        super().__init__(config)

    @informative("15m", "BTC/{stake}")
    @informative("30m", "BTC/{stake}")
    @informative("1h", "BTC/{stake}")
    def populate_indicators_btc_first(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        self.informative_calls.append(("first", metadata["pair"], metadata["timeframe"]))
        dataframe["first_mean"] = dataframe["close"].rolling(3).mean()
        return dataframe

    @informative("15m", "BTC/{stake}", "{base}_{quote}_{column}_{timeframe}_second")
    @informative("30m", "BTC/{stake}", "{base}_{quote}_{column}_{timeframe}_second")
    @informative("1h", "BTC/{stake}", "{base}_{quote}_{column}_{timeframe}_second")
    def populate_indicators_btc_second(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        self.informative_calls.append(("second", metadata["pair"], metadata["timeframe"]))
        dataframe["second_range"] = dataframe["high"] - dataframe["low"]
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe


class MixedInformativeCacheStrategy(IStrategy):
    timeframe = "5m"
    stoploss = -0.10
    minimal_roi = {}

    def __init__(self, config) -> None:
        self.informative_calls: list[str] = []
        super().__init__(config)

    @informative("15m", "BTC/{stake}")
    @informative("30m", "BTC/{stake}", cache=False)
    def populate_indicators_btc(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        self.informative_calls.append(metadata["timeframe"])
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe


class _CountingInformativeFormatter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, column: str, asset: str, timeframe: str, **kwargs: str) -> str:
        self.calls += 1
        return f"{asset.split('/')[0].lower()}_{column.lower()}_second"


_same_source_formatter = _CountingInformativeFormatter()


class SameSourceInformativeCacheStrategy(IStrategy):
    timeframe = "5m"
    stoploss = -0.10
    minimal_roi = {}

    def __init__(self, config) -> None:
        self.informative_calls = 0
        super().__init__(config)

    @informative("15m", "BTC/{stake}", "{base}_{column}_first", ffill=True)
    @informative("15m", "BTC/{stake}", _same_source_formatter, ffill=False)
    def populate_indicators_btc(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        self.informative_calls += 1
        dataframe["double_close"] = dataframe["close"] * 2
        # A user column with this name must not collide with the private prepared merge column.
        dataframe["date_merge"] = dataframe["close"] * 3
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe


def _shared_informative_call_count_strategy(
    default_conf_usdt,
) -> tuple[SharedInformativeCallCountStrategy, dict[str, pd.DataFrame]]:
    strategy = SharedInformativeCallCountStrategy(default_conf_usdt)
    informative_data = {
        timeframe: generate_test_data(timeframe, 40) for timeframe in ("15m", "30m", "1h")
    }
    dataprovider = MagicMock()

    def market(pair: str) -> dict[str, str]:
        base, quote = pair.split("/")
        return {"base": base, "quote": quote}

    def get_pair_dataframe(pair: str, timeframe: str, candle_type: str) -> pd.DataFrame:
        assert pair == "BTC/USDT"
        return informative_data[timeframe].copy()

    dataprovider.market.side_effect = market
    dataprovider.get_pair_dataframe.side_effect = get_pair_dataframe
    strategy.dp = dataprovider
    return strategy, informative_data


def _mixed_informative_cache_strategy(
    default_conf_usdt,
) -> MixedInformativeCacheStrategy:
    strategy = MixedInformativeCacheStrategy(default_conf_usdt)
    informative_data = {
        timeframe: generate_test_data(timeframe, 40) for timeframe in ("15m", "30m")
    }
    dataprovider = MagicMock()
    dataprovider.market.side_effect = lambda pair: {
        "base": pair.split("/")[0],
        "quote": pair.split("/")[1],
    }
    dataprovider.get_pair_dataframe.side_effect = lambda pair, timeframe, candle_type: (
        informative_data[timeframe].copy()
    )
    strategy.dp = dataprovider
    return strategy


def _same_source_informative_cache_strategy(
    default_conf_usdt,
) -> SameSourceInformativeCacheStrategy:
    strategy = SameSourceInformativeCacheStrategy(default_conf_usdt)
    informative_data = generate_test_data("15m", 40)
    dataprovider = MagicMock()
    dataprovider.market.side_effect = lambda pair: {
        "base": pair.split("/")[0],
        "quote": pair.split("/")[1],
    }
    dataprovider.get_pair_dataframe.side_effect = lambda pair, timeframe, candle_type: (
        informative_data.copy()
    )
    strategy.dp = dataprovider
    return strategy


def test_merge_informative_pair():
    data = generate_test_data("15m", 40)
    informative = generate_test_data("1h", 40)
    cols_inf = list(informative.columns)

    result = merge_informative_pair(data, informative, "15m", "1h", ffill=True)
    assert isinstance(result, pd.DataFrame)
    assert list(informative.columns) == cols_inf
    assert len(result) == len(data)
    assert "date" in result.columns
    assert result["date"].equals(data["date"])
    assert "date_1h" in result.columns

    assert "open" in result.columns
    assert "open_1h" in result.columns
    assert result["open"].equals(data["open"])

    assert "close" in result.columns
    assert "close_1h" in result.columns
    assert result["close"].equals(data["close"])

    assert "volume" in result.columns
    assert "volume_1h" in result.columns
    assert result["volume"].equals(data["volume"])

    # First 3 rows are empty.
    # Pre-fillup doesn't happen as there is no prior candlw in the informative dataframe
    assert result.iloc[0]["date_1h"] is pd.NaT
    assert result.iloc[1]["date_1h"] is pd.NaT
    assert result.iloc[2]["date_1h"] is pd.NaT
    # Next 4 rows contain the starting date (0:00)
    assert result.iloc[3]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[4]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[5]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[6]["date_1h"] == result.iloc[0]["date"]
    # Next 4 rows contain the next Hourly date original date row 4
    assert result.iloc[7]["date_1h"] == result.iloc[4]["date"]
    assert result.iloc[8]["date_1h"] == result.iloc[4]["date"]

    informative = generate_test_data("1h", 40)
    result = merge_informative_pair(data, informative, "15m", "1h", ffill=False)
    # First 3 rows are empty
    assert result.iloc[0]["date_1h"] is pd.NaT
    assert result.iloc[1]["date_1h"] is pd.NaT
    assert result.iloc[2]["date_1h"] is pd.NaT
    # Next 4 rows contain the starting date (0:00)
    assert result.iloc[3]["date_1h"] == result.iloc[0]["date"]
    assert result.iloc[4]["date_1h"] is pd.NaT
    assert result.iloc[5]["date_1h"] is pd.NaT
    assert result.iloc[6]["date_1h"] is pd.NaT
    # Next 4 rows contain the next Hourly date original date row 4
    assert result.iloc[7]["date_1h"] == result.iloc[4]["date"]
    assert result.iloc[8]["date_1h"] is pd.NaT


def test_merge_informative_pair_weekly():
    # Covers roughly 2 months - until 2023-01-10
    data = generate_test_data("1h", 1040, "2022-11-28")
    informative = generate_test_data("1w", 40, "2022-11-01")
    informative["day"] = informative["date"].dt.day_name()

    result = merge_informative_pair(data, informative, "1h", "1w", ffill=True)
    assert isinstance(result, pd.DataFrame)
    # 2022-12-24 is a Saturday
    candle1 = result.loc[(result["date"] == "2022-12-24T22:00:00.000Z")]
    assert candle1.iloc[0]["date"] == pd.Timestamp("2022-12-24T22:00:00.000Z")
    assert candle1.iloc[0]["date_1w"] == pd.Timestamp("2022-12-12T00:00:00.000Z")

    candle2 = result.loc[(result["date"] == "2022-12-24T23:00:00.000Z")]
    assert candle2.iloc[0]["date"] == pd.Timestamp("2022-12-24T23:00:00.000Z")
    assert candle2.iloc[0]["date_1w"] == pd.Timestamp("2022-12-12T00:00:00.000Z")

    # 2022-12-25 is a Sunday
    candle3 = result.loc[(result["date"] == "2022-12-25T22:00:00.000Z")]
    assert candle3.iloc[0]["date"] == pd.Timestamp("2022-12-25T22:00:00.000Z")
    # Still old candle
    assert candle3.iloc[0]["date_1w"] == pd.Timestamp("2022-12-12T00:00:00.000Z")

    candle4 = result.loc[(result["date"] == "2022-12-25T23:00:00.000Z")]
    assert candle4.iloc[0]["date"] == pd.Timestamp("2022-12-25T23:00:00.000Z")
    assert candle4.iloc[0]["date_1w"] == pd.Timestamp("2022-12-19T00:00:00.000Z")


def test_merge_informative_pair_monthly():
    # Covers roughly 2 months - until 2023-01-10
    data = generate_test_data("1h", 1040, "2022-11-28")
    informative = generate_test_data("1M", 40, "2022-01-01")

    result = merge_informative_pair(data, informative, "1h", "1M", ffill=True)
    assert isinstance(result, pd.DataFrame)
    candle1 = result.loc[(result["date"] == "2022-12-31T22:00:00.000Z")]
    assert candle1.iloc[0]["date"] == pd.Timestamp("2022-12-31T22:00:00.000Z")
    assert candle1.iloc[0]["date_1M"] == pd.Timestamp("2022-11-01T00:00:00.000Z")

    candle2 = result.loc[(result["date"] == "2022-12-31T23:00:00.000Z")]
    assert candle2.iloc[0]["date"] == pd.Timestamp("2022-12-31T23:00:00.000Z")
    assert candle2.iloc[0]["date_1M"] == pd.Timestamp("2022-12-01T00:00:00.000Z")

    # Candle is empty, as the start-date did fail.
    candle3 = result.loc[(result["date"] == "2022-11-30T22:00:00.000Z")]
    assert candle3.iloc[0]["date"] == pd.Timestamp("2022-11-30T22:00:00.000Z")
    # Merged on prior month
    assert candle3.iloc[0]["date_1M"] == pd.Timestamp("2022-10-01T00:00:00.000Z")

    # First candle with 1M data merged.
    candle4 = result.loc[(result["date"] == "2022-11-30T23:00:00.000Z")]
    assert candle4.iloc[0]["date"] == pd.Timestamp("2022-11-30T23:00:00.000Z")
    assert candle4.iloc[0]["date_1M"] == pd.Timestamp("2022-11-01T00:00:00.000Z")

    # Very first candle in the result dataframe
    # Merged the latest informative candle before the start-date
    candle5 = result.iloc[0]
    assert candle5["date"] == pd.Timestamp("2022-11-28T00:00:00.000Z")
    assert candle5["date_1M"] == pd.Timestamp("2022-10-01T00:00:00.000Z")


def test_merge_informative_pair_no_overlap():
    # Covers roughly a day
    data = generate_test_data("1m", 1440, "2022-11-28")
    # Data stops WAY before the main data starts
    informative = generate_test_data("1h", 40, "2022-11-01")

    result = merge_informative_pair(data, informative, "1m", "1h", ffill=True)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert "date" in result.columns
    assert result["date"].equals(data["date"])
    assert "date_1h" in result.columns
    # If there's no overlap, forward filling should not fill anything
    assert result["date_1h"].isnull().all()


def test_merge_informative_pair_same():
    data = generate_test_data("15m", 40)
    informative = generate_test_data("15m", 40)

    result = merge_informative_pair(data, informative, "15m", "15m", ffill=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert "date" in result.columns
    assert result["date"].equals(data["date"])
    assert "date_15m" in result.columns

    assert "open" in result.columns
    assert "open_15m" in result.columns
    assert result["open"].equals(data["open"])

    assert "close" in result.columns
    assert "close_15m" in result.columns
    assert result["close"].equals(data["close"])

    assert "volume" in result.columns
    assert "volume_15m" in result.columns
    assert result["volume"].equals(data["volume"])

    # Dates match 1:1
    assert result["date_15m"].equals(result["date"])


def test_merge_informative_pair_lower():
    data = generate_test_data("1h", 40)
    informative = generate_test_data("15m", 40)

    with pytest.raises(ValueError, match=r"Tried to merge a faster timeframe .*"):
        merge_informative_pair(data, informative, "1h", "15m", ffill=True)


def test_merge_informative_pair_empty():
    data = generate_test_data("1h", 40)
    informative = pd.DataFrame(columns=data.columns)

    result = merge_informative_pair(data, informative, "1h", "2h", ffill=True)
    assert result["date"].equals(data["date"])

    assert list(result.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date_2h",
        "open_2h",
        "high_2h",
        "low_2h",
        "close_2h",
        "volume_2h",
    ]
    # We merge an empty dataframe, so all values should be NaN
    for col in ["date_2h", "open_2h", "high_2h", "low_2h", "close_2h", "volume_2h"]:
        assert result[col].isnull().all()


def test_merge_informative_pair_suffix():
    data = generate_test_data("15m", 20)
    informative = generate_test_data("1h", 20)

    result = merge_informative_pair(
        data, informative, "15m", "1h", append_timeframe=False, suffix="suf"
    )

    assert "date" in result.columns
    assert result["date"].equals(data["date"])
    assert "date_suf" in result.columns

    assert "open_suf" in result.columns
    assert "open_1h" not in result.columns

    assert list(result.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "date_suf",
        "open_suf",
        "high_suf",
        "low_suf",
        "close_suf",
        "volume_suf",
    ]


def test_merge_informative_pair_suffix_append_timeframe():
    data = generate_test_data("15m", 20)
    informative = generate_test_data("1h", 20)

    with pytest.raises(ValueError, match=r"You can not specify `append_timeframe` .*"):
        merge_informative_pair(data, informative, "15m", "1h", suffix="suf")


@pytest.mark.parametrize(
    "side,profitrange",
    [
        # profit range for long is [-1, inf] while for shorts is [-inf, 1]
        ("long", [-0.99, 2, 30]),
        ("short", [-2.0, 0.99, 30]),
    ],
)
def test_stoploss_from_open(side, profitrange):
    open_price_ranges = [
        [0.01, 1.00, 30],
        [1, 100, 30],
        [100, 10000, 30],
    ]

    for open_range in open_price_ranges:
        for open_price in np.linspace(*open_range):
            for desired_stop in np.linspace(-0.50, 0.50, 30):
                if side == "long":
                    # -1 is not a valid current_profit, should return 1
                    assert stoploss_from_open(desired_stop, -1) == 1
                else:
                    # 1 is not a valid current_profit for shorts, should return 1
                    assert stoploss_from_open(desired_stop, 1, True) == 1

                for current_profit in np.linspace(*profitrange):
                    if side == "long":
                        current_price = open_price * (1 + current_profit)
                        expected_stop_price = open_price * (1 + desired_stop)
                        stoploss = stoploss_from_open(desired_stop, current_profit)
                        stop_price = current_price * (1 - stoploss)
                    else:
                        current_price = open_price * (1 - current_profit)
                        expected_stop_price = open_price * (1 - desired_stop)
                        stoploss = stoploss_from_open(desired_stop, current_profit, True)
                        stop_price = current_price * (1 + stoploss)

                    assert stoploss >= 0
                    # Technically the formula can yield values greater than 1 for shorts
                    # even though it doesn't make sense because the position would be liquidated
                    if side == "long":
                        assert stoploss <= 1

                    # there is no correct answer if the expected stop price is above
                    # the current price
                    if (side == "long" and expected_stop_price > current_price) or (
                        side == "short" and expected_stop_price < current_price
                    ):
                        assert stoploss == 0
                    else:
                        assert pytest.approx(stop_price) == expected_stop_price


@pytest.mark.parametrize(
    "side,rel_stop,curr_profit,leverage,expected",
    [
        # profit range for long is [-1, inf] while for shorts is [-inf, 1]
        ("long", 0, -1, 1, 1),
        ("long", 0, 0.1, 1, 0.09090909),
        ("long", -0.1, 0.1, 1, 0.18181818),
        ("long", 0.1, 0.2, 1, 0.08333333),
        ("long", 0.1, 0.5, 1, 0.266666666),
        ("long", 0.1, 5, 1, 0.816666666),  # 500% profit, set stoploss to 10% above open price
        ("long", 0, 5, 10, 3.3333333),  # 500% profit, set stoploss break even
        ("long", 0.1, 5, 10, 3.26666666),  # 500% profit, set stoploss to 10% above open price
        ("long", -0.1, 5, 10, 3.3999999),  # 500% profit, set stoploss to 10% belowopen price
        ("short", 0, 0.1, 1, 0.1111111),
        ("short", -0.1, 0.1, 1, 0.2222222),
        ("short", 0.1, 0.2, 1, 0.125),
        ("short", 0.1, 1, 1, 1),
        ("short", -0.01, 5, 10, 10.01999999),  # 500% profit at 10x
    ],
)
def test_stoploss_from_open_leverage(side, rel_stop, curr_profit, leverage, expected):
    stoploss = stoploss_from_open(rel_stop, curr_profit, side == "short", leverage)
    assert pytest.approx(stoploss) == expected
    open_rate = 100
    if stoploss != 1:
        if side == "long":
            current_rate = open_rate * (1 + curr_profit / leverage)
            stop = current_rate * (1 - stoploss / leverage)
            assert pytest.approx(stop) == open_rate * (1 + rel_stop / leverage)
        else:
            current_rate = open_rate * (1 - curr_profit / leverage)
            stop = current_rate * (1 + stoploss / leverage)
            assert pytest.approx(stop) == open_rate * (1 - rel_stop / leverage)


def test_stoploss_from_absolute():
    assert pytest.approx(stoploss_from_absolute(90, 100)) == 1 - (90 / 100)
    assert pytest.approx(stoploss_from_absolute(90, 100)) == 0.1
    assert pytest.approx(stoploss_from_absolute(95, 100)) == 0.05
    assert pytest.approx(stoploss_from_absolute(100, 100)) == 0
    assert pytest.approx(stoploss_from_absolute(110, 100)) == 0
    assert pytest.approx(stoploss_from_absolute(100, 0)) == 1
    assert pytest.approx(stoploss_from_absolute(0, 100)) == 1
    assert pytest.approx(stoploss_from_absolute(0, 100, False, leverage=5)) == 5

    assert pytest.approx(stoploss_from_absolute(90, 100, True)) == 0
    assert pytest.approx(stoploss_from_absolute(100, 100, True)) == 0
    assert pytest.approx(stoploss_from_absolute(110, 100, True)) == -(1 - (110 / 100))
    assert pytest.approx(stoploss_from_absolute(110, 100, True)) == 0.1
    assert pytest.approx(stoploss_from_absolute(105, 100, True)) == 0.05
    assert pytest.approx(stoploss_from_absolute(105, 100, True, 5)) == 0.05 * 5
    assert pytest.approx(stoploss_from_absolute(100, 0, True)) == 1
    assert pytest.approx(stoploss_from_absolute(0, 100, True)) == 0
    assert pytest.approx(stoploss_from_absolute(100, 99, is_short=True)) == 0.01010101
    assert pytest.approx(stoploss_from_absolute(100, 90, is_short=True)) == 0.1111111
    assert pytest.approx(stoploss_from_absolute(100, 1, is_short=True)) == 99.0
    assert pytest.approx(stoploss_from_absolute(100, 1, is_short=True, leverage=5)) == 495.0
    assert pytest.approx(stoploss_from_absolute(100, 90, is_short=True, leverage=5)) == 0.55555555


@pytest.mark.parametrize("trading_mode", ["futures", "spot"])
def test_informative_decorator(mocker, default_conf_usdt, trading_mode):
    candle_def = CandleType.get_default(trading_mode)
    default_conf_usdt["candle_type_def"] = candle_def
    test_data_5m = generate_test_data("5m", 40)
    test_data_30m = generate_test_data("30m", 40)
    test_data_1h = generate_test_data("1h", 40)
    data = {
        ("XRP/USDT", "5m", candle_def): test_data_5m,
        ("XRP/USDT", "30m", candle_def): test_data_30m,
        ("XRP/USDT", "1h", candle_def): test_data_1h,
        ("XRP/BTC", "1h", candle_def): test_data_1h,  # from {base}/BTC
        ("LTC/USDT", "5m", candle_def): test_data_5m,
        ("LTC/USDT", "30m", candle_def): test_data_30m,
        ("LTC/USDT", "1h", candle_def): test_data_1h,
        ("LTC/BTC", "1h", candle_def): test_data_1h,  # from {base}/BTC
        ("NEO/USDT", "30m", candle_def): test_data_30m,
        ("NEO/USDT", "5m", CandleType.SPOT): test_data_5m,  # Explicit request with '' as candletype
        ("NEO/USDT", "15m", candle_def): test_data_5m,  # Explicit request with '' as candletype
        ("NEO/USDT", "1h", candle_def): test_data_1h,
        ("ETH/USDT", "1h", candle_def): test_data_1h,
        ("ETH/USDT", "30m", candle_def): test_data_30m,
        ("ETH/BTC", "1h", CandleType.SPOT): test_data_1h,  # Explicitly selected as spot
    }
    default_conf_usdt["strategy"] = "InformativeDecoratorTest"
    strategy = StrategyResolver.load_strategy(default_conf_usdt)
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    default_conf_usdt["candle_type_def"] = candle_def
    strategy.dp = DataProvider({}, exchange, None)
    mocker.patch.object(
        strategy.dp, "current_whitelist", return_value=["XRP/USDT", "LTC/USDT", "NEO/USDT"]
    )

    assert len(strategy._ft_informative) == 7  # Equal to number of decorators used
    informative_pairs = [
        ("XRP/USDT", "1h", candle_def),
        ("XRP/BTC", "1h", candle_def),
        ("LTC/USDT", "1h", candle_def),
        ("LTC/BTC", "1h", candle_def),
        ("XRP/USDT", "30m", candle_def),
        ("LTC/USDT", "30m", candle_def),
        ("NEO/USDT", "1h", candle_def),
        ("NEO/USDT", "30m", candle_def),
        ("NEO/USDT", "5m", candle_def),
        ("NEO/USDT", "15m", candle_def),
        ("NEO/USDT", "2h", CandleType.FUTURES),
        ("ETH/BTC", "1h", CandleType.SPOT),  # One candle remains as spot
        ("ETH/USDT", "30m", candle_def),
    ]
    for inf_pair in informative_pairs:
        assert inf_pair in strategy.gather_informative_pairs()

    def test_historic_ohlcv(pair, timeframe, candle_type):
        return data.get(
            (pair, timeframe or strategy.timeframe, CandleType.from_string(candle_type)),
            pd.DataFrame(),
        ).copy()

    mocker.patch(
        "freqtrade.data.dataprovider.DataProvider.historic_ohlcv", side_effect=test_historic_ohlcv
    )

    analyzed = strategy.advise_all_indicators(
        {p: data[(p, strategy.timeframe, candle_def)] for p in ("XRP/USDT", "LTC/USDT")}
    )
    expected_columns = [
        "rsi_1h",
        "rsi_30m",  # Stacked informative decorators
        "neo_usdt_rsi_1h",  # NEO 1h informative
        "rsi_NEO_USDT_neo_usdt_NEO/USDT_30m",  # Column formatting
        "rsi_from_callable",  # Custom column formatter
        "eth_btc_rsi_1h",  # Quote currency not matching stake currency
        "rsi",
        "rsi_less",  # Non-informative columns
        "rsi_5m",  # Manual informative dataframe
    ]
    for _, dataframe in analyzed.items():
        for col in expected_columns:
            assert col in dataframe.columns

    # Test non-available pairs
    del data[("ETH/BTC", "1h", CandleType.SPOT)]
    with pytest.raises(
        ValueError, match=r"Informative dataframe for \(ETH\/BTC, 1h, spot\) is empty.*"
    ):
        strategy.advise_all_indicators(
            {p: data[(p, strategy.timeframe, candle_def)] for p in ("XRP/USDT", "LTC/USDT")}
        )


def test_shared_informative_call_count_for_each_base_pair(default_conf_usdt):
    strategy, _ = _shared_informative_call_count_strategy(default_conf_usdt)
    base_data = generate_test_data("5m", 40)
    base_pairs = ("XRP/USDT", "LTC/USDT", "ETH/USDT")

    strategy.advise_all_indicators({pair: base_data.copy() for pair in base_pairs})

    # Each combination of inf pair, timeframe and populate functions is called once.
    assert len(strategy.informative_calls) == 6

    # Check the cache that each calls indeed only called once
    assert Counter(strategy.informative_calls) == Counter(
        {
            ("first", "BTC/USDT", "15m"): 1,
            ("first", "BTC/USDT", "30m"): 1,
            ("first", "BTC/USDT", "1h"): 1,
            ("second", "BTC/USDT", "15m"): 1,
            ("second", "BTC/USDT", "30m"): 1,
            ("second", "BTC/USDT", "1h"): 1,
        }
    )

    # Raw data is still requested for every informative pair/timeframe/function combinations.
    assert strategy.dp.get_pair_dataframe.call_count == 18


@pytest.mark.parametrize("runmode", [RunMode.DRY_RUN, RunMode.LIVE])
def test_informative_cache_is_opt_out_per_decorator(default_conf_usdt, runmode):
    default_conf_usdt["runmode"] = runmode
    strategy = _mixed_informative_cache_strategy(default_conf_usdt)
    base_data = generate_test_data("5m", 40)

    # Make sure only 15m is cached, 30m isn't cached, and there is no other informative tf.
    assert {inf_data.timeframe: inf_data.cache for inf_data, _ in strategy._ft_informative} == {
        "15m": True,
        "30m": False,
    }

    for pair in ("XRP/USDT", "LTC/USDT", "ETH/USDT"):
        strategy.advise_indicators(base_data.copy(), {"pair": pair})

    # 15m is cached, so only one call for the first pair.
    # 30m is not cached, so 3 calls for each pair.
    assert Counter(strategy.informative_calls) == Counter({"15m": 1, "30m": 3})


def test_shared_informative_cache_preserves_indicator_data(default_conf_usdt):
    strategy, informative_data = _shared_informative_call_count_strategy(default_conf_usdt)
    base_data = generate_test_data("5m", 40)
    base_pairs = ("XRP/USDT", "LTC/USDT", "ETH/USDT")

    # Calculated dataframe using cached informative data
    analyzed = strategy.advise_all_indicators({pair: base_data.copy() for pair in base_pairs})

    for timeframe, source in informative_data.items():
        expected_informative = source[["date"]].copy()
        expected_informative["first_mean"] = source["close"].rolling(3).mean()
        expected_informative["second_range"] = source["high"] - source["low"]

        # Manually calculated-and-merged informative dataframe
        expected = merge_informative_pair(
            base_data.copy(), expected_informative, "5m", timeframe, ffill=True
        )

        # Make sure both dataframes are equal
        for pair in base_pairs:
            pd.testing.assert_series_equal(
                analyzed[pair][f"btc_usdt_first_mean_{timeframe}"],
                expected[f"first_mean_{timeframe}"],
                check_names=False,
            )
            pd.testing.assert_series_equal(
                analyzed[pair][f"btc_usdt_second_range_{timeframe}_second"],
                expected[f"second_range_{timeframe}"],
                check_names=False,
            )


def test_informative_cache_reuses_prepared_frame_and_isolates_output(mocker, default_conf_usdt):
    # Function that convert informative data to merge-ready dataframe
    prepare_spy = mocker.spy(informative_decorator_module, "_prepare_informative_pair")
    strategy, informative_data = _shared_informative_call_count_strategy(default_conf_usdt)
    base_data = generate_test_data("5m", 40)

    first = strategy.advise_indicators(base_data.copy(), {"pair": "XRP/USDT"})
    # Prepare data function is called once for each informative pair/timeframe/function combination.
    assert prepare_spy.call_count == 6

    # Modify the first dataframe to later make sure informative cache is isolated.
    first.loc[:, "btc_usdt_first_mean_15m"] = -12345

    second = strategy.advise_indicators(base_data.copy(), {"pair": "LTC/USDT"})

    # Prepare count remains the same, meaning the cached informative datas are reused.
    assert prepare_spy.call_count == 6

    # Make sure the cache only contains 6 unique informative pair/timeframe/function combinations.
    assert len(strategy.informative_calls) == 6

    # Running the same strategy in backtest mode to disable cache.
    baseline_conf = default_conf_usdt.copy()
    baseline_conf["runmode"] = RunMode.BACKTEST
    baseline, baseline_data = _shared_informative_call_count_strategy(baseline_conf)
    baseline_data.update(
        {timeframe: dataframe.copy() for timeframe, dataframe in informative_data.items()}
    )
    expected = baseline.advise_indicators(base_data.copy(), {"pair": "LTC/USDT"})

    # Cached data is the same as non-cached data
    # Also proving cached data is isolated, as the change on first dataframe didn't have any effect.
    pd.testing.assert_frame_equal(second, expected)


def test_informative_cache_shares_preparation_across_formatters(mocker, default_conf_usdt):
    prepare_spy = mocker.spy(informative_decorator_module, "_prepare_informative_pair")
    strategy = _same_source_informative_cache_strategy(default_conf_usdt)
    base_data = generate_test_data("5m", 40)

    first = strategy.advise_indicators(base_data.copy(), {"pair": "XRP/USDT"})
    second = strategy.advise_indicators(base_data.copy(), {"pair": "LTC/USDT"})

    # Informative data is calculated once. Column's format isn't used in cache key
    assert strategy.informative_calls == 1

    # The informative data is prepared once
    assert prepare_spy.call_count == 1

    # Make sure the data is available in both dataframes,
    # and the first dataframe has more non-NaN values than the second one,
    # meaning ffill only applied when merging data, not when preparing the informative data.
    for dataframe in (first, second):
        assert "btc_double_close_first" in dataframe.columns
        assert "btc_double_close_second" in dataframe.columns
        assert "btc_date_merge_first" in dataframe.columns
        assert "btc_date_merge_second" in dataframe.columns
        assert (
            dataframe["btc_double_close_first"].notna().sum()
            > dataframe["btc_double_close_second"].notna().sum()
        )


def test_informative_cache_reuses_result_for_new_base_candle(default_conf_usdt):
    strategy, _ = _shared_informative_call_count_strategy(default_conf_usdt)
    metadata = {"pair": "XRP/USDT"}

    # Initial indicator calculation
    strategy.advise_indicators(generate_test_data("5m", 40), metadata)
    assert len(strategy.informative_calls) == 6

    # New candle is added, but the informative data isn't changed
    strategy.advise_indicators(generate_test_data("5m", 41), metadata)
    # Informative count remains the same, meaning the cached informative datas are reused.
    assert len(strategy.informative_calls) == 6


def test_informative_cache_invalidates_changed_latest_candle(default_conf_usdt):
    strategy, informative_data = _shared_informative_call_count_strategy(default_conf_usdt)
    informative = informative_data["30m"]
    last_index = informative.index[-1]
    base_data = generate_test_data("5m", 40)
    metadata = {"pair": "XRP/USDT"}

    strategy.advise_indicators(base_data.copy(), metadata)
    expected_call_counts = Counter(
        {
            ("first", "BTC/USDT", "15m"): 1,
            ("first", "BTC/USDT", "30m"): 1,
            ("first", "BTC/USDT", "1h"): 1,
            ("second", "BTC/USDT", "15m"): 1,
            ("second", "BTC/USDT", "30m"): 1,
            ("second", "BTC/USDT", "1h"): 1,
        }
    )
    assert Counter(strategy.informative_calls) == expected_call_counts

    changes = [
        ("date", pd.Timedelta(minutes=30)),
        ("open", 1),
        ("high", 1),
        ("low", 1),
        ("close", 1),
        ("volume", 1),
    ]

    # Modify 30m candle's last row, which should invalidate the cache for 30m informative data.
    for column, change in changes:
        informative.loc[last_index, column] += change
        strategy.advise_indicators(base_data.copy(), metadata)

        # Only the 30m informative data should be recalculated.
        # The other informative data should be reused.
        expected_call_counts.update(
            {
                ("first", "BTC/USDT", "30m"): 1,
                ("second", "BTC/USDT", "30m"): 1,
            }
        )
        assert Counter(strategy.informative_calls) == expected_call_counts


def test_shared_informative_cache_ignores_process_only_new_candles(default_conf_usdt):
    strategy, _ = _shared_informative_call_count_strategy(default_conf_usdt)
    strategy.process_only_new_candles = False
    base_data = generate_test_data("5m", 40)

    strategy.advise_indicators(base_data.copy(), {"pair": "XRP/USDT"})
    strategy.advise_indicators(base_data.copy(), {"pair": "LTC/USDT"})

    # process_only_new_candles don't affect whether the informative data is cached or not
    assert len(strategy.informative_calls) == 6


@pytest.mark.parametrize("runmode", [RunMode.BACKTEST, RunMode.HYPEROPT])
def test_shared_informative_cache_disabled_outside_trade_modes(default_conf_usdt, runmode):
    default_conf_usdt["runmode"] = runmode
    strategy, _ = _shared_informative_call_count_strategy(default_conf_usdt)
    base_data = generate_test_data("5m", 40)

    strategy.advise_indicators(base_data.copy(), {"pair": "XRP/USDT"})
    strategy.advise_indicators(base_data.copy(), {"pair": "LTC/USDT"})

    # Cache only works in DRY_RUN and LIVE modes
    assert strategy._ft_informative_cache is None
    assert len(strategy.informative_calls) == 12
