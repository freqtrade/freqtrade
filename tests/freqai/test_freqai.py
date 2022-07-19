# from unittest.mock import MagicMock
# from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_edge
import copy

import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
# from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.freqai.conftest import freqai_conf, get_patched_data_kitchen, get_patched_strategy


@pytest.mark.parametrize(
    "timerange, train_period_days, expected_result",
    [
        ("20220101-20220201", 30, "20211202-20220201"),
        ("20220301-20220401", 15, "20220214-20220401"),
    ],
)
def test_create_fulltimerange(
    timerange, train_period_days, expected_result, default_conf, mocker, caplog
):
    dk = get_patched_data_kitchen(mocker, freqai_conf(copy.deepcopy(default_conf)))
    assert dk.create_fulltimerange(timerange, train_period_days) == expected_result


def test_create_fulltimerange_incorrect_backtest_period(mocker, default_conf):
    dk = get_patched_data_kitchen(mocker, freqai_conf(copy.deepcopy(default_conf)))
    with pytest.raises(OperationalException, match=r"backtest_period_days must be an integer"):
        dk.create_fulltimerange("20220101-20220201", 0.5)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be positive"):
        dk.create_fulltimerange("20220101-20220201", -1)


def test_split_timerange(mocker, default_conf):
    freqaiconf = freqai_conf(copy.deepcopy(default_conf))
    freqaiconf.update({"timerange": "20220101-20220401"})
    dk = get_patched_data_kitchen(mocker, freqaiconf)
    tr_list, bt_list = dk.split_timerange("20220101-20220201", 30, 7)
    assert len(tr_list) == len(bt_list) == 9

    tr_list, bt_list = dk.split_timerange("20220101-20220201", 30, 0.5)
    assert len(tr_list) == len(bt_list) == 120

    tr_list, bt_list = dk.split_timerange("20220101-20220201", 10, 1)
    assert len(tr_list) == len(bt_list) == 80

    with pytest.raises(
        OperationalException, match=r"train_period_days must be an integer greater than 0."
    ):
        dk.split_timerange("20220101-20220201", -1, 0.5)


def test_update_historic_data(mocker, default_conf):
    freqaiconf = freqai_conf(copy.deepcopy(default_conf))
    strategy = get_patched_strategy(mocker, freqaiconf)
    exchange = get_patched_exchange(mocker, freqaiconf)
    strategy.dp = DataProvider(freqaiconf, exchange)
    freqai = strategy.model.bridge
    freqai.live = True
    freqai.dk = FreqaiDataKitchen(freqaiconf, freqai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")

    freqai.dk.load_all_pair_histories(timerange)
    historic_candles = len(freqai.dd.historic_data["ADA/BTC"]["5m"])
    dp_candles = len(strategy.dp.get_pair_dataframe("ADA/BTC", "5m"))
    candle_difference = dp_candles - historic_candles
    freqai.dk.update_historic_data(strategy)

    updated_historic_candles = len(freqai.dd.historic_data["ADA/BTC"]["5m"])

    assert updated_historic_candles - historic_candles == candle_difference


# def generate_test_data(timeframe: str, size: int, start: str = '2020-07-05'):
#     np.random.seed(42)
#     tf_mins = timeframe_to_minutes(timeframe)

#     base = np.random.normal(20, 2, size=size)

#     date = pd.date_range(start, periods=size, freq=f'{tf_mins}min', tz='UTC')
#     df = pd.DataFrame({
#         'date': date,
#         'open': base,
#         'high': base + np.random.normal(2, 1, size=size),
#         'low': base - np.random.normal(2, 1, size=size),
#         'close': base + np.random.normal(0, 1, size=size),
#         'volume': np.random.normal(200, size=size)
#     }
#     )
#     df = df.dropna()
#     return df
