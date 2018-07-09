# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest
from arrow import get as getdate

from freqtrade.optimize.backtesting import Backtesting
from freqtrade.tests.conftest import patch_exchange, log_has


columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'buy', 'sell']
data_profit = pd.DataFrame([[getdate('2018-07-08 18:00:00').datetime,
                             0.0009910, 0.001011, 0.00098618, 0.001000, 47027.0, 1, 0],
                            [getdate('2018-07-08 19:00:00').datetime,
                             0.001000, 0.001010, 0.0009900, 0.0009900, 87116.0, 0, 0],
                            [getdate('2018-07-08 20:00:00').datetime,
                             0.0009900, 0.001011, 0.00091618, 0.0009900, 58539.0, 0, 0],
                            [getdate('2018-07-08 21:00:00').datetime,
                             0.001000, 0.001011, 0.00098618, 0.001100,  37498.0, 0, 1],
                            [getdate('2018-07-08 22:00:00').datetime,
                             0.001000, 0.001011, 0.00098618, 0.0009900,  59792.0, 0, 0]],
                           columns=columns)

data_loss = pd.DataFrame([[getdate('2018-07-08 18:00:00').datetime,
                           0.0009910, 0.001011, 0.00098618, 0.001000, 47027.0, 1, 0],
                          [getdate('2018-07-08 19:00:00').datetime,
                           0.001000, 0.001010, 0.0009900, 0.001000, 87116.0, 0, 0],
                          [getdate('2018-07-08 20:00:00').datetime,
                           0.001000, 0.001011, 0.0010618, 0.00091618, 58539.0, 0, 0],
                          [getdate('2018-07-08 21:00:00').datetime,
                           0.001000, 0.001011, 0.00098618, 0.00091618,  37498.0, 0, 0],
                          [getdate('2018-07-08 22:00:00').datetime,
                           0.001000, 0.001011, 0.00098618, 0.00091618,  59792.0, 0, 0]],
                         columns=columns)


@pytest.mark.parametrize("data, stoploss, tradecount, profit_perc, sl", [
                         (data_profit, -0.01, 1, 0.10557, False),  # should be stoploss - drops 8%
                         # (data_profit, -0.10, 1, 0.10557, True),  # win
                         (data_loss, -0.05, 1, -0.08839, True),  # Stoploss ...
                         ])
def test_backtest_results(default_conf, fee, mocker, caplog,
                          data, stoploss, tradecount, profit_perc, sl) -> None:
    """
    run functional tests
    """
    default_conf["stoploss"] = stoploss
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch('freqtrade.analyze.Analyze.populate_sell_trend', MagicMock(return_value=data))
    mocker.patch('freqtrade.analyze.Analyze.populate_buy_trend', MagicMock(return_value=data))
    patch_exchange(mocker)

    backtesting = Backtesting(default_conf)
    caplog.set_level(logging.DEBUG)

    pair = 'UNITTEST/BTC'
    # Dummy data as we mock the analyze functions
    data_processed = {pair: pd.DataFrame()}
    results = backtesting.backtest(
        {
            'stake_amount': default_conf['stake_amount'],
            'processed': data_processed,
            'max_open_trades': 10,
            'realistic': True
        }
    )
    print(results.T)

    assert len(results) == tradecount
    assert round(results["profit_percent"].sum(), 5) == profit_perc
    if sl:
        assert log_has("Stop loss hit.", caplog.record_tuples)
    else:

        assert not log_has("Stop loss hit.", caplog.record_tuples)
