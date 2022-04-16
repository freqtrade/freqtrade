# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

from copy import deepcopy

import pandas as pd
from arrow import Arrow

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import ExitType
from freqtrade.optimize.backtesting import Backtesting
from tests.conftest import patch_exchange


def test_backtest_position_adjustment(default_conf, fee, mocker, testdatadir) -> None:
    default_conf['use_exit_signal'] = False
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch("freqtrade.exchange.Exchange.get_max_pair_stake_amount", return_value=float('inf'))
    patch_exchange(mocker)
    default_conf.update({
        "stake_amount": 100.0,
        "dry_run_wallet": 1000.0,
        "strategy": "StrategyTestV2"
    })
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'UNITTEST/BTC'
    timerange = TimeRange('date', None, 1517227800, 0)
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'],
                             timerange=timerange)
    backtesting.strategy.position_adjustment_enable = True
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    result = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
        max_open_trades=10,
        position_stacking=False,
    )
    results = result['results']
    assert not results.empty
    assert len(results) == 2

    expected = pd.DataFrame(
        {'pair': [pair, pair],
         'stake_amount': [500.0, 100.0],
         'amount': [4806.87657523, 970.63960782],
         'open_date': pd.to_datetime([Arrow(2018, 1, 29, 18, 40, 0).datetime,
                                      Arrow(2018, 1, 30, 3, 30, 0).datetime], utc=True
                                     ),
         'close_date': pd.to_datetime([Arrow(2018, 1, 29, 22, 00, 0).datetime,
                                       Arrow(2018, 1, 30, 4, 10, 0).datetime], utc=True),
         'open_rate': [0.10401764894444211, 0.10302485],
         'close_rate': [0.10453904066847439, 0.103541],
         'fee_open': [0.0025, 0.0025],
         'fee_close': [0.0025, 0.0025],
         'trade_duration': [200, 40],
         'profit_ratio': [0.0, 0.0],
         'profit_abs': [0.0, 0.0],
         'exit_reason': [ExitType.ROI.value, ExitType.ROI.value],
         'initial_stop_loss_abs': [0.0940005, 0.09272236],
         'initial_stop_loss_ratio': [-0.1, -0.1],
         'stop_loss_abs': [0.0940005, 0.09272236],
         'stop_loss_ratio': [-0.1, -0.1],
         'min_rate': [0.10370188, 0.10300000000000001],
         'max_rate': [0.10481985, 0.1038888],
         'is_open': [False, False],
         'enter_tag': [None, None],
         'is_short': [False, False],
         })
    pd.testing.assert_frame_equal(results, expected)
    data_pair = processed[pair]
    for _, t in results.iterrows():
        ln = data_pair.loc[data_pair["date"] == t["open_date"]]
        # Check open trade rate alignes to open rate
        assert ln is not None
        # check close trade rate alignes to close rate or is between high and low
        ln = data_pair.loc[data_pair["date"] == t["close_date"]]
        assert (round(ln.iloc[0]["open"], 6) == round(t["close_rate"], 6) or
                round(ln.iloc[0]["low"], 6) < round(
                t["close_rate"], 6) < round(ln.iloc[0]["high"], 6))
