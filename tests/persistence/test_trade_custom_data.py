from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from freqtrade.data.history.history_utils import get_timerange
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import Trade, disable_database_use, enable_database_use
from freqtrade.persistence.custom_data import CustomDataWrapper
from tests.conftest import (
    EXMS,
    create_mock_trades_usdt,
    generate_test_data,
    get_patched_freqtradebot,
    patch_exchange,
)


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("use_db", [True, False])
def test_trade_custom_data(fee, use_db):
    if not use_db:
        disable_database_use("5m")
    Trade.reset_trades()
    CustomDataWrapper.reset_custom_data()

    create_mock_trades_usdt(fee, use_db=use_db)

    trade1 = Trade.get_trades_proxy()[0]
    if not use_db:
        trade1.id = 1

    assert trade1.get_all_custom_data() == []
    trade1.set_custom_data("test_str", "test_value")
    trade1.set_custom_data("test_int", 1)
    trade1.set_custom_data("test_float", 1.55)
    trade1.set_custom_data("test_bool", True)
    trade1.set_custom_data("test_dict", {"test": "dict"})

    assert len(trade1.get_all_custom_data()) == 5
    assert trade1.get_custom_data("test_str") == "test_value"
    trade1.set_custom_data("test_str", "test_value_updated")
    assert trade1.get_custom_data("test_str") == "test_value_updated"

    assert trade1.get_custom_data("test_int") == 1
    assert isinstance(trade1.get_custom_data("test_int"), int)

    assert trade1.get_custom_data("test_float") == 1.55
    assert isinstance(trade1.get_custom_data("test_float"), float)

    assert trade1.get_custom_data("test_bool") is True
    assert isinstance(trade1.get_custom_data("test_bool"), bool)

    assert trade1.get_custom_data("test_dict") == {"test": "dict"}
    assert isinstance(trade1.get_custom_data("test_dict"), dict)
    if not use_db:
        enable_database_use()


def test_trade_custom_data_strategy_compat(mocker, default_conf_usdt, fee):
    mocker.patch(f"{EXMS}.get_rate", return_value=0.50)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.get_real_amount", return_value=None)
    default_conf_usdt["minimal_roi"] = {"0": 100}

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)

    trade1 = Trade.get_trades_proxy(pair="ADA/USDT")[0]
    trade1.set_custom_data("test_str", "test_value")
    trade1.set_custom_data("test_int", 1)

    def custom_exit(pair, trade, **kwargs):
        if pair == "ADA/USDT":
            custom_val = trade.get_custom_data("test_str")
            custom_val_i = trade.get_custom_data("test_int")

            return f"{custom_val}_{custom_val_i}"

    freqtrade.strategy.custom_exit = custom_exit
    ff_spy = mocker.spy(freqtrade.strategy, "custom_exit")
    trades = Trade.get_open_trades()
    freqtrade.exit_positions(trades)
    Trade.commit()

    trade_after = Trade.get_trades_proxy(pair="ADA/USDT")[0]
    assert trade_after.get_custom_data("test_str") == "test_value"
    assert trade_after.get_custom_data("test_int") == 1
    # 2 open pairs eligible for exit
    assert ff_spy.call_count == 2

    assert trade_after.exit_reason == "test_value_1"


def test_trade_custom_data_strategy_backtest_compat(mocker, default_conf_usdt, fee):
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=10)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_max_leverage", return_value=10)
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.1, 0.1))
    mocker.patch("freqtrade.optimize.backtesting.Backtesting._run_funding_fees")

    patch_exchange(mocker)
    default_conf_usdt.update(
        {
            "stake_amount": 100.0,
            "max_open_trades": 2,
            "dry_run_wallet": 1000.0,
            "strategy": "StrategyTestV3",
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "stoploss": -2,
            "minimal_roi": {"0": 100},
        }
    )
    default_conf_usdt["pairlists"] = [{"method": "StaticPairList", "allow_inactive": True}]
    backtesting = Backtesting(default_conf_usdt)

    df = generate_test_data(default_conf_usdt["timeframe"], 100, "2022-01-01 00:00:00+00:00")

    pair_exp = "XRP/USDT:USDT"

    def custom_exit(pair, trade, **kwargs):
        custom_val = trade.get_custom_data("test_str")
        custom_val_i = trade.get_custom_data("test_int", 0)

        if pair == pair_exp:
            trade.set_custom_data("test_str", "test_value")
            trade.set_custom_data("test_int", custom_val_i + 1)

        if custom_val_i >= 2:
            return f"{custom_val}_{custom_val_i}"

    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(
        {
            pair_exp: df,
            "BTC/USDT:USDT": df,
        }
    )

    def fun(dataframe, *args, **kwargs):
        dataframe.loc[dataframe.index == 50, "enter_long"] = 1
        return dataframe

    backtesting.strategy.advise_entry = fun
    backtesting.strategy.leverage = MagicMock(return_value=1)
    backtesting.strategy.custom_exit = custom_exit
    ff_spy = mocker.spy(backtesting.strategy, "custom_exit")

    min_date, max_date = get_timerange(processed)

    result = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results = result["results"]
    assert not results.empty
    assert len(results) == 2
    assert results["pair"][0] == pair_exp
    assert results["pair"][1] == "BTC/USDT:USDT"
    assert results["exit_reason"][0] == "test_value_2"
    assert results["exit_reason"][1] == "exit_signal"

    assert ff_spy.call_count == 7
    Backtesting.cleanup()
