import pytest

from freqtrade.persistence import Trade, disable_database_use, enable_database_use
from freqtrade.persistence.custom_data import CustomDataWrapper
from tests.conftest import EXMS, create_mock_trades_usdt, get_patched_freqtradebot


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("use_db", [True, False])
def test_trade_custom_data(fee, use_db):
    if not use_db:
        disable_database_use('5m')
    Trade.reset_trades()
    CustomDataWrapper.reset_custom_data()

    create_mock_trades_usdt(fee, use_db=use_db)

    trade1 = Trade.get_trades_proxy()[0]
    if not use_db:
        trade1.id = 1

    assert trade1.get_all_custom_data() == []
    trade1.set_custom_data('test_str', 'test_value')
    trade1.set_custom_data('test_int', 1)
    trade1.set_custom_data('test_float', 1.55)
    trade1.set_custom_data('test_bool', True)
    trade1.set_custom_data('test_dict', {'test': 'dict'})

    assert len(trade1.get_all_custom_data()) == 5
    assert trade1.get_custom_data('test_str') == 'test_value'
    trade1.set_custom_data('test_str', 'test_value_updated')
    assert trade1.get_custom_data('test_str') == 'test_value_updated'

    assert trade1.get_custom_data('test_int') == 1
    assert isinstance(trade1.get_custom_data('test_int'), int)

    assert trade1.get_custom_data('test_float') == 1.55
    assert isinstance(trade1.get_custom_data('test_float'), float)

    assert trade1.get_custom_data('test_bool') is True
    assert isinstance(trade1.get_custom_data('test_bool'), bool)

    assert trade1.get_custom_data('test_dict') == {'test': 'dict'}
    assert isinstance(trade1.get_custom_data('test_dict'), dict)
    enable_database_use()


def test_trade_custom_data_strategy_compat(mocker, default_conf_usdt, fee):

    mocker.patch(f'{EXMS}.get_rate', return_value=0.50)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=None)
    default_conf_usdt["minimal_roi"] = {
        "0":  100
    }

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)

    trade1 = Trade.get_trades_proxy(pair='ADA/USDT')[0]
    trade1.set_custom_data('test_str', 'test_value')
    trade1.set_custom_data('test_int', 1)

    def custom_exit(pair, trade, **kwargs):

        if pair == 'ADA/USDT':
            custom_val = trade.get_custom_data('test_str')
            custom_val_i = trade.get_custom_data('test_int')

            return f"{custom_val}_{custom_val_i}"

    freqtrade.strategy.custom_exit = custom_exit
    ff_spy = mocker.spy(freqtrade.strategy, 'custom_exit')
    trades = Trade.get_open_trades()
    freqtrade.exit_positions(trades)
    Trade.commit()

    trade_after = Trade.get_trades_proxy(pair='ADA/USDT')[0]
    assert trade_after.get_custom_data('test_str') == 'test_value'
    assert trade_after.get_custom_data('test_int') == 1
    # 2 open pairs eligible for exit
    assert ff_spy.call_count == 2

    assert trade_after.exit_reason == 'test_value_1'
