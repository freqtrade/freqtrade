import pytest

from freqtrade.persistence import Trade, disable_database_use, enable_database_use
from freqtrade.persistence.custom_data import CustomDataWrapper
from tests.conftest import create_mock_trades_usdt


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
