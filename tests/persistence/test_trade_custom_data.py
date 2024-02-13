import pytest
from sqlalchemy import select

from freqtrade.persistence import Trade
from tests.conftest import create_mock_trades_usdt


@pytest.mark.usefixtures("init_persistence")
def test_trade_custom_data(fee):
    create_mock_trades_usdt(fee)

    trade1 = Trade.session.scalars(select(Trade)).first()

    assert trade1.get_all_custom_data() == []
    trade1.set_custom_data('test_str', 'test_value')
    trade1.set_custom_data('test_int', 1)
    trade1.set_custom_data('test_float', 1.55)
    trade1.set_custom_data('test_bool', True)
    trade1.set_custom_data('test_dict', {'test': 'dict'})

    assert trade1.get_custom_data('test_str') == 'test_value'

    assert trade1.get_custom_data('test_int') == 1
    assert isinstance(trade1.get_custom_data('test_int'), int)

    assert trade1.get_custom_data('test_float') == 1.55
    assert isinstance(trade1.get_custom_data('test_float'), float)

    assert trade1.get_custom_data('test_bool') is True
    assert isinstance(trade1.get_custom_data('test_bool'), bool)

    assert trade1.get_custom_data('test_dict') == {'test': 'dict'}
    assert isinstance(trade1.get_custom_data('test_dict'), dict)
    assert len(trade1.get_all_custom_data()) == 5
