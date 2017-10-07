# pragma pylint: disable=missing-docstring
from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade

def test_exec_sell_order(mocker):
    api_mock = mocker.patch('freqtrade.main.exchange.sell', side_effect='mocked_order_id')
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=1.00,
        open_rate=0.50,
        amount=10.00,
        exchange=Exchange.BITTREX,
        open_order_id='mocked'
    )
    profit = trade.exec_sell_order(1.00, 10.00)
    api_mock.assert_called_once_with('BTC_ETH', 1.0, 10.0)
    assert profit == 100.0
    assert trade.close_rate == 1.0
    assert trade.close_profit == profit
    assert trade.close_date is not None
