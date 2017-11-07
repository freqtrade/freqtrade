# pragma pylint: disable=missing-docstring
import copy
from unittest.mock import MagicMock, call

from freqtrade.exchange import Exchanges
from freqtrade.main import create_trade, handle_trade, close_trade_if_fulfilled, init, \
    get_target_bid
from freqtrade.persistence import Trade


def test_create_trade(default_conf, ticker, limit_buy_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    buy_signal = mocker.patch('freqtrade.main.get_buy_signal', side_effect=lambda _: True)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))
    # Save state of current whitelist
    whitelist = copy.deepcopy(default_conf['exchange']['pair_whitelist'])

    init(default_conf, 'sqlite://')
    for _ in ['BTC_ETH', 'BTC_TKN', 'BTC_TRST', 'BTC_SWT']:
        trade = create_trade(15.0)
        Trade.session.add(trade)
        Trade.session.flush()
        assert trade is not None
        assert trade.stake_amount == 15.0
        assert trade.is_open
        assert trade.open_date is not None
        assert trade.exchange == Exchanges.BITTREX.name

        # Simulate fulfilled LIMIT_BUY order for trade
        trade.update(limit_buy_order)

        assert trade.open_rate == 0.07256061
        assert trade.amount == 206.43811673387373

        assert whitelist == default_conf['exchange']['pair_whitelist']

    buy_signal.assert_has_calls(
        [call('BTC_ETH'), call('BTC_TKN'), call('BTC_TRST'), call('BTC_SWT')]
    )


def test_handle_trade(default_conf, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.main.telegram', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.17256061,
                              'ask': 0.172661,
                              'last': 0.17256061
                          }),
                          sell=MagicMock(return_value='mocked_limit_sell'))
    trade = Trade.query.filter(Trade.is_open.is_(True)).first()
    assert trade

    handle_trade(trade)
    assert trade.open_order_id == 'mocked_limit_sell'

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    assert trade.close_rate == 0.0802134
    assert trade.close_profit == 0.10046755
    assert trade.close_date is not None


def test_close_trade(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    trade = Trade.query.filter(Trade.is_open.is_(True)).first()
    assert trade

    # Simulate that there is no open order
    trade.open_order_id = None

    closed = close_trade_if_fulfilled(trade)
    assert closed
    assert not trade.is_open


def test_balance_fully_ask_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 0.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 20


def test_balance_fully_last_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 10


def test_balance_bigger_last_ask(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 5, 'last': 10}) == 5
