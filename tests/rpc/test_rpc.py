from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
from numpy import isnan
from sqlalchemy import select

from freqtrade.edge import PairInfo
from freqtrade.enums import SignalDirection, State, TradingMode
from freqtrade.exceptions import ExchangeError, InvalidOrderException, TemporaryError
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.key_value_store import set_startup_time
from freqtrade.rpc import RPC, RPCException
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from tests.conftest import (EXMS, create_mock_trades, create_mock_trades_usdt,
                            get_patched_freqtradebot, patch_get_signal)


def test_rpc_trade_status(default_conf, ticker, fee, mocker) -> None:
    gen_response = {
        'trade_id': 1,
        'pair': 'ETH/BTC',
        'base_currency': 'ETH',
        'quote_currency': 'BTC',
        'open_date': ANY,
        'open_timestamp': ANY,
        'open_fill_date': ANY,
        'open_fill_timestamp': ANY,
        'is_open': ANY,
        'fee_open': ANY,
        'fee_open_cost': ANY,
        'fee_open_currency': ANY,
        'fee_close': fee.return_value,
        'fee_close_cost': ANY,
        'fee_close_currency': ANY,
        'open_rate_requested': ANY,
        'open_trade_value': 0.0010025,
        'close_rate_requested': ANY,
        'exit_reason': ANY,
        'exit_order_status': ANY,
        'min_rate': ANY,
        'max_rate': ANY,
        'strategy': ANY,
        'enter_tag': ANY,
        'timeframe': 5,
        'close_date': None,
        'close_timestamp': None,
        'open_rate': 1.098e-05,
        'close_rate': None,
        'current_rate': 1.099e-05,
        'amount': 91.07468123,
        'amount_requested': 91.07468124,
        'stake_amount': 0.001,
        'max_stake_amount': None,
        'trade_duration': None,
        'trade_duration_s': None,
        'close_profit': None,
        'close_profit_pct': None,
        'close_profit_abs': None,
        'profit_ratio': -0.00408133,
        'profit_pct': -0.41,
        'profit_abs': -4.09e-06,
        'profit_fiat': ANY,
        'stop_loss_abs': 9.89e-06,
        'stop_loss_pct': -10.0,
        'stop_loss_ratio': -0.1,
        'stoploss_last_update': ANY,
        'stoploss_last_update_timestamp': ANY,
        'initial_stop_loss_abs': 9.89e-06,
        'initial_stop_loss_pct': -10.0,
        'initial_stop_loss_ratio': -0.1,
        'stoploss_current_dist': pytest.approx(-1.0999999e-06),
        'stoploss_current_dist_ratio': -0.10009099,
        'stoploss_current_dist_pct': -10.01,
        'stoploss_entry_dist': -0.00010402,
        'stoploss_entry_dist_ratio': -0.10376381,
        'open_orders': '',
        'realized_profit': 0.0,
        'realized_profit_ratio': None,
        'total_profit_abs': -4.09e-06,
        'total_profit_fiat': ANY,
        'total_profit_ratio': None,
        'exchange': 'binance',
        'leverage': 1.0,
        'interest_rate': 0.0,
        'liquidation_price': None,
        'is_short': False,
        'funding_fees': 0.0,
        'trading_mode': TradingMode.SPOT,
        'amount_precision': 8.0,
        'price_precision': 8.0,
        'precision_mode': 2,
        'contract_size': 1,
        'has_open_orders': False,
        'orders': [{
            'amount': 91.07468123, 'average': 1.098e-05, 'safe_price': 1.098e-05,
            'cost': 0.0009999999999054, 'filled': 91.07468123, 'ft_order_side': 'buy',
            'order_date': ANY, 'order_timestamp': ANY, 'order_filled_date': ANY,
            'order_filled_timestamp': ANY, 'order_type': 'limit', 'price': 1.098e-05,
            'is_open': False, 'pair': 'ETH/BTC', 'order_id': ANY,
            'remaining': ANY, 'status': ANY, 'ft_is_entry': True, 'ft_fee_base': None,
            'funding_fee': ANY, 'ft_order_tag': None,
        }],
    }
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(side_effect=[False, True]),
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active trade*'):
        rpc._rpc_trade_status()

    freqtradebot.enter_positions()

    # Open order...
    results = rpc._rpc_trade_status()
    response_unfilled = deepcopy(gen_response)
    # Different from "filled" response:
    response_unfilled.update({
        'amount': 91.07468124,
        'profit_ratio': 0.0,
        'profit_pct': 0.0,
        'profit_abs': 0.0,
        'total_profit_abs': 0.0,
        'open_orders': '(limit buy rem=91.07468123)',
        'has_open_orders': True,
    })
    response_unfilled['orders'][0].update({
        'is_open': True,
        'filled': 0.0,
        'remaining': 91.07468123
    })
    assert results[0] == response_unfilled

    # Open order without remaining
    trade = Trade.get_open_trades()[0]
    # kucoin case (no remaining set).
    trade.orders[0].remaining = None
    Trade.commit()

    results = rpc._rpc_trade_status()
    # Reuse above object, only remaining changed.
    response_unfilled['orders'][0].update({
        'remaining': None,
    })
    assert results[0] == response_unfilled

    trade = Trade.get_open_trades()[0]
    trade.orders[0].remaining = trade.amount
    Trade.commit()

    # Fill open order ...
    freqtradebot.manage_open_orders()
    trades = Trade.get_open_trades()
    freqtradebot.exit_positions(trades)

    results = rpc._rpc_trade_status()

    response = deepcopy(gen_response)
    response.update({
        'max_stake_amount': 0.001,
        'total_profit_ratio': pytest.approx(-0.00409153),
        'has_open_orders': False,
    })
    assert results[0] == response

    mocker.patch(f'{EXMS}.get_rate',
                 MagicMock(side_effect=ExchangeError("Pair 'ETH/BTC' not available")))
    results = rpc._rpc_trade_status()
    assert isnan(results[0]['profit_ratio'])
    assert isnan(results[0]['current_rate'])
    response_norate = deepcopy(gen_response)
    # Update elements that are NaN when no rate is available.
    response_norate.update({
        'stoploss_current_dist': ANY,
        'stoploss_current_dist_ratio': ANY,
        'stoploss_current_dist_pct': ANY,
        'max_stake_amount': 0.001,
        'profit_ratio': ANY,
        'profit_pct': ANY,
        'profit_abs': ANY,
        'total_profit_abs': ANY,
        'total_profit_ratio': ANY,
        'current_rate': ANY,
    })
    assert results[0] == response_norate


def test_rpc_status_table(default_conf, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.CoinGeckoAPI',
        get_price=MagicMock(return_value={'bitcoin': {'usd': 15000.0}}),
    )
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )
    del default_conf['fiat_display_currency']
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active trade*'):
        rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=False)
    freqtradebot.enter_positions()

    result, headers, fiat_profit_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert "Since" in headers
    assert "Pair" in headers
    assert 'instantly' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert '0.00 (0.00)' == result[0][3]
    assert '0.00' == f'{fiat_profit_sum:.2f}'

    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=True)
    freqtradebot.process()

    result, headers, fiat_profit_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert "Since" in headers
    assert "Pair" in headers
    assert 'instantly' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert '-0.41% (-0.00)' == result[0][3]
    assert '-0.00' == f'{fiat_profit_sum:.2f}'

    # Test with fiat convert
    rpc._fiat_converter = CryptoToFiatConverter()
    result, headers, fiat_profit_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert "Since" in headers
    assert "Pair" in headers
    assert len(result[0]) == 4
    assert 'instantly' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert '-0.41% (-0.06)' == result[0][3]
    assert '-0.06' == f'{fiat_profit_sum:.2f}'

    rpc._config['position_adjustment_enable'] = True
    rpc._config['max_entry_position_adjustment'] = 3
    result, headers, fiat_profit_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert "# Entries" in headers
    assert len(result[0]) == 5
    # 4th column should be 1/4 - as 1 order filled (a total of 4 is possible)
    # 3 on top of the initial one.
    assert result[0][4] == '1/4'

    mocker.patch(f'{EXMS}.get_rate',
                 MagicMock(side_effect=ExchangeError("Pair 'ETH/BTC' not available")))
    result, headers, fiat_profit_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert 'instantly' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert 'nan%' == result[0][3]
    assert isnan(fiat_profit_sum)


def test__rpc_timeunit_profit(
        default_conf_usdt, ticker, fee, markets, mocker, time_machine) -> None:

    time_machine.move_to("2023-09-05 10:00:00 +00:00", tick=False)

    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)

    stake_currency = default_conf_usdt['stake_currency']
    fiat_display_currency = default_conf_usdt['fiat_display_currency']

    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()

    # Try valid data
    days = rpc._rpc_timeunit_profit(7, stake_currency, fiat_display_currency)
    assert len(days['data']) == 7
    assert days['stake_currency'] == default_conf_usdt['stake_currency']
    assert days['fiat_display_currency'] == default_conf_usdt['fiat_display_currency']
    for day in days['data']:
        # {'date': datetime.date(2022, 6, 11), 'abs_profit': 13.8299999,
        #  'starting_balance': 1055.37, 'rel_profit': 0.0131044,
        #  'fiat_value': 0.0, 'trade_count': 2}
        assert day['abs_profit'] in (0.0, pytest.approx(6.83), pytest.approx(-4.09))
        assert day['rel_profit'] in (0.0, pytest.approx(0.00642902), pytest.approx(-0.00383512))
        assert day['trade_count'] in (0, 1, 2)
        assert day['starting_balance'] in (pytest.approx(1062.37), pytest.approx(1066.46))
        assert day['fiat_value'] in (0.0, )
    # ensure first day is current date
    assert str(days['data'][0]['date']) == str(datetime.now(timezone.utc).date())

    # Try invalid data
    with pytest.raises(RPCException, match=r'.*must be an integer greater than 0*'):
        rpc._rpc_timeunit_profit(0, stake_currency, fiat_display_currency)


@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_trade_history(mocker, default_conf, markets, fee, is_short):
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    create_mock_trades(fee, is_short)
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()
    trades = rpc._rpc_trade_history(2)
    assert len(trades['trades']) == 2
    assert trades['trades_count'] == 2
    assert isinstance(trades['trades'][0], dict)
    assert isinstance(trades['trades'][1], dict)

    trades = rpc._rpc_trade_history(0)
    assert len(trades['trades']) == 2
    assert trades['trades_count'] == 2
    # The first closed trade is for ETC ... sorting is descending
    assert trades['trades'][-1]['pair'] == 'ETC/BTC'
    assert trades['trades'][0]['pair'] == 'XRP/BTC'


@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_delete_trade(mocker, default_conf, fee, markets, caplog, is_short):
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        cancel_order=cancel_mock,
        cancel_stoploss_order=stoploss_mock,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    freqtradebot.strategy.order_types['stoploss_on_exchange'] = True
    create_mock_trades(fee, is_short)
    rpc = RPC(freqtradebot)
    with pytest.raises(RPCException, match='invalid argument'):
        rpc._rpc_delete('200')

    trades = Trade.session.scalars(select(Trade)).all()
    trades[2].orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trades[2].pair,
            ft_is_open=True,
            ft_amount=trades[2].amount,
            ft_price=trades[2].stop_loss,
            order_id='102',
            status='open',
        )
    )
    assert len(trades) > 2

    res = rpc._rpc_delete('1')
    assert isinstance(res, dict)
    assert res['result'] == 'success'
    assert res['trade_id'] == '1'
    assert res['cancel_order_count'] == 1
    assert cancel_mock.call_count == 1
    assert stoploss_mock.call_count == 0
    cancel_mock.reset_mock()
    stoploss_mock.reset_mock()

    res = rpc._rpc_delete('5')
    assert isinstance(res, dict)
    assert stoploss_mock.call_count == 1
    assert res['cancel_order_count'] == 1

    stoploss_mock = mocker.patch(f'{EXMS}.cancel_stoploss_order', side_effect=InvalidOrderException)

    res = rpc._rpc_delete('3')
    assert stoploss_mock.call_count == 1
    stoploss_mock.reset_mock()

    cancel_mock = mocker.patch(f'{EXMS}.cancel_order', side_effect=InvalidOrderException)

    res = rpc._rpc_delete('4')
    assert cancel_mock.call_count == 1
    assert stoploss_mock.call_count == 0


def test_rpc_trade_statistics(default_conf_usdt, ticker, fee, mocker) -> None:
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=1.1)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf_usdt)
    stake_currency = default_conf_usdt['stake_currency']
    fiat_display_currency = default_conf_usdt['fiat_display_currency']

    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()

    res = rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert res['trade_count'] == 0
    assert res['first_trade_date'] == ''
    assert res['first_trade_timestamp'] == 0
    assert res['latest_trade_date'] == ''
    assert res['latest_trade_timestamp'] == 0
    assert res['expectancy'] == 0
    assert res['expectancy_ratio'] == 100

    # Create some test data
    create_mock_trades_usdt(fee)

    stats = rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert pytest.approx(stats['profit_closed_coin']) == 2.74
    assert pytest.approx(stats['profit_closed_percent_mean']) == -1.67
    assert pytest.approx(stats['profit_closed_fiat']) == 3.014
    assert pytest.approx(stats['profit_all_coin']) == -77.45964918
    assert pytest.approx(stats['profit_all_percent_mean']) == -57.86
    assert pytest.approx(stats['profit_all_fiat']) == -85.205614098
    assert pytest.approx(stats['winrate']) == 0.666666667
    assert pytest.approx(stats['expectancy']) == 0.913333333
    assert pytest.approx(stats['expectancy_ratio']) == 0.223308883
    assert stats['trade_count'] == 7
    assert stats['first_trade_humanized'] == '2 days ago'
    assert stats['latest_trade_humanized'] == '17 minutes ago'
    assert stats['avg_duration'] in ('0:17:40')
    assert stats['best_pair'] == 'XRP/USDT'
    assert stats['best_rate'] == 10.0

    # Test non-available pair
    mocker.patch(f'{EXMS}.get_rate',
                 MagicMock(side_effect=ExchangeError("Pair 'XRP/USDT' not available")))
    stats = rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert stats['trade_count'] == 7
    assert stats['first_trade_humanized'] == '2 days ago'
    assert stats['latest_trade_humanized'] == '17 minutes ago'
    assert stats['avg_duration'] in ('0:17:40')
    assert stats['best_pair'] == 'XRP/USDT'
    assert stats['best_rate'] == 10.0
    assert isnan(stats['profit_all_coin'])


def test_rpc_balance_handle_error(default_conf, mocker):
    mock_balance = {
        'BTC': {
            'free': 10.0,
            'total': 12.0,
            'used': 2.0,
        },
        'ETH': {
            'free': 1.0,
            'total': 5.0,
            'used': 4.0,
        }
    }
    # ETH will be skipped due to mocked Error below

    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.CoinGeckoAPI',
        get_price=MagicMock(return_value={'bitcoin': {'usd': 15000.0}}),
    )
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=mock_balance),
        get_tickers=MagicMock(side_effect=TemporaryError('Could not load ticker due to xxx'))
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()
    with pytest.raises(RPCException, match="Error getting current tickers."):
        rpc._rpc_balance(default_conf['stake_currency'], default_conf['fiat_display_currency'])


def test_rpc_balance_handle(default_conf_usdt, mocker, tickers):
    mock_balance = {
        'BTC': {
            'free': 0.01,
            'total': 0.012,
            'used': 0.002,
        },
        'ETH': {
            'free': 1.0,
            'total': 5.0,
            'used': 4.0,
        },
        'USDT': {
            'free': 50.0,
            'total': 100.0,
            'used': 5.0,
        }
    }
    mock_pos = [
        {
            "symbol": "ETH/USDT:USDT",
            "timestamp": None,
            "datetime": None,
            "initialMargin": 0.0,
            "initialMarginPercentage": None,
            "maintenanceMargin": 0.0,
            "maintenanceMarginPercentage": 0.005,
            "entryPrice": 0.0,
            "notional": 10.0,
            "leverage": 5.0,
            "unrealizedPnl": 0.0,
            "contracts": 1.0,
            "contractSize": 1,
            "marginRatio": None,
            "liquidationPrice": 0.0,
            "markPrice": 2896.41,
            "collateral": 20,
            "marginType": "isolated",
            "side": 'short',
            "percentage": None
        }
    ]

    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.CoinGeckoAPI',
        get_price=MagicMock(return_value={'bitcoin': {'usd': 1.2}}),
    )
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=1.2)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        validate_trading_mode_and_margin_mode=MagicMock(),
        get_balances=MagicMock(return_value=mock_balance),
        fetch_positions=MagicMock(return_value=mock_pos),
        get_tickers=tickers,
        get_valid_pair_combination=MagicMock(
            side_effect=lambda a, b: f"{b}/{a}" if a == "USDT" else f"{a}/{b}")
    )
    default_conf_usdt['dry_run'] = False
    default_conf_usdt['trading_mode'] = 'futures'
    freqtradebot = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()

    result = rpc._rpc_balance(
        default_conf_usdt['stake_currency'], default_conf_usdt['fiat_display_currency'])

    assert pytest.approx(result['total']) == 2824.83464
    assert pytest.approx(result['value']) == 2824.83464 * 1.2
    assert tickers.call_count == 1
    assert tickers.call_args_list[0][1]['cached'] is True
    assert 'USD' == result['symbol']
    assert result['currencies'] == [
        {
            'currency': 'BTC',
            'free': 0.01,
            'balance': 0.012,
            'used': 0.002,
            'bot_owned': 0,
            'est_stake': 103.78464,
            'est_stake_bot': 0,
            'stake': 'USDT',
            'side': 'long',
            'leverage': 1,
            'position': 0,
            'is_bot_managed': False,
            'is_position': False
        },
        {
            'currency': 'ETH',
            'free': 1.0,
            'balance': 5.0,
            'used': 4.0,
            'bot_owned': 0,
            'est_stake': 2651.05,
            'est_stake_bot': 0,
            'stake': 'USDT',
            'side': 'long',
            'leverage': 1,
            'position': 0,
            'is_bot_managed': False,
            'is_position': False
        },
        {
            'currency': 'USDT',
            'free': 50.0,
            'balance': 100.0,
            'used': 5.0,
            'bot_owned': 49.5,
            'est_stake': 50.0,
            'est_stake_bot': 49.5,
            'stake': 'USDT',
            'side': 'long',
            'leverage': 1,
            'position': 0,
            'is_bot_managed': True,
            'is_position': False
        },
        {
            'currency': 'ETH/USDT:USDT',
            'free': 0,
            'balance': 0,
            'used': 0,
            'position': 10.0,
            'est_stake': 20,
            'est_stake_bot': 20,
            'stake': 'USDT',
            'leverage': 5.0,
            'side': 'short',
            'is_bot_managed': True,
            'is_position': True
        }
    ]
    assert pytest.approx(result['total_bot']) == 69.5
    assert pytest.approx(result['total']) == 2824.83464  # ETH stake is missing.
    assert result['starting_capital'] == 50
    assert result['starting_capital_ratio'] == pytest.approx(0.3899999)


def test_rpc_start(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock()
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.STOPPED

    result = rpc._rpc_start()
    assert {'status': 'starting trader ...'} == result
    assert freqtradebot.state == State.RUNNING

    result = rpc._rpc_start()
    assert {'status': 'already running'} == result
    assert freqtradebot.state == State.RUNNING


def test_rpc_stop(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock()
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING

    result = rpc._rpc_stop()
    assert {'status': 'stopping trader ...'} == result
    assert freqtradebot.state == State.STOPPED

    result = rpc._rpc_stop()

    assert {'status': 'already stopped'} == result
    assert freqtradebot.state == State.STOPPED


def test_rpc_stopentry(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock()
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING

    assert freqtradebot.config['max_open_trades'] != 0
    result = rpc._rpc_stopentry()
    assert {'status': 'No more entries will occur from now. Run /reload_config to reset.'} == result
    assert freqtradebot.config['max_open_trades'] == 0


def test_rpc_force_exit(default_conf, ticker, fee, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker,
        cancel_order=cancel_order_mock,
        fetch_order=MagicMock(
            return_value={
                'status': 'closed',
                'type': 'limit',
                'side': 'buy',
                'filled': 0.0,
            }
        ),
        _dry_is_price_crossed=MagicMock(return_value=True),
        get_fee=fee,
    )
    mocker.patch('freqtrade.wallets.Wallets.get_free', return_value=1000)

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_force_exit(None)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*invalid argument*'):
        rpc._rpc_force_exit(None)

    msg = rpc._rpc_force_exit('all')
    assert msg == {'result': 'Created exit orders for all open trades.'}

    freqtradebot.enter_positions()
    msg = rpc._rpc_force_exit('all')
    assert msg == {'result': 'Created exit orders for all open trades.'}

    freqtradebot.enter_positions()
    msg = rpc._rpc_force_exit('2')
    assert msg == {'result': 'Created exit order for trade 2.'}

    freqtradebot.state = State.STOPPED
    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_force_exit(None)

    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_force_exit('all')

    freqtradebot.state = State.RUNNING
    assert cancel_order_mock.call_count == 0
    mocker.patch(f'{EXMS}._dry_is_price_crossed', MagicMock(return_value=False))
    freqtradebot.enter_positions()
    # make an limit-buy open trade
    trade = Trade.session.scalars(select(Trade).filter(Trade.id == '3')).first()
    filled_amount = trade.amount / 2
    # Fetch order - it's open first, and closed after cancel_order is called.
    mocker.patch(
        f'{EXMS}.fetch_order',
        side_effect=[{
            'id': trade.orders[0].order_id,
            'status': 'open',
            'type': 'limit',
            'side': 'buy',
            'filled': filled_amount
        }, {
            'id': trade.orders[0].order_id,
            'status': 'closed',
            'type': 'limit',
            'side': 'buy',
            'filled': filled_amount
        }]
    )
    # check that the trade is called, which is done by ensuring exchange.cancel_order is called
    # and trade amount is updated
    rpc._rpc_force_exit('3')
    assert cancel_order_mock.call_count == 1
    assert pytest.approx(trade.amount) == filled_amount

    mocker.patch(
        f'{EXMS}.fetch_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'buy',
            'filled': filled_amount
        })

    freqtradebot.config['max_open_trades'] = 3
    freqtradebot.enter_positions()

    cancel_order_mock.reset_mock()
    trade = Trade.session.scalars(select(Trade).filter(Trade.id == '3')).first()
    amount = trade.amount
    # make an limit-sell open order trade
    mocker.patch(
        f'{EXMS}.fetch_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'sell',
            'amount': amount,
            'remaining': amount,
            'filled': 0.0,
            'id': trade.orders[-1].order_id,
        }
    )
    cancel_order_3 = mocker.patch(
        f'{EXMS}.cancel_order_with_result',
        return_value={
            'status': 'canceled',
            'type': 'limit',
            'side': 'sell',
            'amount': amount,
            'remaining': amount,
            'filled': 0.0,
            'id': trade.orders[-1].order_id,
        }
    )
    msg = rpc._rpc_force_exit('3')
    assert msg == {'result': 'Created exit order for trade 3.'}
    # status quo, no exchange calls
    assert cancel_order_3.call_count == 1
    assert cancel_order_mock.call_count == 0

    trade = Trade.session.scalars(select(Trade).filter(Trade.id == '4')).first()
    amount = trade.amount
    # make an limit-buy open trade, if there is no 'filled', don't sell it
    mocker.patch(
        f'{EXMS}.fetch_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'buy',
            'filled': None
        }
    )
    cancel_order_4 = mocker.patch(
            f'{EXMS}.cancel_order_with_result',
            return_value={
                'status': 'canceled',
                'type': 'limit',
                'side': 'sell',
                'amount': amount,
                'remaining': 0.0,
                'filled': amount,
                'id': trade.orders[0].order_id,
                }
            )
    # check that the trade is called, which is done by ensuring exchange.cancel_order is called
    msg = rpc._rpc_force_exit('4')
    assert msg == {'result': 'Created exit order for trade 4.'}
    assert cancel_order_4.call_count == 1
    assert cancel_order_mock.call_count == 0
    assert pytest.approx(trade.amount) == amount


def test_performance_handle(default_conf_usdt, ticker, fee, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    # Create some test data
    create_mock_trades_usdt(fee)

    res = rpc._rpc_performance()
    assert len(res) == 3
    assert res[0]['pair'] == 'NEO/USDT'
    assert res[0]['count'] == 1
    assert res[0]['profit_pct'] == 5.0


def test_enter_tag_performance_handle(default_conf, ticker, fee, mocker) -> None:

    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)

    # Create some test data
    create_mock_trades_usdt(fee)
    freqtradebot.enter_positions()

    res = rpc._rpc_enter_tag_performance(None)

    assert len(res) == 3
    assert res[0]['enter_tag'] == 'TEST1'
    assert res[0]['count'] == 1
    assert res[0]['profit_pct'] == 5.0

    res = rpc._rpc_enter_tag_performance(None)

    assert len(res) == 3
    assert res[0]['enter_tag'] == 'TEST1'
    assert res[0]['count'] == 1
    assert res[0]['profit_pct'] == 5.0


def test_enter_tag_performance_handle_2(mocker, default_conf, markets, fee):
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    create_mock_trades(fee)
    rpc = RPC(freqtradebot)

    res = rpc._rpc_enter_tag_performance(None)

    assert len(res) == 2
    assert res[0]['enter_tag'] == 'TEST1'
    assert res[0]['count'] == 1
    assert pytest.approx(res[0]['profit_pct']) == 0.5
    assert res[1]['enter_tag'] == 'Other'
    assert res[1]['count'] == 1
    assert pytest.approx(res[1]['profit_pct']) == 1.0

    # Test for a specific pair
    res = rpc._rpc_enter_tag_performance('ETC/BTC')
    assert len(res) == 1
    assert res[0]['count'] == 1
    assert res[0]['enter_tag'] == 'TEST1'
    assert pytest.approx(res[0]['profit_pct']) == 0.5


def test_exit_reason_performance_handle(default_conf_usdt, ticker, fee, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)

    # Create some test data
    create_mock_trades_usdt(fee)

    res = rpc._rpc_exit_reason_performance(None)

    assert len(res) == 3
    assert res[0]['exit_reason'] == 'exit_signal'
    assert res[0]['count'] == 1
    assert res[0]['profit_pct'] == 5.0

    assert res[1]['exit_reason'] == 'roi'
    assert res[2]['exit_reason'] == 'Other'


def test_exit_reason_performance_handle_2(mocker, default_conf, markets, fee):
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    create_mock_trades(fee)
    rpc = RPC(freqtradebot)

    res = rpc._rpc_exit_reason_performance(None)

    assert len(res) == 2
    assert res[0]['exit_reason'] == 'sell_signal'
    assert res[0]['count'] == 1
    assert pytest.approx(res[0]['profit_pct']) == 0.5
    assert res[1]['exit_reason'] == 'roi'
    assert res[1]['count'] == 1
    assert pytest.approx(res[1]['profit_pct']) == 1.0

    # Test for a specific pair
    res = rpc._rpc_exit_reason_performance('ETC/BTC')
    assert len(res) == 1
    assert res[0]['count'] == 1
    assert res[0]['exit_reason'] == 'sell_signal'
    assert pytest.approx(res[0]['profit_pct']) == 0.5


def test_mix_tag_performance_handle(default_conf, ticker, fee, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)

    # Create some test data
    create_mock_trades_usdt(fee)

    res = rpc._rpc_mix_tag_performance(None)

    assert len(res) == 3
    assert res[0]['mix_tag'] == 'TEST1 exit_signal'
    assert res[0]['count'] == 1
    assert res[0]['profit_pct'] == 5.0


def test_mix_tag_performance_handle_2(mocker, default_conf, markets, fee):
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    create_mock_trades(fee)
    rpc = RPC(freqtradebot)

    res = rpc._rpc_mix_tag_performance(None)

    assert len(res) == 2
    assert res[0]['mix_tag'] == 'TEST1 sell_signal'
    assert res[0]['count'] == 1
    assert pytest.approx(res[0]['profit_pct']) == 0.5
    assert res[1]['mix_tag'] == 'Other roi'
    assert res[1]['count'] == 1
    assert pytest.approx(res[1]['profit_pct']) == 1.0

    # Test for a specific pair
    res = rpc._rpc_mix_tag_performance('ETC/BTC')

    assert len(res) == 1
    assert res[0]['count'] == 1
    assert res[0]['mix_tag'] == 'TEST1 sell_signal'
    assert pytest.approx(res[0]['profit_pct']) == 0.5


def test_rpc_count(mocker, default_conf, ticker, fee) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)

    counts = rpc._rpc_count()
    assert counts["current"] == 0

    # Create some test data
    freqtradebot.enter_positions()
    counts = rpc._rpc_count()
    assert counts["current"] == 1


def test_rpc_force_entry(mocker, default_conf, ticker, fee, limit_buy_order_open) -> None:
    default_conf['force_entry_enable'] = True
    default_conf['max_open_trades'] = 0
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    buy_mm = MagicMock(return_value=limit_buy_order_open)
    mocker.patch.multiple(
        EXMS,
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        create_order=buy_mm
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match='Maximum number of trades is reached.'):
        rpc._rpc_force_entry(pair, None)
    freqtradebot.config['max_open_trades'] = 5

    trade = rpc._rpc_force_entry(pair, None)
    assert isinstance(trade, Trade)
    assert trade.pair == pair
    assert trade.open_rate == ticker()['bid']

    # Test buy duplicate
    with pytest.raises(RPCException, match=r'position for ETH/BTC already open - id: 1'):
        rpc._rpc_force_entry(pair, 0.0001)
    pair = 'XRP/BTC'
    trade = rpc._rpc_force_entry(pair, 0.0001, order_type='limit')
    assert isinstance(trade, Trade)
    assert trade.pair == pair
    assert trade.open_rate == 0.0001

    with pytest.raises(RPCException,
                       match=r'Symbol does not exist or market is not active.'):
        rpc._rpc_force_entry('LTC/NOTHING', 0.0001)

    # Test buy pair not with stakes
    with pytest.raises(RPCException,
                       match=r'Wrong pair selected. Only pairs with stake-currency.*'):
        rpc._rpc_force_entry('LTC/ETH', 0.0001)

    # Test with defined stake_amount
    pair = 'LTC/BTC'
    trade = rpc._rpc_force_entry(pair, 0.0001, order_type='limit', stake_amount=0.05)
    assert trade.stake_amount == 0.05
    assert trade.buy_tag == 'force_entry'

    assert trade.open_orders_ids[-1] == 'mocked_limit_buy'

    freqtradebot.strategy.position_adjustment_enable = True
    with pytest.raises(RPCException, match=r'position for LTC/BTC already open.*open order.*'):
        rpc._rpc_force_entry(pair, 0.0001, order_type='limit', stake_amount=0.05)

    # Test not buying
    pair = 'XRP/BTC'
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    freqtradebot.config['stake_amount'] = 0
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    pair = 'TKN/BTC'
    with pytest.raises(RPCException, match=r"Failed to enter position for TKN/BTC."):
        trade = rpc._rpc_force_entry(pair, None)


def test_rpc_force_entry_stopped(mocker, default_conf) -> None:
    default_conf['force_entry_enable'] = True
    default_conf['initial_state'] = 'stopped'
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match=r'trader is not running'):
        rpc._rpc_force_entry(pair, None)


def test_rpc_force_entry_disabled(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match=r'Force_entry not enabled.'):
        rpc._rpc_force_entry(pair, None)


def test_rpc_force_entry_wrong_mode(mocker, default_conf) -> None:
    default_conf['force_entry_enable'] = True
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match="Can't go short on Spot markets."):
        rpc._rpc_force_entry(pair, None, order_side=SignalDirection.SHORT)


@pytest.mark.usefixtures("init_persistence")
def test_rpc_add_and_delete_lock(mocker, default_conf):
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'

    rpc._rpc_add_lock(pair, datetime.now(timezone.utc) + timedelta(minutes=4), '', '*')
    rpc._rpc_add_lock(pair, datetime.now(timezone.utc) + timedelta(minutes=5), '', '*')
    rpc._rpc_add_lock(pair, datetime.now(timezone.utc) + timedelta(minutes=10), '', '*')

    locks = rpc._rpc_locks()
    assert locks['lock_count'] == 3
    locks1 = rpc._rpc_delete_lock(lockid=locks['locks'][0]['id'])
    assert locks1['lock_count'] == 2

    locks2 = rpc._rpc_delete_lock(pair=pair)
    assert locks2['lock_count'] == 0


def test_rpc_whitelist(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    ret = rpc._rpc_whitelist()
    assert len(ret['method']) == 1
    assert 'StaticPairList' in ret['method']
    assert ret['whitelist'] == default_conf['exchange']['pair_whitelist']


def test_rpc_whitelist_dynamic(mocker, default_conf) -> None:
    default_conf['pairlists'] = [{'method': 'VolumePairList',
                                  'number_assets': 4,
                                  }]
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    ret = rpc._rpc_whitelist()
    assert len(ret['method']) == 1
    assert 'VolumePairList' in ret['method']
    assert ret['length'] == 4
    assert ret['whitelist'] == default_conf['exchange']['pair_whitelist']


def test_rpc_blacklist(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    ret = rpc._rpc_blacklist(None)
    assert len(ret['method']) == 1
    assert 'StaticPairList' in ret['method']
    assert len(ret['blacklist']) == 2
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC']

    ret = rpc._rpc_blacklist(["ETH/BTC"])
    assert 'StaticPairList' in ret['method']
    assert len(ret['blacklist']) == 3
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC']

    ret = rpc._rpc_blacklist(["ETH/BTC"])
    assert 'errors' in ret
    assert isinstance(ret['errors'], dict)
    assert ret['errors']['ETH/BTC']['error_msg'] == 'Pair ETH/BTC already in pairlist.'

    ret = rpc._rpc_blacklist(["*/BTC"])
    assert 'StaticPairList' in ret['method']
    assert len(ret['blacklist']) == 3
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC']
    assert ret['blacklist_expanded'] == ['ETH/BTC']
    assert 'errors' in ret
    assert isinstance(ret['errors'], dict)
    assert ret['errors'] == {'*/BTC': {'error_msg': 'Pair */BTC is not a valid wildcard.'}}

    ret = rpc._rpc_blacklist(["XRP/.*"])
    assert 'StaticPairList' in ret['method']
    assert len(ret['blacklist']) == 4
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC', 'XRP/.*']
    assert ret['blacklist_expanded'] == ['ETH/BTC', 'XRP/BTC', 'XRP/USDT']
    assert 'errors' in ret
    assert isinstance(ret['errors'], dict)

    ret = rpc._rpc_blacklist_delete(["DOGE/BTC", 'HOT/BTC'])

    assert 'StaticPairList' in ret['method']
    assert len(ret['blacklist']) == 2
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['ETH/BTC', 'XRP/.*']
    assert ret['blacklist_expanded'] == ['ETH/BTC', 'XRP/BTC', 'XRP/USDT']
    assert 'errors' in ret
    assert isinstance(ret['errors'], dict)


def test_rpc_edge_disabled(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    with pytest.raises(RPCException, match=r'Edge is not enabled.'):
        rpc._rpc_edge()


def test_rpc_edge_enabled(mocker, edge_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.02, 0.66, 3.71, 0.50, 1.71, 10, 60),
        }
    ))
    freqtradebot = get_patched_freqtradebot(mocker, edge_conf)

    rpc = RPC(freqtradebot)
    ret = rpc._rpc_edge()

    assert len(ret) == 1
    assert ret[0]['Pair'] == 'E/F'
    assert ret[0]['Winrate'] == 0.66
    assert ret[0]['Expectancy'] == 1.71
    assert ret[0]['Stoploss'] == -0.02


def test_rpc_health(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    set_startup_time()
    rpc = RPC(freqtradebot)
    result = rpc.health()
    assert result['last_process'] is None
    assert result['last_process_ts'] is None
