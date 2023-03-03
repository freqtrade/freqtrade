from datetime import datetime, timezone
from unittest.mock import MagicMock

from freqtrade.enums.marginmode import MarginMode
from freqtrade.enums.tradingmode import TradingMode
from freqtrade.exchange.exchange_utils import timeframe_to_msecs
from tests.conftest import get_mock_coro, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


def test_additional_exchange_init_bybit(default_conf, mocker):
    default_conf['dry_run'] = False
    default_conf['trading_mode'] = TradingMode.FUTURES
    default_conf['margin_mode'] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.set_position_mode = MagicMock(return_value={"dualSidePosition": False})
    get_patched_exchange(mocker, default_conf, id="bybit", api_mock=api_mock)
    assert api_mock.set_position_mode.call_count == 1
    ccxt_exceptionhandlers(mocker, default_conf, api_mock, 'bybit',
                           "additional_exchange_init", "set_position_mode")


async def test_bybit_fetch_funding_rate(default_conf, mocker):
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    api_mock = MagicMock()
    api_mock.fetch_funding_rate_history = get_mock_coro(return_value=[])
    exchange = get_patched_exchange(mocker, default_conf, id='bybit', api_mock=api_mock)
    limit = 200
    # Test fetch_funding_rate_history (current data)
    await exchange._fetch_funding_rate_history(
        pair='BTC/USDT:USDT',
        timeframe='4h',
        limit=limit,
        )

    assert api_mock.fetch_funding_rate_history.call_count == 1
    assert api_mock.fetch_funding_rate_history.call_args_list[0][0][0] == 'BTC/USDT:USDT'
    kwargs = api_mock.fetch_funding_rate_history.call_args_list[0][1]
    assert kwargs['params'] == {}
    assert kwargs['since'] is None

    api_mock.fetch_funding_rate_history.reset_mock()
    since_ms = 1610000000000
    since_ms_end = since_ms + (timeframe_to_msecs('4h') * limit)
    # Test fetch_funding_rate_history (current data)
    await exchange._fetch_funding_rate_history(
        pair='BTC/USDT:USDT',
        timeframe='4h',
        limit=limit,
        since_ms=since_ms,
        )

    assert api_mock.fetch_funding_rate_history.call_count == 1
    assert api_mock.fetch_funding_rate_history.call_args_list[0][0][0] == 'BTC/USDT:USDT'
    kwargs = api_mock.fetch_funding_rate_history.call_args_list[0][1]
    assert kwargs['params'] == {'until': since_ms_end}
    assert kwargs['since'] == since_ms


def test_bybit_get_funding_fees(default_conf, mocker):
    now = datetime.now(timezone.utc)
    exchange = get_patched_exchange(mocker, default_conf, id='bybit')
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees('BTC/USDT:USDT', 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 0

    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    exchange = get_patched_exchange(mocker, default_conf, id='bybit')
    exchange._fetch_and_calculate_funding_fees = MagicMock()
    exchange.get_funding_fees('BTC/USDT:USDT', 1, False, now)

    assert exchange._fetch_and_calculate_funding_fees.call_count == 1
