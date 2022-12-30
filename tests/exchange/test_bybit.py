from unittest.mock import MagicMock

from freqtrade.exchange.exchange_utils import timeframe_to_msecs
from tests.conftest import get_mock_coro, get_patched_exchange


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
