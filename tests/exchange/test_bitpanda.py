from datetime import datetime
from unittest.mock import MagicMock

from tests.conftest import EXMS, get_patched_exchange


def test_get_trades_for_order(default_conf, mocker):
    exchange_name = 'bitpanda'
    order_id = 'ABCD-ABCD'
    since = datetime(2018, 5, 5, 0, 0, 0)
    default_conf["dry_run"] = False
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    api_mock = MagicMock()

    api_mock.fetch_my_trades = MagicMock(return_value=[{'id': 'TTR67E-3PFBD-76IISV',
                                                        'order': 'ABCD-ABCD',
                                                        'info': {'pair': 'XLTCZBTC',
                                                                 'time': 1519860024.4388,
                                                                 'type': 'buy',
                                                                 'ordertype': 'limit',
                                                                 'price': '20.00000',
                                                                 'cost': '38.62000',
                                                                 'fee': '0.06179',
                                                                 'vol': '5',
                                                                 'id': 'ABCD-ABCD'},
                                                        'timestamp': 1519860024438,
                                                        'datetime': '2018-02-28T23:20:24.438Z',
                                                        'symbol': 'LTC/BTC',
                                                        'type': 'limit',
                                                        'side': 'buy',
                                                        'price': 165.0,
                                                        'amount': 0.2340606,
                                                        'fee': {'cost': 0.06179, 'currency': 'BTC'}
                                                        }])
    exchange = get_patched_exchange(mocker, default_conf, api_mock, id=exchange_name)

    orders = exchange.get_trades_for_order(order_id, 'LTC/BTC', since)
    assert len(orders) == 1
    assert orders[0]['price'] == 165
    assert api_mock.fetch_my_trades.call_count == 1
    # since argument should be
    assert isinstance(api_mock.fetch_my_trades.call_args[0][1], int)
    assert api_mock.fetch_my_trades.call_args[0][0] == 'LTC/BTC'
    # Same test twice, hardcoded number and doing the same calculation
    assert api_mock.fetch_my_trades.call_args[0][1] == 1525478395000
    # bitpanda requires "to" argument.
    assert 'to' in api_mock.fetch_my_trades.call_args[1]['params']
