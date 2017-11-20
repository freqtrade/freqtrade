# pragma pylint: disable=missing-docstring,C0103
from unittest.mock import MagicMock

import pytest

from freqtrade import OperationalException
from freqtrade.exchange import validate_pairs


def test_validate_pairs(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.get_markets = MagicMock(return_value=[
        'BTC_ETH', 'BTC_TKN', 'BTC_TRST', 'BTC_SWT', 'BTC_BCC',
    ])
    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    validate_pairs(default_conf['exchange']['pair_whitelist'])


def test_validate_pairs_not_available(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.get_markets = MagicMock(return_value=[])
    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    with pytest.raises(OperationalException, match=r'not available'):
        validate_pairs(default_conf['exchange']['pair_whitelist'])


def test_validate_pairs_not_compatible(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.get_markets = MagicMock(return_value=['BTC_ETH', 'BTC_TKN', 'BTC_TRST', 'BTC_SWT'])
    default_conf['stake_currency'] = 'ETH'
    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    with pytest.raises(OperationalException, match=r'not compatible'):
        validate_pairs(default_conf['exchange']['pair_whitelist'])
