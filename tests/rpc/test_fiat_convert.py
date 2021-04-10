# pragma pylint: disable=missing-docstring, too-many-arguments, too-many-ancestors,
# pragma pylint: disable=protected-access, C0103

from unittest.mock import MagicMock

import pytest
from requests.exceptions import RequestException

from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from tests.conftest import log_has, log_has_re


def test_fiat_convert_is_supported(mocker):
    fiat_convert = CryptoToFiatConverter()
    assert fiat_convert._is_supported_fiat(fiat='USD') is True
    assert fiat_convert._is_supported_fiat(fiat='usd') is True
    assert fiat_convert._is_supported_fiat(fiat='abc') is False
    assert fiat_convert._is_supported_fiat(fiat='ABC') is False


def test_fiat_convert_find_price(mocker):
    fiat_convert = CryptoToFiatConverter()

    with pytest.raises(ValueError, match=r'The fiat ABC is not supported.'):
        fiat_convert._find_price(crypto_symbol='BTC', fiat_symbol='ABC')

    assert fiat_convert.get_price(crypto_symbol='XRP', fiat_symbol='USD') == 0.0

    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=12345.0)
    assert fiat_convert.get_price(crypto_symbol='BTC', fiat_symbol='USD') == 12345.0
    assert fiat_convert.get_price(crypto_symbol='btc', fiat_symbol='usd') == 12345.0

    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=13000.2)
    assert fiat_convert.get_price(crypto_symbol='BTC', fiat_symbol='EUR') == 13000.2


def test_fiat_convert_unsupported_crypto(mocker, caplog):
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._cryptomap', return_value=[])
    fiat_convert = CryptoToFiatConverter()
    assert fiat_convert._find_price(crypto_symbol='CRYPTO_123', fiat_symbol='EUR') == 0.0
    assert log_has('unsupported crypto-symbol CRYPTO_123 - returning 0.0', caplog)


def test_fiat_convert_get_price(mocker):
    find_price = mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                              return_value=28000.0)

    fiat_convert = CryptoToFiatConverter()

    with pytest.raises(ValueError, match=r'The fiat us dollar is not supported.'):
        fiat_convert.get_price(crypto_symbol='btc', fiat_symbol='US Dollar')

    # Check the value return by the method
    pair_len = len(fiat_convert._pair_price)
    assert pair_len == 0
    assert fiat_convert.get_price(crypto_symbol='btc', fiat_symbol='usd') == 28000.0
    assert fiat_convert._pair_price['btc/usd'] == 28000.0
    assert len(fiat_convert._pair_price) == 1
    assert find_price.call_count == 1

    # Verify the cached is used
    fiat_convert._pair_price['btc/usd'] = 9867.543
    assert fiat_convert.get_price(crypto_symbol='btc', fiat_symbol='usd') == 9867.543
    assert find_price.call_count == 1


def test_fiat_convert_same_currencies(mocker):
    fiat_convert = CryptoToFiatConverter()

    assert fiat_convert.get_price(crypto_symbol='USD', fiat_symbol='USD') == 1.0


def test_fiat_convert_two_FIAT(mocker):
    fiat_convert = CryptoToFiatConverter()

    assert fiat_convert.get_price(crypto_symbol='USD', fiat_symbol='EUR') == 0.0


def test_loadcryptomap(mocker):

    fiat_convert = CryptoToFiatConverter()
    assert len(fiat_convert._cryptomap) == 2

    assert fiat_convert._cryptomap["btc"] == "bitcoin"


def test_fiat_init_network_exception(mocker):
    # Because CryptoToFiatConverter is a Singleton we reset the listings
    listmock = MagicMock(side_effect=RequestException)
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.CoinGeckoAPI',
        get_coins_list=listmock,
    )
    # with pytest.raises(RequestEsxception):
    fiat_convert = CryptoToFiatConverter()
    fiat_convert._cryptomap = {}
    fiat_convert._load_cryptomap()

    length_cryptomap = len(fiat_convert._cryptomap)
    assert length_cryptomap == 0


def test_fiat_convert_without_network(mocker):
    # Because CryptoToFiatConverter is a Singleton we reset the value of _coingekko

    fiat_convert = CryptoToFiatConverter()

    cmc_temp = CryptoToFiatConverter._coingekko
    CryptoToFiatConverter._coingekko = None

    assert fiat_convert._coingekko is None
    assert fiat_convert._find_price(crypto_symbol='btc', fiat_symbol='usd') == 0.0
    CryptoToFiatConverter._coingekko = cmc_temp


def test_fiat_invalid_response(mocker, caplog):
    # Because CryptoToFiatConverter is a Singleton we reset the listings
    listmock = MagicMock(return_value="{'novalidjson':DEADBEEFf}")
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.CoinGeckoAPI',
        get_coins_list=listmock,
    )
    # with pytest.raises(RequestEsxception):
    fiat_convert = CryptoToFiatConverter()
    fiat_convert._cryptomap = {}
    fiat_convert._load_cryptomap()

    length_cryptomap = len(fiat_convert._cryptomap)
    assert length_cryptomap == 0
    assert log_has_re('Could not load FIAT Cryptocurrency map for the following problem: .*',
                      caplog)


def test_convert_amount(mocker):
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter.get_price', return_value=12345.0)

    fiat_convert = CryptoToFiatConverter()
    result = fiat_convert.convert_amount(
        crypto_amount=1.23,
        crypto_symbol="BTC",
        fiat_symbol="USD"
    )
    assert result == 15184.35

    result = fiat_convert.convert_amount(
        crypto_amount=1.23,
        crypto_symbol="BTC",
        fiat_symbol="BTC"
    )
    assert result == 1.23

    result = fiat_convert.convert_amount(
        crypto_amount="1.23",
        crypto_symbol="BTC",
        fiat_symbol="BTC"
    )
    assert result == 1.23
