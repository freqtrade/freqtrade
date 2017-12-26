# pragma pylint: disable=missing-docstring,C0103

import pytest
from requests.exceptions import ContentDecodingError

from freqtrade.exchange import Bittrex


def test_validate_response_success():
    response = {
        'message': '',
        'result': [],
    }
    Bittrex._validate_response(response)


def test_validate_response_no_api_response():
    response = {
        'message': 'NO_API_RESPONSE',
        'result': None,
    }
    with pytest.raises(ContentDecodingError, match=r'.*NO_API_RESPONSE.*'):
        Bittrex._validate_response(response)


def test_validate_response_min_trade_requirement_not_met():
    response = {
        'message': 'MIN_TRADE_REQUIREMENT_NOT_MET',
        'result': None,
    }
    with pytest.raises(ContentDecodingError, match=r'.*MIN_TRADE_REQUIREMENT_NOT_MET.*'):
        Bittrex._validate_response(response)
