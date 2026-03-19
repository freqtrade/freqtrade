from unittest.mock import MagicMock, PropertyMock

import pytest
import requests

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.RedditSentimentPairList import RedditSentimentPairList
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, get_patched_exchange, log_has


@pytest.fixture(scope="function")
def rsp_config(default_conf):
    default_conf["stake_currency"] = "USDT"
    default_conf["exchange"]["pair_whitelist"] = [
        "BTC/USDT",
        "ETH/USDT",
        "XRP/USDT",
        "ADA/USDT",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/USDT"]
    return default_conf


def _mock_response(rows):
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": "application/json"}
    response.json.return_value = rows
    return response


def test_reddit_sentiment_pairlist_generator(mocker, rsp_config, markets):
    rsp_config["pairlists"] = [
        {
            "method": "RedditSentimentPairList",
            "api_url": "https://example.com/trending",
            "api_key": "test-key",
            "number_assets": 2,
        }
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )
    mocker.patch(
        "freqtrade.plugins.pairlist.RedditSentimentPairList.requests.get",
        return_value=_mock_response(
            [
                {"symbol": "ETH", "buzz_score": 82.4, "mentions": 215, "bullish_pct": 63},
                {"symbol": "XRP", "buzz_score": 76.2, "mentions": 144, "bullish_pct": 58},
                {"symbol": "ADA", "buzz_score": 61.8, "mentions": 91, "bullish_pct": 54},
            ]
        ),
    )

    exchange = get_patched_exchange(mocker, rsp_config)
    pm = PairListManager(exchange, rsp_config)
    pm.refresh_pairlist()

    assert pm.whitelist == ["ETH/USDT", "XRP/USDT"]


def test_reddit_sentiment_pairlist_filters_thresholds(mocker, rsp_config, markets):
    rsp_config["pairlists"] = [
        {"method": "StaticPairList"},
        {
            "method": "RedditSentimentPairList",
            "api_url": "https://example.com/trending",
            "api_key": "test-key",
            "number_assets": 3,
            "min_buzz_score": 70,
            "min_mentions": 100,
            "min_bullish_pct": 55,
            "allowed_trends": ["rising", "stable"],
        },
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )
    mocker.patch(
        "freqtrade.plugins.pairlist.RedditSentimentPairList.requests.get",
        return_value=_mock_response(
            [
                {
                    "symbol": "ETH",
                    "buzz_score": 82.4,
                    "mentions": 215,
                    "bullish_pct": 63,
                    "trend": "rising",
                },
                {
                    "symbol": "XRP",
                    "buzz_score": 74.1,
                    "mentions": 80,
                    "bullish_pct": 61,
                    "trend": "rising",
                },
                {
                    "symbol": "ADA",
                    "buzz_score": 71.0,
                    "mentions": 132,
                    "bullish_pct": 49,
                    "trend": "stable",
                },
            ]
        ),
    )

    exchange = get_patched_exchange(mocker, rsp_config)
    pm = PairListManager(exchange, rsp_config)
    pm.refresh_pairlist()

    assert pm.whitelist == ["ETH/USDT"]


def test_reddit_sentiment_pairlist_blacklist_mode(mocker, rsp_config, markets):
    rsp_config["pairlists"] = [
        {
            "method": "RedditSentimentPairList",
            "api_url": "https://example.com/trending",
            "api_key": "test-key",
            "mode": "blacklist",
            "max_tokens": 2,
        }
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )
    mocker.patch(
        "freqtrade.plugins.pairlist.RedditSentimentPairList.requests.get",
        return_value=_mock_response(
            [
                {"symbol": "ETH", "buzz_score": 82.4, "mentions": 215},
                {"symbol": "XRP", "buzz_score": 76.2, "mentions": 144},
            ]
        ),
    )

    exchange = get_patched_exchange(mocker, rsp_config)
    pm = PairListManager(exchange, rsp_config)
    pm.refresh_pairlist()

    assert "ETH/USDT" not in pm.whitelist
    assert "XRP/USDT" not in pm.whitelist
    assert "BTC/USDT" in pm.whitelist


def test_reddit_sentiment_pairlist_prefix_resolution(mocker, rsp_config, markets):
    rsp_config["exchange"]["pair_whitelist"] = []
    rsp_config["pairlists"] = [
        {
            "method": "RedditSentimentPairList",
            "api_url": "https://example.com/trending",
            "api_key": "test-key",
            "number_assets": 2,
        }
    ]
    markets["1000PEPE/USDT"] = markets["ETH/USDT"]
    markets["KBONK/USDT"] = markets["XRP/USDT"]
    del markets["ETH/USDT"]
    del markets["XRP/USDT"]

    markets_mock = MagicMock(return_value=markets)
    mocker.patch.multiple(
        EXMS,
        get_markets=markets_mock,
        exchange_has=MagicMock(return_value=True),
    )
    mocker.patch(
        "freqtrade.plugins.pairlist.RedditSentimentPairList.requests.get",
        return_value=_mock_response(
            [
                {"symbol": "PEPE", "buzz_score": 81.2, "mentions": 451},
                {"symbol": "BONK", "buzz_score": 77.7, "mentions": 302},
            ]
        ),
    )

    exchange = get_patched_exchange(mocker, rsp_config)
    pm = PairListManager(exchange, rsp_config)
    pm.refresh_pairlist()

    assert pm.whitelist == ["1000PEPE/USDT", "KBONK/USDT"]


def test_reddit_sentiment_pairlist_uses_cache(mocker, rsp_config, markets):
    rsp_config["pairlists"] = [
        {
            "method": "RedditSentimentPairList",
            "api_url": "https://example.com/trending",
            "api_key": "test-key",
            "number_assets": 2,
            "refresh_period": 21600,
        }
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )
    get_mock = mocker.patch(
        "freqtrade.plugins.pairlist.RedditSentimentPairList.requests.get",
        return_value=_mock_response(
            [
                {"symbol": "ETH", "buzz_score": 82.4, "mentions": 215},
                {"symbol": "XRP", "buzz_score": 76.2, "mentions": 144},
            ]
        ),
    )

    exchange = get_patched_exchange(mocker, rsp_config)
    pm = PairListManager(exchange, rsp_config)
    pm.refresh_pairlist()
    pm.refresh_pairlist()

    assert get_mock.call_count == 1


def test_reddit_sentiment_pairlist_keep_last_on_failure(mocker, rsp_config, markets, caplog):
    rsp_config["pairlists"] = [
        {
            "method": "RedditSentimentPairList",
            "api_url": "https://example.com/trending",
            "api_key": "test-key",
            "number_assets": 2,
            "refresh_period": 0,
            "keep_pairlist_on_failure": True,
        }
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
    )
    get_mock = mocker.patch(
        "freqtrade.plugins.pairlist.RedditSentimentPairList.requests.get",
        side_effect=[
            _mock_response(
                [
                    {"symbol": "ETH", "buzz_score": 82.4, "mentions": 215},
                    {"symbol": "XRP", "buzz_score": 76.2, "mentions": 144},
                ]
            ),
            requests.exceptions.RequestException,
        ],
    )

    exchange = get_patched_exchange(mocker, rsp_config)
    pairlistmanager = PairListManager(exchange, rsp_config)
    pairlist = RedditSentimentPairList(
        exchange,
        pairlistmanager,
        rsp_config,
        rsp_config["pairlists"][0],
        0,
    )

    first_rows = pairlist.fetch_sentiment_rows()
    pairlist._sentiment_cache.clear()
    second_rows = pairlist.fetch_sentiment_rows()

    assert get_mock.call_count == 2
    assert first_rows == second_rows
    assert log_has("Keeping last fetched sentiment universe", caplog)


def test_reddit_sentiment_pairlist_exceptions(mocker, rsp_config):
    exchange = get_patched_exchange(mocker, rsp_config)
    rsp_config["pairlists"] = [{"method": "RedditSentimentPairList"}]
    with pytest.raises(OperationalException, match=r"`number_assets` not specified.*"):
        PairListManager(exchange, rsp_config)

    rsp_config["pairlists"] = [
        {
            "method": "RedditSentimentPairList",
            "api_url": "https://example.com/trending",
            "api_key": "test-key",
            "number_assets": 10,
            "allowed_trends": ["up-only"],
        }
    ]
    with pytest.raises(OperationalException, match=r"`allowed_trends` not configured correctly.*"):
        PairListManager(exchange, rsp_config)
