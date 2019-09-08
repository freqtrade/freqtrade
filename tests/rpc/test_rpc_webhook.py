# pragma pylint: disable=missing-docstring, C0103, protected-access

from unittest.mock import MagicMock

import pytest
from requests import RequestException

from freqtrade.rpc import RPCMessageType
from freqtrade.rpc.webhook import Webhook
from freqtrade.strategy.interface import SellType
from tests.conftest import get_patched_freqtradebot, log_has


def get_webhook_dict() -> dict:
    return {
          "enabled": True,
          "url": "https://maker.ifttt.com/trigger/freqtrade_test/with/key/c764udvJ5jfSlswVRukZZ2/",
          "webhookbuy": {
              "value1": "Buying {pair}",
              "value2": "limit {limit:8f}",
              "value3": "{stake_amount:8f} {stake_currency}"
          },
          "webhooksell": {
              "value1": "Selling {pair}",
              "value2": "limit {limit:8f}",
              "value3": "profit: {profit_amount:8f} {stake_currency}"
          },
          "webhookstatus": {
              "value1": "Status: {status}",
              "value2": "",
              "value3": ""
          }
      }


def test__init__(mocker, default_conf):
    default_conf['webhook'] = {'enabled': True, 'url': "https://DEADBEEF.com"}
    webhook = Webhook(get_patched_freqtradebot(mocker, default_conf))
    assert webhook._config == default_conf


def test_send_msg(default_conf, mocker):
    default_conf["webhook"] = get_webhook_dict()
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    webhook = Webhook(get_patched_freqtradebot(mocker, default_conf))
    msg = {
        'type': RPCMessageType.BUY_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'limit': 0.005,
        'stake_amount': 0.8,
        'stake_amount_fiat': 500,
        'stake_currency': 'BTC',
        'fiat_currency': 'EUR'
    }
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert (msg_mock.call_args[0][0]["value1"] ==
            default_conf["webhook"]["webhookbuy"]["value1"].format(**msg))
    assert (msg_mock.call_args[0][0]["value2"] ==
            default_conf["webhook"]["webhookbuy"]["value2"].format(**msg))
    assert (msg_mock.call_args[0][0]["value3"] ==
            default_conf["webhook"]["webhookbuy"]["value3"].format(**msg))
    # Test sell
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    msg = {
        'type': RPCMessageType.SELL_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'gain': "profit",
        'limit': 0.005,
        'amount': 0.8,
        'order_type': 'limit',
        'open_rate': 0.004,
        'current_rate': 0.005,
        'profit_amount': 0.001,
        'profit_percent': 0.20,
        'stake_currency': 'BTC',
        'sell_reason': SellType.STOP_LOSS.value
    }
    webhook.send_msg(msg=msg)
    assert msg_mock.call_count == 1
    assert (msg_mock.call_args[0][0]["value1"] ==
            default_conf["webhook"]["webhooksell"]["value1"].format(**msg))
    assert (msg_mock.call_args[0][0]["value2"] ==
            default_conf["webhook"]["webhooksell"]["value2"].format(**msg))
    assert (msg_mock.call_args[0][0]["value3"] ==
            default_conf["webhook"]["webhooksell"]["value3"].format(**msg))

    for msgtype in [RPCMessageType.STATUS_NOTIFICATION,
                    RPCMessageType.WARNING_NOTIFICATION,
                    RPCMessageType.CUSTOM_NOTIFICATION]:
        # Test notification
        msg = {
            'type': msgtype,
            'status': 'Unfilled sell order for BTC cancelled due to timeout'
        }
        msg_mock = MagicMock()
        mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
        webhook.send_msg(msg)
        assert msg_mock.call_count == 1
        assert (msg_mock.call_args[0][0]["value1"] ==
                default_conf["webhook"]["webhookstatus"]["value1"].format(**msg))
        assert (msg_mock.call_args[0][0]["value2"] ==
                default_conf["webhook"]["webhookstatus"]["value2"].format(**msg))
        assert (msg_mock.call_args[0][0]["value3"] ==
                default_conf["webhook"]["webhookstatus"]["value3"].format(**msg))


def test_exception_send_msg(default_conf, mocker, caplog):
    default_conf["webhook"] = get_webhook_dict()
    default_conf["webhook"]["webhookbuy"] = None

    webhook = Webhook(get_patched_freqtradebot(mocker, default_conf))
    webhook.send_msg({'type': RPCMessageType.BUY_NOTIFICATION})
    assert log_has(f"Message type {RPCMessageType.BUY_NOTIFICATION} not configured for webhooks",
                   caplog)

    default_conf["webhook"] = get_webhook_dict()
    default_conf["webhook"]["webhookbuy"]["value1"] = "{DEADBEEF:8f}"
    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    webhook = Webhook(get_patched_freqtradebot(mocker, default_conf))
    msg = {
        'type': RPCMessageType.BUY_NOTIFICATION,
        'exchange': 'Bittrex',
        'pair': 'ETH/BTC',
        'limit': 0.005,
        'order_type': 'limit',
        'stake_amount': 0.8,
        'stake_amount_fiat': 500,
        'stake_currency': 'BTC',
        'fiat_currency': 'EUR'
    }
    webhook.send_msg(msg)
    assert log_has("Problem calling Webhook. Please check your webhook configuration. "
                   "Exception: 'DEADBEEF'", caplog)

    msg_mock = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.Webhook._send_msg", msg_mock)
    msg = {
        'type': 'DEADBEEF',
        'status': 'whatever'
    }
    with pytest.raises(NotImplementedError):
        webhook.send_msg(msg)


def test__send_msg(default_conf, mocker, caplog):
    default_conf["webhook"] = get_webhook_dict()
    webhook = Webhook(get_patched_freqtradebot(mocker, default_conf))
    msg = {'value1': 'DEADBEEF',
           'value2': 'ALIVEBEEF',
           'value3': 'FREQTRADE'}
    post = MagicMock()
    mocker.patch("freqtrade.rpc.webhook.post", post)
    webhook._send_msg(msg)

    assert post.call_count == 1
    assert post.call_args[1] == {'data': msg}
    assert post.call_args[0] == (default_conf['webhook']['url'], )

    post = MagicMock(side_effect=RequestException)
    mocker.patch("freqtrade.rpc.webhook.post", post)
    webhook._send_msg(msg)
    assert log_has('Could not call webhook url. Exception: ', caplog)
