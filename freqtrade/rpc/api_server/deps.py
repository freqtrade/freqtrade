from typing import Any, Dict, Iterator, Optional

from fastapi import Depends

from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from freqtrade.rpc.rpc import RPC, RPCException

from .webserver import ApiServer


def get_rpc_optional() -> Optional[RPC]:
    if ApiServer._has_rpc:
        return ApiServer._rpc
    return None


def get_rpc() -> Optional[Iterator[RPC]]:
    _rpc = get_rpc_optional()
    if _rpc:
        Trade.rollback()
        yield _rpc
        Trade.rollback()
    else:
        raise RPCException('Bot is not in the correct state')


def get_config() -> Dict[str, Any]:
    return ApiServer._config


def get_api_config() -> Dict[str, Any]:
    return ApiServer._config['api_server']


def get_exchange(config=Depends(get_config)):
    if not ApiServer._exchange:
        from freqtrade.resolvers import ExchangeResolver
        ApiServer._exchange = ExchangeResolver.load_exchange(
            config['exchange']['name'], config, load_leverage_tiers=False)
    return ApiServer._exchange


def get_message_stream():
    return ApiServer._message_stream


def is_webserver_mode(config=Depends(get_config)):
    if config['runmode'] != RunMode.WEBSERVER:
        raise RPCException('Bot is not in the correct state')
    return None
