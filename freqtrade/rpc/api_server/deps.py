from typing import Any, Dict, Optional

from freqtrade.persistence import Trade
from freqtrade.rpc.rpc import RPC, RPCException

from .webserver import ApiServer


def get_rpc_optional() -> Optional[RPC]:
    if ApiServer._has_rpc:
        return ApiServer._rpc
    return None


def get_rpc() -> Optional[RPC]:
    _rpc = get_rpc_optional()
    if _rpc:
        Trade.query.session.rollback()
        return _rpc
    else:
        raise RPCException('Bot is not in the correct state')


def get_config() -> Dict[str, Any]:
    return ApiServer._config


def get_api_config() -> Dict[str, Any]:
    return ApiServer._config['api_server']
