from typing import Any, Dict

from freqtrade.rpc.rpc import RPC

from .webserver import ApiServer


def get_rpc() -> RPC:
    return ApiServer._rpc


def get_config() -> Dict[str, Any]:
    return ApiServer._config


def get_api_config() -> Dict[str, Any]:
    return ApiServer._config['api_server']
