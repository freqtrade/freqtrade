import json
import logging
from abc import ABC, abstractmethod

import msgpack
import orjson
import rapidjson
from pandas import DataFrame

from freqtrade.misc import dataframe_to_json, json_to_dataframe
from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy


logger = logging.getLogger(__name__)


class WebSocketSerializer(ABC):
    def __init__(self, websocket: WebSocketProxy):
        self._websocket: WebSocketProxy = websocket

    @abstractmethod
    def _serialize(self, data):
        raise NotImplementedError()

    @abstractmethod
    def _deserialize(self, data):
        raise NotImplementedError()

    async def send(self, data: bytes):
        await self._websocket.send(self._serialize(data))

    async def recv(self) -> bytes:
        data = await self._websocket.recv()

        return self._deserialize(data)

    async def close(self, code: int = 1000):
        await self._websocket.close(code)


class JSONWebSocketSerializer(WebSocketSerializer):
    def _serialize(self, data):
        return json.dumps(data, default=_json_default)

    def _deserialize(self, data):
        return json.loads(data, object_hook=_json_object_hook)


# ORJSON does not support .loads(object_hook=x) parameter, so we must use RapidJSON

class RapidJSONWebSocketSerializer(WebSocketSerializer):
    def _serialize(self, data):
        return rapidjson.dumps(data, default=_json_default)

    def _deserialize(self, data):
        return rapidjson.loads(data, object_hook=_json_object_hook)


class HybridJSONWebSocketSerializer(WebSocketSerializer):
    def _serialize(self, data):
        # ORJSON returns bytes
        return orjson.dumps(data, default=_json_default)

    def _deserialize(self, data):
        # RapidJSON expects strings
        return rapidjson.loads(data, object_hook=_json_object_hook)


class MsgPackWebSocketSerializer(WebSocketSerializer):
    def _serialize(self, data):
        return msgpack.packb(data, use_bin_type=True)

    def _deserialize(self, data):
        return msgpack.unpackb(data, raw=False)


# Support serializing pandas DataFrames
def _json_default(z):
    if isinstance(z, DataFrame):
        return {
            '__type__': 'dataframe',
            '__value__': dataframe_to_json(z)
        }
    raise TypeError


# Support deserializing JSON to pandas DataFrames
def _json_object_hook(z):
    if z.get('__type__') == 'dataframe':
        return json_to_dataframe(z.get('__value__'))
    return z
