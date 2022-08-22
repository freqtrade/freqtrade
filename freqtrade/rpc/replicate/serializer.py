import json
import logging
from abc import ABC, abstractmethod

import msgpack
import orjson

from freqtrade.rpc.replicate.proxy import WebSocketProxy


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

# Going to explore using MsgPack as the serialization,
# as that might be the best method for sending pandas
# dataframes over the wire


class JSONWebSocketSerializer(WebSocketSerializer):
    def _serialize(self, data):
        return json.dumps(data)

    def _deserialize(self, data):
        return json.loads(data)


class ORJSONWebSocketSerializer(WebSocketSerializer):
    ORJSON_OPTIONS = orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY

    def _serialize(self, data):
        return orjson.dumps(data, option=self.ORJSON_OPTIONS)

    def _deserialize(self, data):
        return orjson.loads(data, option=self.ORJSON_OPTIONS)


class MsgPackWebSocketSerializer(WebSocketSerializer):
    def _serialize(self, data):
        return msgpack.packb(data, use_bin_type=True)

    def _deserialize(self, data):
        return msgpack.unpackb(data, raw=False)
