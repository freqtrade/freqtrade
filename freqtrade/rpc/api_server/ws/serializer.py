import logging
from abc import ABC, abstractmethod
from typing import Any, Union

import orjson
import rapidjson
from pandas import DataFrame

from freqtrade.misc import dataframe_to_json, json_to_dataframe
from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy
from freqtrade.rpc.api_server.ws_schemas import WSMessageSchemaType


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

    async def send(self, data: Union[WSMessageSchemaType, dict[str, Any]]):
        await self._websocket.send(self._serialize(data))

    async def recv(self) -> bytes:
        data = await self._websocket.recv()
        return self._deserialize(data)


class HybridJSONWebSocketSerializer(WebSocketSerializer):
    def _serialize(self, data) -> str:
        return str(orjson.dumps(data, default=_json_default), "utf-8")

    def _deserialize(self, data: str):
        # RapidJSON expects strings
        return rapidjson.loads(data, object_hook=_json_object_hook)


# Support serializing pandas DataFrames
def _json_default(z):
    if isinstance(z, DataFrame):
        return {"__type__": "dataframe", "__value__": dataframe_to_json(z)}
    raise TypeError


# Support deserializing JSON to pandas DataFrames
def _json_object_hook(z):
    if z.get("__type__") == "dataframe":
        return json_to_dataframe(z.get("__value__"))
    return z
