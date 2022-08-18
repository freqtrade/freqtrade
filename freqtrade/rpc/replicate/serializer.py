import json
from abc import ABC, abstractmethod

from freqtrade.rpc.replicate.proxy import WebSocketProxy


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
    def _serialize(self, data: bytes) -> bytes:
        # json expects string not bytes
        return json.dumps(data.decode()).encode()

    def _deserialize(self, data: bytes) -> bytes:
        # The WebSocketSerializer gives bytes not string
        return json.loads(data).encode()
