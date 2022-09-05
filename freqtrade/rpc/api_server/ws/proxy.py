from typing import Any, Tuple, Union

from fastapi import WebSocket as FastAPIWebSocket
from websockets import WebSocketClientProtocol as WebSocket

from freqtrade.rpc.api_server.ws.types import WebSocketType


class WebSocketProxy:
    """
    WebSocketProxy object to bring the FastAPIWebSocket and websockets.WebSocketClientProtocol
    under the same API
    """

    def __init__(self, websocket: WebSocketType):
        self._websocket: Union[FastAPIWebSocket, WebSocket] = websocket

    @property
    def remote_addr(self) -> Tuple[Any, ...]:
        if hasattr(self._websocket, "remote_address"):
            return self._websocket.remote_address
        elif hasattr(self._websocket, "client"):
            return tuple(self._websocket.client)
        return ("unknown", 0)

    async def send(self, data):
        """
        Send data on the wrapped websocket
        """

        if not isinstance(data, str):
            # We use HybridJSONWebSocketSerializer, which when serialized returns
            # bytes because of ORJSON, so we explicitly decode into a string
            data = str(data, "utf-8")

        if hasattr(self._websocket, "send_text"):
            await self._websocket.send_text(data)
        else:
            await self._websocket.send(data)

    async def recv(self):
        """
        Receive data on the wrapped websocket
        """
        if hasattr(self._websocket, "receive_text"):
            return await self._websocket.receive_text()
        else:
            return await self._websocket.recv()

    async def ping(self):
        """
        Ping the websocket, not supported by FastAPI WebSockets
        """
        if hasattr(self._websocket, "ping"):
            return await self._websocket.ping()
        return False

    async def close(self, code: int = 1000):
        """
        Close the websocket connection, only supported by FastAPI WebSockets
        """
        if hasattr(self._websocket, "close"):
            return await self._websocket.close(code)
        pass

    async def accept(self):
        """
        Accept the WebSocket connection, only support by FastAPI WebSockets
        """
        if hasattr(self._websocket, "accept"):
            return await self._websocket.accept()
        pass
