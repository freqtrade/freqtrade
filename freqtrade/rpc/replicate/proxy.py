from typing import TYPE_CHECKING, Union

from fastapi import WebSocket as FastAPIWebSocket
from websockets import WebSocketClientProtocol as WebSocket


if TYPE_CHECKING:
    from freqtrade.rpc.replicate.types import WebSocketType


class WebSocketProxy:
    """
    WebSocketProxy object to bring the FastAPIWebSocket and websockets.WebSocketClientProtocol
    under the same API
    """

    def __init__(self, websocket: WebSocketType):
        self._websocket: Union[FastAPIWebSocket, WebSocket] = websocket

    async def send(self, data):
        """
        Send data on the wrapped websocket
        """
        if hasattr(self._websocket, "send_bytes"):
            await self._websocket.send_bytes(data)
        else:
            await self._websocket.send(data)

    async def recv(self):
        """
        Receive data on the wrapped websocket
        """
        if hasattr(self._websocket, "receive_bytes"):
            return await self._websocket.receive_bytes()
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
