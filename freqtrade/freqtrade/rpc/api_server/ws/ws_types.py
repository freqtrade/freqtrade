from typing import Any, TypeVar

from fastapi import WebSocket as FastAPIWebSocket
from websockets.asyncio.client import ClientConnection as WebSocket


WebSocketType = TypeVar("WebSocketType", FastAPIWebSocket, WebSocket)
MessageType = dict[str, Any]
