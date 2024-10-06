from typing import Any, TypeVar

from fastapi import WebSocket as FastAPIWebSocket
from websockets.client import WebSocketClientProtocol as WebSocket


WebSocketType = TypeVar("WebSocketType", FastAPIWebSocket, WebSocket)
MessageType = dict[str, Any]
