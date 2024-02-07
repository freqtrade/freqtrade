from typing import Any, Dict, TypeVar

from fastapi import WebSocket as FastAPIWebSocket
from websockets.client import WebSocketClientProtocol as WebSocket


WebSocketType = TypeVar("WebSocketType", FastAPIWebSocket, WebSocket)
MessageType = Dict[str, Any]
