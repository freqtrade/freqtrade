from typing import TypeVar

from fastapi import WebSocket as FastAPIWebSocket
from websockets import WebSocketClientProtocol as WebSocket


WebSocketType = TypeVar("WebSocketType", FastAPIWebSocket, WebSocket)
