from typing import TypeVar

from fastapi import WebSocket as FastAPIWebSocket
from websockets import WebSocketClientProtocol as WebSocket

from freqtrade.rpc.replicate.channel import WebSocketProxy


WebSocketType = TypeVar("WebSocketType", FastAPIWebSocket, WebSocket, WebSocketProxy)
