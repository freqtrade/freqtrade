import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy
from freqtrade.rpc.api_server.ws.serializer import (HybridJSONWebSocketSerializer,
                                                    WebSocketSerializer)
from freqtrade.rpc.api_server.ws.types import WebSocketType
from freqtrade.rpc.api_server.ws_schemas import WSMessageSchemaType


logger = logging.getLogger(__name__)


class WebSocketChannel:
    """
    Object to help facilitate managing a websocket connection
    """
    def __init__(
        self,
        websocket: WebSocketType,
        channel_id: Optional[str] = None,
        serializer_cls: Type[WebSocketSerializer] = HybridJSONWebSocketSerializer
    ):
        self.channel_id = channel_id if channel_id else uuid4().hex[:8]
        self._websocket = WebSocketProxy(websocket)

        # Internal event to signify a closed websocket
        self._closed = asyncio.Event()

        # The subscribed message types
        self._subscriptions: List[str] = []

        # Wrap the WebSocket in the Serializing class
        self._wrapped_ws = serializer_cls(self._websocket)

    def __repr__(self):
        return f"WebSocketChannel({self.channel_id}, {self.remote_addr})"

    @property
    def raw_websocket(self):
        return self._websocket.raw_websocket

    @property
    def remote_addr(self):
        return self._websocket.remote_addr

    async def send(self, message: Union[WSMessageSchemaType, Dict[str, Any]]):
        """
        Send a message on the wrapped websocket
        """
        await self._wrapped_ws.send(message)

    async def recv(self):
        """
        Receive a message on the wrapped websocket
        """
        return await self._wrapped_ws.recv()

    async def ping(self):
        """
        Ping the websocket
        """
        return await self._websocket.ping()

    async def accept(self):
        """
        Accept the underlying websocket connection
        """
        return await self._websocket.accept()

    async def close(self):
        """
        Close the WebSocketChannel
        """

        self._closed.set()
        self._relay_task.cancel()

        try:
            await self._websocket.close()
        except Exception:
            pass

    def is_closed(self) -> bool:
        """
        Closed flag
        """
        return self._closed.is_set()

    def set_subscriptions(self, subscriptions: List[str] = []) -> None:
        """
        Set which subscriptions this channel is subscribed to

        :param subscriptions: List of subscriptions, List[str]
        """
        self._subscriptions = subscriptions

    def subscribed_to(self, message_type: str) -> bool:
        """
        Check if this channel is subscribed to the message_type

        :param message_type: The message type to check
        """
        return message_type in self._subscriptions

    async def __aiter__(self):
        """
        Generator for received messages
        """
        while True:
            try:
                yield await self.recv()
            except Exception:
                break

    @asynccontextmanager
    async def connect(self):
        """
        Context manager for safely opening and closing the websocket connection
        """
        try:
            await self.accept()
            yield self
        finally:
            await self.close()
