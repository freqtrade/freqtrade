import logging
from threading import RLock
from typing import List, Optional, Type
from uuid import uuid4

from fastapi import WebSocket as FastAPIWebSocket

from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy
from freqtrade.rpc.api_server.ws.serializer import (HybridJSONWebSocketSerializer,
                                                    WebSocketSerializer)
from freqtrade.rpc.api_server.ws.types import WebSocketType


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

        # The WebSocket object
        self._websocket = WebSocketProxy(websocket)
        # The Serializing class for the WebSocket object
        self._serializer_cls = serializer_cls

        self._subscriptions: List[str] = []

        # Internal event to signify a closed websocket
        self._closed = False

        # Wrap the WebSocket in the Serializing class
        self._wrapped_ws = self._serializer_cls(self._websocket)

    def __repr__(self):
        return f"WebSocketChannel({self.channel_id}, {self.remote_addr})"

    @property
    def remote_addr(self):
        return self._websocket.remote_addr

    async def send(self, data):
        """
        Send data on the wrapped websocket
        """
        await self._wrapped_ws.send(data)

    async def recv(self):
        """
        Receive data on the wrapped websocket
        """
        return await self._wrapped_ws.recv()

    async def ping(self):
        """
        Ping the websocket
        """
        return await self._websocket.ping()

    async def close(self):
        """
        Close the WebSocketChannel
        """

        self._closed = True

    def is_closed(self) -> bool:
        """
        Closed flag
        """
        return self._closed

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


class ChannelManager:
    def __init__(self):
        self.channels = dict()
        self._lock = RLock()  # Re-entrant Lock

    async def on_connect(self, websocket: WebSocketType):
        """
        Wrap websocket connection into Channel and add to list

        :param websocket: The WebSocket object to attach to the Channel
        """
        if isinstance(websocket, FastAPIWebSocket):
            try:
                await websocket.accept()
            except RuntimeError:
                # The connection was closed before we could accept it
                return

        ws_channel = WebSocketChannel(websocket)

        with self._lock:
            self.channels[websocket] = ws_channel

        return ws_channel

    async def on_disconnect(self, websocket: WebSocketType):
        """
        Call close on the channel if it's not, and remove from channel list

        :param websocket: The WebSocket objet attached to the Channel
        """
        with self._lock:
            channel = self.channels.get(websocket)
            if channel:
                if not channel.is_closed():
                    await channel.close()

                del self.channels[websocket]

    async def disconnect_all(self):
        """
        Disconnect all Channels
        """
        with self._lock:
            for websocket, channel in self.channels.copy().items():
                if not channel.is_closed():
                    await channel.close()

            self.channels = dict()

    async def broadcast(self, data):
        """
        Broadcast data on all Channels

        :param data: The data to send
        """
        with self._lock:
            message_type = data.get('type')
            for websocket, channel in self.channels.copy().items():
                try:
                    if channel.subscribed_to(message_type):
                        await channel.send(data)
                except RuntimeError:
                    # Handle cannot send after close cases
                    await self.on_disconnect(websocket)

    async def send_direct(self, channel, data):
        """
        Send data directly through direct_channel only

        :param direct_channel: The WebSocketChannel object to send data through
        :param data: The data to send
        """
        await channel.send(data)

    def has_channels(self):
        """
        Flag for more than 0 channels
        """
        return len(self.channels) > 0
