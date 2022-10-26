import asyncio
import logging
from threading import RLock
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from fastapi import WebSocket as FastAPIWebSocket

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
        drain_timeout: int = 3,
        throttle: float = 0.01,
        serializer_cls: Type[WebSocketSerializer] = HybridJSONWebSocketSerializer
    ):

        self.channel_id = channel_id if channel_id else uuid4().hex[:8]

        # The WebSocket object
        self._websocket = WebSocketProxy(websocket)
        # The Serializing class for the WebSocket object
        self._serializer_cls = serializer_cls

        self.drain_timeout = drain_timeout
        self.throttle = throttle

        self._subscriptions: List[str] = []
        # 32 is the size of the receiving queue in websockets package
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=32)
        self._relay_task = asyncio.create_task(self.relay())

        # Internal event to signify a closed websocket
        self._closed = False

        # Wrap the WebSocket in the Serializing class
        self._wrapped_ws = self._serializer_cls(self._websocket)

    def __repr__(self):
        return f"WebSocketChannel({self.channel_id}, {self.remote_addr})"

    @property
    def raw_websocket(self):
        return self._websocket.raw_websocket

    @property
    def remote_addr(self):
        return self._websocket.remote_addr

    async def _send(self, data):
        """
        Send data on the wrapped websocket
        """
        await self._wrapped_ws.send(data)

    async def send(self, data) -> bool:
        """
        Add the data to the queue to be sent.
        :returns: True if data added to queue, False otherwise
        """
        try:
            await asyncio.wait_for(
                self.queue.put(data),
                timeout=self.drain_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

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
        self._relay_task.cancel()

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

    async def relay(self):
        """
        Relay messages from the channel's queue and send them out. This is started
        as a task.
        """
        while True:
            message = await self.queue.get()
            try:
                await self._send(message)
                self.queue.task_done()

                # Limit messages per sec.
                # Could cause problems with queue size if too low, and
                # problems with network traffik if too high.
                # 0.01 = 100/s
                await asyncio.sleep(self.throttle)
            except RuntimeError:
                # The connection was closed, just exit the task
                return


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
                logger.info(f"Disconnecting channel {channel}")
                if not channel.is_closed():
                    await channel.close()

                del self.channels[websocket]

    async def disconnect_all(self):
        """
        Disconnect all Channels
        """
        with self._lock:
            for websocket in self.channels.copy().keys():
                await self.on_disconnect(websocket)

    async def broadcast(self, message: WSMessageSchemaType):
        """
        Broadcast a message on all Channels

        :param message: The message to send
        """
        with self._lock:
            for channel in self.channels.copy().values():
                if channel.subscribed_to(message.get('type')):
                    await self.send_direct(channel, message)

    async def send_direct(
            self, channel: WebSocketChannel, message: Union[WSMessageSchemaType, Dict[str, Any]]):
        """
        Send a message directly through direct_channel only

        :param direct_channel: The WebSocketChannel object to send the message through
        :param message: The message to send
        """
        if not await channel.send(message):
            await self.on_disconnect(channel.raw_websocket)

    def has_channels(self):
        """
        Flag for more than 0 channels
        """
        return len(self.channels) > 0
