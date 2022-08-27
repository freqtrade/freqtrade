import logging
from threading import RLock
from typing import Type

from freqtrade.rpc.external_signal.proxy import WebSocketProxy
from freqtrade.rpc.external_signal.serializer import MsgPackWebSocketSerializer, WebSocketSerializer
from freqtrade.rpc.external_signal.types import WebSocketType


logger = logging.getLogger(__name__)


class WebSocketChannel:
    """
    Object to help facilitate managing a websocket connection
    """

    def __init__(
        self,
        websocket: WebSocketType,
        serializer_cls: Type[WebSocketSerializer] = MsgPackWebSocketSerializer
    ):
        # The WebSocket object
        self._websocket = WebSocketProxy(websocket)
        # The Serializing class for the WebSocket object
        self._serializer_cls = serializer_cls

        # Internal event to signify a closed websocket
        self._closed = False

        # Wrap the WebSocket in the Serializing class
        self._wrapped_ws = self._serializer_cls(self._websocket)

    async def send(self, data):
        """
        Send data on the wrapped websocket
        """
        # logger.info(f"Serialized Send - {self._wrapped_ws._serialize(data)}")
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

    def is_closed(self):
        return self._closed


class ChannelManager:
    def __init__(self):
        self.channels = dict()
        self._lock = RLock()  # Re-entrant Lock

    async def on_connect(self, websocket: WebSocketType):
        """
        Wrap websocket connection into Channel and add to list

        :param websocket: The WebSocket object to attach to the Channel
        """
        if hasattr(websocket, "accept"):
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
                logger.debug(f"Disconnecting channel - {channel}")

                if not channel.is_closed():
                    await channel.close()

                del self.channels[websocket]

    async def disconnect_all(self):
        """
        Disconnect all Channels
        """
        with self._lock:
            for websocket, channel in self.channels.items():
                if not channel.is_closed():
                    await channel.close()

            self.channels = dict()

    async def broadcast(self, data):
        """
        Broadcast data on all Channels

        :param data: The data to send
        """
        with self._lock:
            for websocket, channel in self.channels.items():
                try:
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
        # We iterate over the channels to get reference to the websocket object
        # so we can disconnect incase of failure
        await channel.send(data)

    def has_channels(self):
        """
        Flag for more than 0 channels
        """
        return len(self.channels) > 0
