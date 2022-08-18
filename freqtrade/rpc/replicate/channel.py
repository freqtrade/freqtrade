from typing import Type

from freqtrade.rpc.replicate.proxy import WebSocketProxy
from freqtrade.rpc.replicate.serializer import JSONWebSocketSerializer, WebSocketSerializer
from freqtrade.rpc.replicate.types import WebSocketType


class WebSocketChannel:
    """
    Object to help facilitate managing a websocket connection
    """

    def __init__(
        self,
        websocket: WebSocketType,
        serializer_cls: Type[WebSocketSerializer] = JSONWebSocketSerializer
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
        self.channels[websocket] = ws_channel

        return ws_channel

    async def on_disconnect(self, websocket: WebSocketType):
        """
        Call close on the channel if it's not, and remove from channel list

        :param websocket: The WebSocket objet attached to the Channel
        """
        if websocket in self.channels.keys():
            channel = self.channels[websocket]
            if not channel.is_closed():
                await channel.close()
            del channel

    async def disconnect_all(self):
        """
        Disconnect all Channels
        """
        for websocket in self.channels.keys():
            await self.on_disconnect(websocket)

    async def broadcast(self, data):
        """
        Broadcast data on all Channels

        :param data: The data to send
        """
        for channel in self.channels.values():
            await channel.send(data)
