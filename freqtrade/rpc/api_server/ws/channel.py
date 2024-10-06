import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional, Union
from uuid import uuid4

from fastapi import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy
from freqtrade.rpc.api_server.ws.serializer import (
    HybridJSONWebSocketSerializer,
    WebSocketSerializer,
)
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
        serializer_cls: type[WebSocketSerializer] = HybridJSONWebSocketSerializer,
        send_throttle: float = 0.01,
    ):
        self.channel_id = channel_id if channel_id else uuid4().hex[:8]
        self._websocket = WebSocketProxy(websocket)

        # Internal event to signify a closed websocket
        self._closed = asyncio.Event()
        # The async tasks created for the channel
        self._channel_tasks: list[asyncio.Task] = []

        # Deque for average send times
        self._send_times: deque[float] = deque([], maxlen=10)
        # High limit defaults to 3 to start
        self._send_high_limit = 3
        self._send_throttle = send_throttle

        # The subscribed message types
        self._subscriptions: list[str] = []

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

    @property
    def avg_send_time(self):
        return sum(self._send_times) / len(self._send_times)

    def _calc_send_limit(self):
        """
        Calculate the send high limit for this channel
        """

        # Only update if we have enough data
        if len(self._send_times) == self._send_times.maxlen:
            # At least 1s or twice the average of send times, with a
            # maximum of 3 seconds per message
            self._send_high_limit = min(max(self.avg_send_time * 2, 1), 3)

    async def send(
        self, message: Union[WSMessageSchemaType, dict[str, Any]], use_timeout: bool = False
    ):
        """
        Send a message on the wrapped websocket. If the sending
        takes too long, it will raise a TimeoutError and
        disconnect the connection.

        :param message: The message to send
        :param use_timeout: Enforce send high limit, defaults to False
        """
        try:
            _ = time.time()
            # If the send times out, it will raise
            # a TimeoutError and bubble up to the
            # message_endpoint to close the connection
            await asyncio.wait_for(
                self._wrapped_ws.send(message),
                timeout=self._send_high_limit if use_timeout else None,
            )
            total_time = time.time() - _
            self._send_times.append(total_time)

            self._calc_send_limit()
        except asyncio.TimeoutError:
            logger.info(f"Connection for {self} timed out, disconnecting")
            raise

        # Explicitly give control back to event loop as
        # websockets.send does not
        # Also throttles how fast we send
        await asyncio.sleep(self._send_throttle)

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
        Accept the underlying websocket connection,
        if the connection has been closed before we can
        accept, just close the channel.
        """
        try:
            return await self._websocket.accept()
        except RuntimeError:
            await self.close()

    async def close(self):
        """
        Close the WebSocketChannel
        """

        self._closed.set()

        try:
            await self._websocket.close()
        except RuntimeError:
            pass

    def is_closed(self) -> bool:
        """
        Closed flag
        """
        return self._closed.is_set()

    def set_subscriptions(self, subscriptions: list[str]) -> None:
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

    async def run_channel_tasks(self, *tasks, **kwargs):
        """
        Create and await on the channel tasks unless an exception
        was raised, then cancel them all.

        :params *tasks: All coros or tasks to be run concurrently
        :param **kwargs: Any extra kwargs to pass to gather
        """

        if not self.is_closed():
            # Wrap the coros into tasks if they aren't already
            self._channel_tasks = [
                task if isinstance(task, asyncio.Task) else asyncio.create_task(task)
                for task in tasks
            ]

            try:
                return await asyncio.gather(*self._channel_tasks, **kwargs)
            except Exception:
                # If an exception occurred, cancel the rest of the tasks
                await self.cancel_channel_tasks()

    async def cancel_channel_tasks(self):
        """
        Cancel and wait on all channel tasks
        """
        for task in self._channel_tasks:
            task.cancel()

            # Wait for tasks to finish cancelling
            try:
                await task
            except (
                asyncio.CancelledError,
                asyncio.TimeoutError,
                WebSocketDisconnect,
                ConnectionClosed,
                RuntimeError,
            ):
                pass
            except Exception as e:
                logger.info(f"Encountered unknown exception: {e}", exc_info=e)

        self._channel_tasks = []

    async def __aiter__(self):
        """
        Generator for received messages
        """
        # We can not catch any errors here as websocket.recv is
        # the first to catch any disconnects and bubble it up
        # so the connection is garbage collected right away
        while not self.is_closed():
            yield await self.recv()


@asynccontextmanager
async def create_channel(websocket: WebSocketType, **kwargs) -> AsyncIterator[WebSocketChannel]:
    """
    Context manager for safely opening and closing a WebSocketChannel
    """
    channel = WebSocketChannel(websocket, **kwargs)
    try:
        await channel.accept()
        logger.info(f"Connected to channel - {channel}")

        yield channel
    finally:
        await channel.close()
        logger.info(f"Disconnected from channel - {channel}")
