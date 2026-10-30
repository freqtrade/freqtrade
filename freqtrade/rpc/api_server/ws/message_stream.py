import asyncio
import time


class MessageStream:
    """
    A message stream for consumers to subscribe to,
    and for producers to publish to.
    """

    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._waiter = self._loop.create_future()

    def publish(self, message):
        """
        Publish a message to this MessageStream

        :param message: The message to publish
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            self._publish(message)
        elif not self._loop.is_closed():
            # Futures are not thread-safe - hand over to the stream's event loop.
            try:
                self._loop.call_soon_threadsafe(self._publish, message, time.time())
            except RuntimeError:
                # drop the message
                pass

    def _publish(self, message, ts: float | None = None):
        waiter, self._waiter = self._waiter, self._loop.create_future()
        waiter.set_result((message, ts or time.time(), self._waiter))

    async def __aiter__(self):
        """
        Iterate over the messages in the message stream
        """
        waiter = self._waiter
        while True:
            # Shield the future from being cancelled by a task waiting on it
            message, ts, waiter = await asyncio.shield(waiter)
            yield message, ts
