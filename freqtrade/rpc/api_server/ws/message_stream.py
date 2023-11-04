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
        waiter, self._waiter = self._waiter, self._loop.create_future()
        waiter.set_result((message, time.time(), self._waiter))

    async def __aiter__(self):
        """
        Iterate over the messages in the message stream
        """
        waiter = self._waiter
        while True:
            # Shield the future from being cancelled by a task waiting on it
            message, ts, waiter = await asyncio.shield(waiter)
            yield message, ts
