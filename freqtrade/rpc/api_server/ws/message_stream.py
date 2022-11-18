import asyncio


class MessageStream:
    """
    A message stream for consumers to subscribe to,
    and for producers to publish to.
    """
    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._waiter = self._loop.create_future()

    def publish(self, message):
        waiter, self._waiter = self._waiter, self._loop.create_future()
        waiter.set_result((message, self._waiter))

    async def subscribe(self):
        waiter = self._waiter
        while True:
            # Shield the future from being cancelled by a task waiting on it
            message, waiter = await asyncio.shield(waiter)
            yield message

    __aiter__ = subscribe
