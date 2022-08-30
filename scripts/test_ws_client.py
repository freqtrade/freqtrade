import asyncio
import logging
import socket
from typing import Any

import websockets

from freqtrade.enums import RPCMessageType
from freqtrade.rpc.api_server.ws.channel import WebSocketChannel


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def compose_consumer_request(type_: str, data: Any):
    return {"type": type_, "data": data}


async def _client():
    # Trying to recreate multiple topic issue. Wait until first whitelist message,
    # then CTRL-C to get the status message.
    topics = [RPCMessageType.WHITELIST, RPCMessageType.STATUS]
    try:
        while True:
            try:
                url = "ws://localhost:8080/api/v1/message/ws?token=testtoken"
                async with websockets.connect(url) as ws:
                    channel = WebSocketChannel(ws)

                    logger.info("Connection successful")
                    # Tell the producer we only want these topics
                    await channel.send(compose_consumer_request("subscribe", topics))

                    while True:
                        try:
                            data = await asyncio.wait_for(
                                channel.recv(),
                                timeout=5
                            )
                            logger.info(f"Data received - {data}")
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                            # We haven't received data yet. Check the connection and continue.
                            try:
                                # ping
                                ping = await channel.ping()
                                await asyncio.wait_for(ping, timeout=2)
                                logger.debug(f"Connection to {url} still alive...")
                                continue
                            except Exception:
                                logger.info(
                                    f"Ping error {url} - retrying in 5s")
                                await asyncio.sleep(2)
                                break

            except (socket.gaierror, ConnectionRefusedError):
                logger.info("Connection Refused - retrying connection in 5s")
                await asyncio.sleep(2)
                continue
            except websockets.exceptions.InvalidStatusCode as e:
                logger.error(f"Connection Refused - {e}")
                await asyncio.sleep(2)
                continue

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass


def main():
    asyncio.run(_client())


if __name__ == "__main__":
    main()
