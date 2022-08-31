"""
ExternalMessageConsumer module

Main purpose is to connect to external bot's message websocket to consume data
from it
"""
import asyncio
import logging
import socket
from threading import Thread
from typing import Any, Dict

import websockets

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RPCMessageType, RPCRequestType
from freqtrade.misc import json_to_dataframe, remove_entry_exit_signals
from freqtrade.rpc.api_server.ws.channel import WebSocketChannel


logger = logging.getLogger(__name__)


class ExternalMessageConsumer:
    """
    The main controller class for consuming external messages from
    other FreqTrade bot's
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dataprovider: DataProvider
    ):
        self._config = config
        self._dp = dataprovider

        self._running = False
        self._thread = None
        self._loop = None
        self._main_task = None
        self._sub_tasks = None

        self._emc_config = self._config.get('external_message_consumer', {})

        self.enabled = self._emc_config.get('enabled', False)
        self.producers = self._emc_config.get('producers', [])

        if self.enabled and len(self.producers) < 1:
            raise ValueError("You must specify at least 1 Producer to connect to.")

        self.reply_timeout = self._emc_config.get('reply_timeout', 10)
        self.ping_timeout = self._emc_config.get('ping_timeout', 2)
        self.sleep_time = self._emc_config.get('sleep_time', 5)

        # Setting these explicitly as they probably shouldn't be changed by a user
        # Unless we somehow integrate this with the strategy to allow creating
        # callbacks for the messages
        self.topics = [RPCMessageType.WHITELIST, RPCMessageType.ANALYZED_DF]

        self.start()

    def start(self):
        """
        Start the main internal loop in another thread to run coroutines
        """
        self._loop = asyncio.new_event_loop()

        if not self._thread:
            logger.info("Starting ExternalMessageConsumer")

            self._thread = Thread(target=self._loop.run_forever)
            self._thread.start()
            self._running = True
        else:
            raise RuntimeError("A loop is already running")

        self._main_task = asyncio.run_coroutine_threadsafe(self._main(), loop=self._loop)

    def shutdown(self):
        """
        Shutdown the loop, thread, and tasks
        """
        if self._thread and self._loop:
            logger.info("Stopping ExternalMessageConsumer")

            if self._sub_tasks:
                # Cancel sub tasks
                for task in self._sub_tasks:
                    task.cancel()

            if self._main_task:
                # Cancel the main task
                self._main_task.cancel()

            self._thread.join()

    async def _main(self):
        """
        The main task coroutine
        """
        lock = asyncio.Lock()

        try:
            # Create a connection to each producer
            self._sub_tasks = [
                self._loop.create_task(self._handle_producer_connection(producer, lock))
                for producer in self.producers
            ]

            await asyncio.gather(*self._sub_tasks)
        except asyncio.CancelledError:
            pass
        finally:
            # Stop the loop once we are done
            self._loop.stop()

    async def _handle_producer_connection(self, producer: Dict[str, Any], lock: asyncio.Lock):
        """
        Main connection loop for the consumer

        :param producer: Dictionary containing producer info: {'url': '', 'ws_token': ''}
        :param lock: An asyncio Lock
        """
        try:
            await self._create_connection(producer, lock)
        except asyncio.CancelledError:
            # Exit silently
            pass

    async def _create_connection(self, producer: Dict[str, Any], lock: asyncio.Lock):
        """
        Actually creates and handles the websocket connection, pinging on timeout
        and handling connection errors.

        :param producer: Dictionary containing producer info: {'url': '', 'ws_token': ''}
        :param lock: An asyncio Lock
        """
        while self._running:
            try:
                url, token = producer['url'], producer['ws_token']
                ws_url = f"{url}?token={token}"

                # This will raise InvalidURI if the url is bad
                async with websockets.connect(ws_url) as ws:
                    logger.info("Connection successful")
                    channel = WebSocketChannel(ws)

                    # Tell the producer we only want these topics
                    # Should always be the first thing we send
                    await channel.send(
                        self.compose_consumer_request(RPCRequestType.SUBSCRIBE, self.topics)
                    )

                    # Now receive data, if none is within the time limit, ping
                    while True:
                        try:
                            message = await asyncio.wait_for(
                                channel.recv(),
                                timeout=self.reply_timeout
                            )

                            async with lock:
                                # Handle the message
                                self.handle_producer_message(message)

                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                            # We haven't received data yet. Check the connection and continue.
                            try:
                                # ping
                                ping = await channel.ping()

                                await asyncio.wait_for(ping, timeout=self.ping_timeout)
                                logger.debug(f"Connection to {url} still alive...")

                                continue
                            except Exception:
                                logger.info(
                                    f"Ping error {url} - retrying in {self.sleep_time}s")
                                await asyncio.sleep(self.sleep_time)

                                break
                        except Exception as e:
                            logger.exception(e)
                            continue
            except (
                socket.gaierror,
                ConnectionRefusedError,
                websockets.exceptions.InvalidStatusCode
            ) as e:
                logger.error(f"Connection Refused - {e} retrying in {self.sleep_time}s")
                await asyncio.sleep(self.sleep_time)

                continue

            # Catch invalid ws_url, and break the loop
            except websockets.exceptions.InvalidURI as e:
                logger.error(f"{ws_url} is an invalid WebSocket URL - {e}")
                break

    def compose_consumer_request(self, type_: RPCRequestType, data: Any) -> Dict[str, Any]:
        """
        Create a request for sending to a producer

        :param type_: The RPCRequestType
        :param data: The data to send
        :returns: Dict[str, Any]
        """
        return {'type': type_, 'data': data}

    # How we do things here isn't set in stone. There seems to be some interest
    # in figuring out a better way, but we shall do this for now.
    def handle_producer_message(self, message: Dict[str, Any]):
        """
        Handles external messages from a Producer
        """
        # Should we have a default message type?
        message_type = message.get('type', RPCMessageType.STATUS)
        message_data = message.get('data')

        # We shouldn't get empty messages
        if message_data is None:
            return

        logger.debug(f"Received message of type {message_type}")

        # Handle Whitelists
        if message_type == RPCMessageType.WHITELIST:
            pairlist = message_data

            # Add the pairlist data to the DataProvider
            self._dp.set_producer_pairs(pairlist)

        # Handle analyzed dataframes
        elif message_type == RPCMessageType.ANALYZED_DF:
            key, value = message_data.get('key'), message_data.get('value')

            if key and value:
                pair, timeframe, candle_type = key

                # Convert the JSON to a pandas DataFrame
                dataframe = json_to_dataframe(value)

                # If set, remove the Entry and Exit signals from the Producer
                if self._emc_config.get('remove_entry_exit_signals', False):
                    dataframe = remove_entry_exit_signals(dataframe)

                # Add the dataframe to the dataprovider
                self._dp.add_external_df(pair, timeframe, dataframe, candle_type)
