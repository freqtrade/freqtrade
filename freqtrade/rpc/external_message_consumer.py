"""
ExternalMessageConsumer module

Main purpose is to connect to external bot's message websocket to consume data
from it
"""
import asyncio
import logging
import socket
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, List, TypedDict

import websockets
from pydantic import ValidationError

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import RPCMessageType
from freqtrade.misc import remove_entry_exit_signals
from freqtrade.rpc.api_server.ws import WebSocketChannel
from freqtrade.rpc.api_server.ws_schemas import (WSAnalyzedDFMessage, WSAnalyzedDFRequest,
                                                 WSMessageSchema, WSRequestSchema,
                                                 WSSubscribeRequest, WSWhitelistMessage,
                                                 WSWhitelistRequest)


if TYPE_CHECKING:
    import websockets.connect


class Producer(TypedDict):
    name: str
    host: str
    port: int
    ws_token: str


logger = logging.getLogger(__name__)


class ExternalMessageConsumer:
    """
    The main controller class for consuming external messages from
    other freqtrade bot's
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
        self.producers: List[Producer] = self._emc_config.get('producers', [])

        self.wait_timeout = self._emc_config.get('wait_timeout', 300)  # in seconds
        self.ping_timeout = self._emc_config.get('ping_timeout', 10)  # in seconds
        self.sleep_time = self._emc_config.get('sleep_time', 10)  # in seconds

        # The amount of candles per dataframe on the initial request
        self.initial_candle_limit = self._emc_config.get('initial_candle_limit', 1500)

        # Message size limit, in megabytes. Default 8mb, Use bitwise operator << 20 to convert
        # as the websockets client expects bytes.
        self.message_size_limit = (self._emc_config.get('message_size_limit', 8) << 20)

        # Setting these explicitly as they probably shouldn't be changed by a user
        # Unless we somehow integrate this with the strategy to allow creating
        # callbacks for the messages
        self.topics = [RPCMessageType.WHITELIST, RPCMessageType.ANALYZED_DF]

        # Allow setting data for each initial request
        self._initial_requests: List[WSRequestSchema] = [
            WSSubscribeRequest(data=self.topics),
            WSWhitelistRequest(),
            WSAnalyzedDFRequest()
        ]

        # Specify which function to use for which RPCMessageType
        self._message_handlers: Dict[str, Callable[[str, WSMessageSchema], None]] = {
            RPCMessageType.WHITELIST: self._consume_whitelist_message,
            RPCMessageType.ANALYZED_DF: self._consume_analyzed_df_message,
        }

        self.start()

    def start(self):
        """
        Start the main internal loop in another thread to run coroutines
        """
        if self._thread and self._loop:
            return

        logger.info("Starting ExternalMessageConsumer")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever)
        self._running = True
        self._thread.start()

        self._main_task = asyncio.run_coroutine_threadsafe(self._main(), loop=self._loop)

    def shutdown(self):
        """
        Shutdown the loop, thread, and tasks
        """
        if self._thread and self._loop:
            logger.info("Stopping ExternalMessageConsumer")
            self._running = False

            if self._sub_tasks:
                # Cancel sub tasks
                for task in self._sub_tasks:
                    task.cancel()

            if self._main_task:
                # Cancel the main task
                self._main_task.cancel()

            self._thread.join()

            self._thread = None
            self._loop = None
            self._sub_tasks = None
            self._main_task = None

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

    async def _handle_producer_connection(self, producer: Producer, lock: asyncio.Lock):
        """
        Main connection loop for the consumer

        :param producer: Dictionary containing producer info
        :param lock: An asyncio Lock
        """
        try:
            await self._create_connection(producer, lock)
        except asyncio.CancelledError:
            # Exit silently
            pass

    async def _create_connection(self, producer: Producer, lock: asyncio.Lock):
        """
        Actually creates and handles the websocket connection, pinging on timeout
        and handling connection errors.

        :param producer: Dictionary containing producer info
        :param lock: An asyncio Lock
        """
        while self._running:
            try:
                host, port = producer['host'], producer['port']
                token = producer['ws_token']
                name = producer['name']
                ws_url = f"ws://{host}:{port}/api/v1/message/ws?token={token}"

                # This will raise InvalidURI if the url is bad
                async with websockets.connect(ws_url, max_size=self.message_size_limit) as ws:
                    channel = WebSocketChannel(ws, channel_id=name)

                    logger.info(f"Producer connection success - {channel}")

                    # Now request the initial data from this Producer
                    for request in self._initial_requests:
                        await channel.send(
                            request.dict(exclude_none=True)
                        )

                    # Now receive data, if none is within the time limit, ping
                    await self._receive_messages(channel, producer, lock)

            except (websockets.exceptions.InvalidURI, ValueError) as e:
                logger.error(f"{ws_url} is an invalid WebSocket URL - {e}")
                break

            except (
                socket.gaierror,
                ConnectionRefusedError,
                websockets.exceptions.InvalidStatusCode,
                websockets.exceptions.InvalidMessage
            ) as e:
                logger.error(f"Connection Refused - {e} retrying in {self.sleep_time}s")
                await asyncio.sleep(self.sleep_time)
                continue

            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK
            ):
                # Just keep trying to connect again indefinitely
                await asyncio.sleep(self.sleep_time)
                continue

            except Exception as e:
                # An unforseen error has occurred, log and continue
                logger.error("Unexpected error has occurred:")
                logger.exception(e)
                continue

    async def _receive_messages(
        self,
        channel: WebSocketChannel,
        producer: Producer,
        lock: asyncio.Lock
    ):
        """
        Loop to handle receiving messages from a Producer

        :param channel: The WebSocketChannel object for the WebSocket
        :param producer: Dictionary containing producer info
        :param lock: An asyncio Lock
        """
        while self._running:
            try:
                message = await asyncio.wait_for(
                    channel.recv(),
                    timeout=self.wait_timeout
                )

                try:
                    async with lock:
                        # Handle the message
                        self.handle_producer_message(producer, message)
                except Exception as e:
                    logger.exception(f"Error handling producer message: {e}")

            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                # We haven't received data yet. Check the connection and continue.
                try:
                    # ping
                    ping = await channel.ping()

                    await asyncio.wait_for(ping, timeout=self.ping_timeout)
                    logger.debug(f"Connection to {channel} still alive...")

                    continue
                except Exception as e:
                    logger.warning(f"Ping error {channel} - retrying in {self.sleep_time}s")
                    logger.debug(e, exc_info=e)
                    await asyncio.sleep(self.sleep_time)

                    break

    def handle_producer_message(self, producer: Producer, message: Dict[str, Any]):
        """
        Handles external messages from a Producer
        """
        producer_name = producer.get('name', 'default')

        try:
            producer_message = WSMessageSchema.parse_obj(message)
        except ValidationError as e:
            logger.error(f"Invalid message from `{producer_name}`: {e}")
            return

        if not producer_message.data:
            logger.error(f"Empty message received from `{producer_name}`")
            return

        logger.debug(f"Received message of type `{producer_message.type}` from `{producer_name}`")

        message_handler = self._message_handlers.get(producer_message.type)

        if not message_handler:
            logger.info(f"Received unhandled message: `{producer_message.data}`, ignoring...")
            return

        message_handler(producer_name, producer_message)

    def _consume_whitelist_message(self, producer_name: str, message: WSMessageSchema):
        try:
            # Validate the message
            whitelist_message = WSWhitelistMessage.parse_obj(message)
        except ValidationError as e:
            logger.error(f"Invalid message from `{producer_name}`: {e}")
            return

        # Add the pairlist data to the DataProvider
        self._dp._set_producer_pairs(whitelist_message.data, producer_name=producer_name)

        logger.debug(f"Consumed message from `{producer_name}` of type `RPCMessageType.WHITELIST`")

    def _consume_analyzed_df_message(self, producer_name: str, message: WSMessageSchema):
        try:
            df_message = WSAnalyzedDFMessage.parse_obj(message)
        except ValidationError as e:
            logger.error(f"Invalid message from `{producer_name}`: {e}")
            return

        key = df_message.data.key
        df = df_message.data.df
        la = df_message.data.la

        pair, timeframe, candle_type = key

        # If set, remove the Entry and Exit signals from the Producer
        if self._emc_config.get('remove_entry_exit_signals', False):
            df = remove_entry_exit_signals(df)

        # Add the dataframe to the dataprovider
        self._dp._add_external_df(pair, df,
                                  last_analyzed=la,
                                  timeframe=timeframe,
                                  candle_type=candle_type,
                                  producer_name=producer_name)

        logger.debug(
            f"Consumed message from `{producer_name}` of type `RPCMessageType.ANALYZED_DF`")
