"""
This module manages replicate mode communication
"""
import asyncio
import logging
import secrets
import socket
import traceback
from threading import Event, Thread
from typing import Any, Coroutine, Dict, Union

import websockets
from fastapi import Depends
from fastapi import WebSocket as FastAPIWebSocket
from fastapi import WebSocketDisconnect, status

from freqtrade.enums import LeaderMessageType, ReplicateModeType, RPCMessageType
from freqtrade.rpc import RPC, RPCHandler
from freqtrade.rpc.replicate.channel import ChannelManager
from freqtrade.rpc.replicate.thread_queue import Queue as ThreadedQueue
from freqtrade.rpc.replicate.types import MessageType
from freqtrade.rpc.replicate.utils import is_websocket_alive


logger = logging.getLogger(__name__)


class ReplicateController(RPCHandler):
    """  This class handles all websocket communication """

    def __init__(
        self,
        rpc: RPC,
        config: Dict[str, Any],
        api_server: Union[Any, None] = None
    ) -> None:
        """
        Init the ReplicateRPC class, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        super().__init__(rpc, config)

        self.freqtrade = rpc._freqtrade
        self.api_server = api_server

        if not self.api_server:
            raise RuntimeError("The API server must be enabled for replicate to work")

        self._loop = None
        self._running = False
        self._thread = None
        self._queue = None

        self._stop_event = Event()
        self._follower_tasks = None

        self.channel_manager = ChannelManager()

        self.replicate_config = config.get('replicate', {})

        # What the config should look like
        # "replicate": {
        #     "enabled": true,
        #     "mode": "follower",
        #     "leaders": [
        #       {
        #         "url": "ws://localhost:8080/replicate/ws",
        #         "token": "test"
        #       }
        #     ]
        # }

        # "replicate": {
        #     "enabled": true,
        #     "mode": "leader",
        #     "api_key": "test"
        # }

        self.mode = ReplicateModeType[self.replicate_config.get('mode', 'leader').lower()]

        self.leaders_list = self.replicate_config.get('leaders', [])
        self.push_throttle_secs = self.replicate_config.get('push_throttle_secs', 0.1)

        self.reply_timeout = self.replicate_config.get('follower_reply_timeout', 10)
        self.ping_timeout = self.replicate_config.get('follower_ping_timeout', 2)
        self.sleep_time = self.replicate_config.get('follower_sleep_time', 5)

        if self.mode == ReplicateModeType.follower and len(self.leaders_list) == 0:
            raise ValueError("You must specify at least 1 leader in follower mode.")

        # This is only used by the leader, the followers use the tokens specified
        # in each of the leaders
        # If you do not specify an API key in the config, one will be randomly
        # generated and logged on startup
        default_api_key = secrets.token_urlsafe(16)
        self.secret_api_key = self.replicate_config.get('api_key', default_api_key)

        self.start_threaded_loop()

        self.start()

    def start_threaded_loop(self):
        """
        Start the main internal loop in another thread to run coroutines
        """
        self._loop = asyncio.new_event_loop()

        if not self._thread:
            self._thread = Thread(target=self._loop.run_forever)
            self._thread.start()
            self._running = True
        else:
            raise RuntimeError("A loop is already running")

    def submit_coroutine(self, coroutine: Coroutine):
        """
        Submit a coroutine to the threaded loop
        """
        if not self._running:
            raise RuntimeError("Cannot schedule new futures after shutdown")

        if not self._loop or not self._loop.is_running():
            raise RuntimeError("Loop must be started before any function can"
                               " be submitted")

        try:
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except Exception as e:
            logger.error(f"Error running coroutine - {str(e)}")
            return None

    async def main_loop(self):
        """
        Main loop coro

        Start the loop based on what mode we're in
        """
        try:
            if self.mode == ReplicateModeType.leader:
                await self.leader_loop()
            elif self.mode == ReplicateModeType.follower:
                await self.follower_loop()

        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self._loop.stop()

    def start(self):
        """
        Start the controller main loop
        """
        self.submit_coroutine(self.main_loop())

    def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        """
        if self._thread:
            if self._loop.is_running():

                self._running = False

                # Tell all coroutines submitted to the loop they're cancelled
                pending = asyncio.all_tasks(loop=self._loop)
                for task in pending:
                    task.cancel()

                self._loop.call_soon_threadsafe(self.channel_manager.disconnect_all)

            self._thread.join()

    def send_msg(self, msg: MessageType) -> None:
        """
        Support RPC calls
        """
        if msg["type"] == RPCMessageType.EMIT_DATA:
            message = msg.get("message")
            if message:
                self.send_message(message)
            else:
                logger.error(f"Message is empty! {msg}")

    def send_message(self, msg: MessageType) -> None:
        """ Broadcast message over all channels if there are any """

        if self.channel_manager.has_channels():
            self._send_message(msg)
        else:
            logger.debug("No listening followers, skipping...")
            pass

    def _send_message(self, msg: MessageType):
        """
        Add data to the internal queue to be broadcasted. This func will block
        if the queue is full. This is meant to be called in the main thread.
        """
        if self._queue:
            queue = self._queue.sync_q
            queue.put(msg)  # This will block if the queue is full
        else:
            logger.warning("Can not send data, leader loop has not started yet!")

    def is_leader(self):
        """
        Leader flag
        """
        return self.enabled() and self.mode == ReplicateModeType.leader

    def enabled(self):
        """
        Enabled flag
        """
        return self.replicate_config.get('enabled', False)

    # ----------------------- LEADER LOGIC ------------------------------

    async def leader_loop(self):
        """
        Main leader coroutine

        This starts all of the leader coros and registers the endpoint on
        the ApiServer
        """
        logger.info("Running rpc.replicate in Leader mode")
        logger.info("-" * 15)
        logger.info(f"API_KEY: {self.secret_api_key}")
        logger.info("-" * 15)

        self.register_leader_endpoint()

        try:
            await self._broadcast_queue_data()
        except Exception as e:
            logger.error("Exception occurred in Leader loop: ")
            logger.exception(e)

    async def _broadcast_queue_data(self):
        """
        Loop over queue data and broadcast it
        """
        # Instantiate the queue in this coroutine so it's attached to our loop
        self._queue = ThreadedQueue()
        async_queue = self._queue.async_q

        try:
            while self._running:
                # Get data from queue
                data = await async_queue.get()

                # Broadcast it to everyone
                await self.channel_manager.broadcast(data)

                # Sleep
                await asyncio.sleep(self.push_throttle_secs)

        except asyncio.CancelledError:
            # Silently stop
            pass
        except Exception as e:
            logger.exception(e)

    async def get_api_token(
        self,
        websocket: FastAPIWebSocket,
        token: Union[str, None] = None
    ):
        """
        Extract the API key from query param. Must match the
        set secret_api_key or the websocket connection will be closed.
        """
        if token == self.secret_api_key:
            return token
        else:
            logger.info("Denying websocket request...")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)

    def register_leader_endpoint(self, path: str = "/replicate/ws"):
        """
        Attach and start the main leader loop to the ApiServer

        :param path: The endpoint path
        """
        if not self.api_server:
            raise RuntimeError("The leader needs the ApiServer to be active")

        # The endpoint function for running the main leader loop
        @self.api_server.app.websocket(path)
        async def leader_endpoint(
            websocket: FastAPIWebSocket,
            api_key: str = Depends(self.get_api_token)
        ):
            await self.leader_endpoint_loop(websocket)

    async def leader_endpoint_loop(self, websocket: FastAPIWebSocket):
        """
        The WebSocket endpoint served by the ApiServer. This handles connections,
        and adding them to the channel manager.
        """
        try:
            if is_websocket_alive(websocket):
                logger.info(f"Follower connected - {websocket.client}")
                channel = await self.channel_manager.on_connect(websocket)

                # Send initial data here
                # Data is being broadcasted right away as soon as startup,
                # we may not have to send initial data at all. Further testing
                # required.

                await self.send_initial_data(channel)

                # Keep connection open until explicitly closed, and sleep
                try:
                    while not channel.is_closed():
                        request = await channel.recv()
                        logger.info(f"Follower request - {request}")

                except WebSocketDisconnect:
                    # Handle client disconnects
                    logger.info(f"Follower disconnected - {websocket.client}")
                    await self.channel_manager.on_disconnect(websocket)
                except Exception as e:
                    logger.info(f"Follower connection failed - {websocket.client}")
                    logger.exception(e)
                    # Handle cases like -
                    # RuntimeError('Cannot call "send" once a closed message has been sent')
                    await self.channel_manager.on_disconnect(websocket)

        except Exception:
            logger.error(f"Failed to serve - {websocket.client}")
            await self.channel_manager.on_disconnect(websocket)

    async def send_initial_data(self, channel):
        logger.info("Sending initial data through channel")

        # We first send pairlist data
        initial_data = {
            "data_type": LeaderMessageType.pairlist,
            "data": self.freqtrade.pairlists.whitelist
        }

        await channel.send(initial_data)

    # -------------------------------FOLLOWER LOGIC----------------------------

    async def follower_loop(self):
        """
        Main follower coroutine

        This starts all of the follower connection coros
        """
        logger.info("Starting rpc.replicate in Follower mode")

        responses = await self._connect_to_leaders()

        # Eventually add the ability to send requests to the Leader
        # await self._send_requests()

        for result in responses:
            if isinstance(result, Exception):
                logger.debug(f"Exception in Follower loop: {result}")
                traceback_message = ''.join(traceback.format_tb(result.__traceback__))
                logger.error(traceback_message)

    async def _handle_leader_message(self, message: MessageType):
        """
        Handle message received from a Leader
        """
        type = message.get("data_type")
        data = message.get("data")

        self._rpc._handle_emitted_data(type, data)

    async def _connect_to_leaders(self):
        """
        For each leader in `self.leaders_list` create a connection and
        listen for data.
        """
        rpc_lock = asyncio.Lock()

        logger.info("Starting connections to Leaders...")

        self.follower_tasks = [
            self._loop.create_task(self._handle_leader_connection(leader, rpc_lock))
            for leader in self.leaders_list
        ]
        return await asyncio.gather(*self.follower_tasks, return_exceptions=True)

    async def _handle_leader_connection(self, leader, lock):
        """
        Given a leader, connect and wait on data. If connection is lost,
        it will attempt to reconnect.
        """
        try:
            url, token = leader["url"], leader["token"]
            websocket_url = f"{url}?token={token}"

            logger.info(f"Attempting to connect to Leader at: {url}")
            # TODO: limit the amount of connection retries
            while True:
                try:
                    async with websockets.connect(websocket_url) as ws:
                        channel = await self.channel_manager.on_connect(ws)
                        logger.info(f"Connection to Leader at {url} successful")
                        while True:
                            try:
                                data = await asyncio.wait_for(
                                    channel.recv(),
                                    timeout=self.reply_timeout
                                )
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
                                    asyncio.sleep(self.sleep_time)
                                    break

                            async with lock:
                                # Acquire lock so only 1 coro handling at a time
                                # as we might call the RPC module in the main thread
                                await self._handle_leader_message(data)

                except socket.gaierror:
                    logger.info(f"Socket error - retrying connection in {self.sleep_time}s")
                    await asyncio.sleep(self.sleep_time)
                    continue
                except ConnectionRefusedError:
                    logger.info(f"Connection Refused - retrying connection in {self.sleep_time}s")
                    await asyncio.sleep(self.sleep_time)
                    continue

        except asyncio.CancelledError:
            pass
