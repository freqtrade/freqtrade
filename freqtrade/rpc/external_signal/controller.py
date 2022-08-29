# """
# This module manages replicate mode communication
# """
# import asyncio
# import logging
# import secrets
# import socket
# from threading import Thread
# from typing import Any, Callable, Coroutine, Dict, Union
#
# import websockets
# from fastapi import Depends
# from fastapi import WebSocket as FastAPIWebSocket
# from fastapi import WebSocketDisconnect, status
# from janus import Queue as ThreadedQueue
#
# from freqtrade.enums import ExternalSignalModeType, LeaderMessageType, RPCMessageType
# from freqtrade.rpc import RPC, RPCHandler
# from freqtrade.rpc.external_signal.channel import ChannelManager
# from freqtrade.rpc.external_signal.types import MessageType
# from freqtrade.rpc.external_signal.utils import is_websocket_alive
#
#
# logger = logging.getLogger(__name__)
#
#
# class ExternalSignalController(RPCHandler):
#     """  This class handles all websocket communication """
#
#     def __init__(
#         self,
#         rpc: RPC,
#         config: Dict[str, Any],
#         api_server: Union[Any, None] = None
#     ) -> None:
#         """
#         Init the ExternalSignalController class, and init the super class RPCHandler
#         :param rpc: instance of RPC Helper class
#         :param config: Configuration object
#         :param api_server: The ApiServer object
#         :return: None
#         """
#         super().__init__(rpc, config)
#
#         self.freqtrade = rpc._freqtrade
#         self.api_server = api_server
#
#         if not self.api_server:
#             raise RuntimeError("The API server must be enabled for external signals to work")
#
#         self._loop = None
#         self._running = False
#         self._thread = None
#         self._queue = None
#
#         self._main_task = None
#         self._sub_tasks = None
#
#         self._message_handlers = {
#             LeaderMessageType.pairlist: self._rpc._handle_pairlist_message,
#             LeaderMessageType.analyzed_df: self._rpc._handle_analyzed_df_message,
#             LeaderMessageType.default: self._rpc._handle_default_message
#         }
#
#         self.channel_manager = ChannelManager()
#         self.external_signal_config = config.get('external_signal', {})
#
#         # What the config should look like
#         # "external_signal": {
#         #     "enabled": true,
#         #     "mode": "follower",
#         #     "leaders": [
#         #       {
#         #         "url": "ws://localhost:8080/signals/ws",
#         #         "api_token": "test"
#         #       }
#         #     ]
#         # }
#
#         # "external_signal": {
#         #     "enabled": true,
#         #     "mode": "leader",
#         #     "api_token": "test"
#         # }
#
#         self.mode = ExternalSignalModeType[
#             self.external_signal_config.get('mode', 'leader').lower()
#         ]
#
#         self.leaders_list = self.external_signal_config.get('leaders', [])
#         self.push_throttle_secs = self.external_signal_config.get('push_throttle_secs', 0.1)
#
#         self.reply_timeout = self.external_signal_config.get('follower_reply_timeout', 10)
#         self.ping_timeout = self.external_signal_config.get('follower_ping_timeout', 2)
#         self.sleep_time = self.external_signal_config.get('follower_sleep_time', 5)
#
#         # Validate external_signal_config here?
#
#         if self.mode == ExternalSignalModeType.follower and len(self.leaders_list) == 0:
#             raise ValueError("You must specify at least 1 leader in follower mode.")
#
#         # This is only used by the leader, the followers use the tokens specified
#         # in each of the leaders
#         # If you do not specify an API key in the config, one will be randomly
#         # generated and logged on startup
#         default_api_key = secrets.token_urlsafe(16)
#         self.secret_api_key = self.external_signal_config.get('api_token', default_api_key)
#
#         self.start()
#
#     def is_leader(self):
#         """
#         Leader flag
#         """
#         return self.enabled() and self.mode == ExternalSignalModeType.leader
#
#     def enabled(self):
#         """
#         Enabled flag
#         """
#         return self.external_signal_config.get('enabled', False)
#
#     def num_leaders(self):
#         """
#         The number of leaders we should be connected to
#         """
#         return len(self.leaders_list)
#
#     def start_threaded_loop(self):
#         """
#         Start the main internal loop in another thread to run coroutines
#         """
#         self._loop = asyncio.new_event_loop()
#
#         if not self._thread:
#             self._thread = Thread(target=self._loop.run_forever)
#             self._thread.start()
#             self._running = True
#         else:
#             raise RuntimeError("A loop is already running")
#
#     def submit_coroutine(self, coroutine: Coroutine):
#         """
#         Submit a coroutine to the threaded loop
#         """
#         if not self._running:
#             raise RuntimeError("Cannot schedule new futures after shutdown")
#
#         if not self._loop or not self._loop.is_running():
#             raise RuntimeError("Loop must be started before any function can"
#                                " be submitted")
#
#         return asyncio.run_coroutine_threadsafe(coroutine, self._loop)
#
#     def start(self):
#         """
#         Start the controller main loop
#         """
#         self.start_threaded_loop()
#         self._main_task = self.submit_coroutine(self.main())
#
#     async def shutdown(self):
#         """
#         Shutdown all tasks and close up
#         """
#         logger.info("Stopping rpc.externalsignalcontroller")
#
#         # Flip running flag
#         self._running = False
#
#         # Cancel sub tasks
#         for task in self._sub_tasks:
#             task.cancel()
#
#         # Then disconnect all channels
#         await self.channel_manager.disconnect_all()
#
#     def cleanup(self) -> None:
#         """
#         Cleanup pending module resources.
#         """
#         if self._thread:
#             if self._loop.is_running():
#                 self._main_task.cancel()
#             self._thread.join()
#
#     async def main(self):
#         """
#         Main coro
#
#         Start the loop based on what mode we're in
#         """
#         try:
#             if self.mode == ExternalSignalModeType.leader:
#                 logger.info("Starting rpc.externalsignalcontroller in Leader mode")
#
#                 await self.run_leader_mode()
#             elif self.mode == ExternalSignalModeType.follower:
#                 logger.info("Starting rpc.externalsignalcontroller in Follower mode")
#
#                 await self.run_follower_mode()
#
#         except asyncio.CancelledError:
#             # We're cancelled
#             await self.shutdown()
#         except Exception as e:
#             # Log the error
#             logger.error(f"Exception occurred in main task: {e}")
#             logger.exception(e)
#         finally:
#             # This coroutine is the last thing to be ended, so it should stop the loop
#             self._loop.stop()
#
#     def log_api_token(self):
#         """
#         Log the API token
#         """
#         logger.info("-" * 15)
#         logger.info(f"API_KEY: {self.secret_api_key}")
#         logger.info("-" * 15)
#
#     def send_msg(self, msg: MessageType) -> None:
#         """
#         Support RPC calls
#         """
#         if msg["type"] == RPCMessageType.EMIT_DATA:
#             message = msg.get("message")
#             if message:
#                 self.send_message(message)
#             else:
#                 logger.error(f"Message is empty! {msg}")
#
#     def send_message(self, msg: MessageType) -> None:
#         """
#         Broadcast message over all channels if there are any
#         """
#
#         if self.channel_manager.has_channels():
#             self._send_message(msg)
#         else:
#             logger.debug("No listening followers, skipping...")
#             pass
#
#     def _send_message(self, msg: MessageType):
#         """
#         Add data to the internal queue to be broadcasted. This func will block
#         if the queue is full. This is meant to be called in the main thread.
#         """
#         if self._queue:
#             queue = self._queue.sync_q
#             queue.put(msg)  # This will block if the queue is full
#         else:
#             logger.warning("Can not send data, leader loop has not started yet!")
#
#     async def send_initial_data(self, channel):
#         logger.info("Sending initial data through channel")
#
#         data = self._rpc._initial_leader_data()
#
#         for message in data:
#             await channel.send(message)
#
#     async def _handle_leader_message(self, message: MessageType):
#         """
#         Handle message received from a Leader
#         """
#         type = message.get("data_type", LeaderMessageType.default)
#         data = message.get("data")
#
#         handler: Callable = self._message_handlers[type]
#         handler(type, data)
#
#     # ----------------------------------------------------------------------
#
#     async def run_leader_mode(self):
#         """
#         Main leader coroutine
#
#         This starts all of the leader coros and registers the endpoint on
#         the ApiServer
#         """
#         self.register_leader_endpoint()
#         self.log_api_token()
#
#         self._sub_tasks = [
#             self._loop.create_task(self._broadcast_queue_data())
#         ]
#
#         return await asyncio.gather(*self._sub_tasks)
#
#     async def run_follower_mode(self):
#         """
#         Main follower coroutine
#
#         This starts all of the follower connection coros
#         """
#
#         rpc_lock = asyncio.Lock()
#
#         self._sub_tasks = [
#             self._loop.create_task(self._handle_leader_connection(leader, rpc_lock))
#             for leader in self.leaders_list
#         ]
#
#         return await asyncio.gather(*self._sub_tasks)
#
#     async def _broadcast_queue_data(self):
#         """
#         Loop over queue data and broadcast it
#         """
#         # Instantiate the queue in this coroutine so it's attached to our loop
#         self._queue = ThreadedQueue()
#         async_queue = self._queue.async_q
#
#         try:
#             while self._running:
#                 # Get data from queue
#                 data = await async_queue.get()
#
#                 # Broadcast it to everyone
#                 await self.channel_manager.broadcast(data)
#
#                 # Sleep
#                 await asyncio.sleep(self.push_throttle_secs)
#
#         except asyncio.CancelledError:
#             # Silently stop
#             pass
#
#     async def get_api_token(
#         self,
#         websocket: FastAPIWebSocket,
#         token: Union[str, None] = None
#     ):
#         """
#         Extract the API key from query param. Must match the
#         set secret_api_key or the websocket connection will be closed.
#         """
#         if token == self.secret_api_key:
#             return token
#         else:
#             logger.info("Denying websocket request...")
#             await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
#
#     def register_leader_endpoint(self, path: str = "/signals/ws"):
#         """
#         Attach and start the main leader loop to the ApiServer
#
#         :param path: The endpoint path
#         """
#         if not self.api_server:
#             raise RuntimeError("The leader needs the ApiServer to be active")
#
#         # The endpoint function for running the main leader loop
#         @self.api_server.app.websocket(path)
#         async def leader_endpoint(
#             websocket: FastAPIWebSocket,
#             api_key: str = Depends(self.get_api_token)
#         ):
#             await self.leader_endpoint_loop(websocket)
#
#     async def leader_endpoint_loop(self, websocket: FastAPIWebSocket):
#         """
#         The WebSocket endpoint served by the ApiServer. This handles connections,
#         and adding them to the channel manager.
#         """
#         try:
#             if is_websocket_alive(websocket):
#                 logger.info(f"Follower connected - {websocket.client}")
#                 channel = await self.channel_manager.on_connect(websocket)
#
#                 # Send initial data here
#                 # Data is being broadcasted right away as soon as startup,
#                 # we may not have to send initial data at all. Further testing
#                 # required.
#                 await self.send_initial_data(channel)
#
#                 # Keep connection open until explicitly closed, and sleep
#                 try:
#                     while not channel.is_closed():
#                         request = await channel.recv()
#                         logger.info(f"Follower request - {request}")
#
#                 except WebSocketDisconnect:
#                     # Handle client disconnects
#                     logger.info(f"Follower disconnected - {websocket.client}")
#                     await self.channel_manager.on_disconnect(websocket)
#                 except Exception as e:
#                     logger.info(f"Follower connection failed - {websocket.client}")
#                     logger.exception(e)
#                     # Handle cases like -
#                     # RuntimeError('Cannot call "send" once a closed message has been sent')
#                     await self.channel_manager.on_disconnect(websocket)
#
#         except Exception:
#             logger.error(f"Failed to serve - {websocket.client}")
#             await self.channel_manager.on_disconnect(websocket)
#
#     async def _handle_leader_connection(self, leader, lock):
#         """
#         Given a leader, connect and wait on data. If connection is lost,
#         it will attempt to reconnect.
#         """
#         try:
#             url, token = leader["url"], leader["api_token"]
#             websocket_url = f"{url}?token={token}"
#
#             logger.info(f"Attempting to connect to Leader at: {url}")
#             while True:
#                 try:
#                     async with websockets.connect(websocket_url) as ws:
#                         channel = await self.channel_manager.on_connect(ws)
#                         logger.info(f"Connection to Leader at {url} successful")
#                         while True:
#                             try:
#                                 data = await asyncio.wait_for(
#                                     channel.recv(),
#                                     timeout=self.reply_timeout
#                                 )
#                             except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
#                                 # We haven't received data yet. Check the connection and continue.
#                                 try:
#                                     # ping
#                                     ping = await channel.ping()
#                                     await asyncio.wait_for(ping, timeout=self.ping_timeout)
#                                     logger.debug(f"Connection to {url} still alive...")
#                                     continue
#                                 except Exception:
#                                     logger.info(
#                                         f"Ping error {url} - retrying in {self.sleep_time}s")
#                                     asyncio.sleep(self.sleep_time)
#                                     break
#
#                             async with lock:
#                                 # Acquire lock so only 1 coro handling at a time
#                                 # as we call the RPC module in the main thread
#                                 await self._handle_leader_message(data)
#
#                 except (socket.gaierror, ConnectionRefusedError):
#                     logger.info(f"Connection Refused - retrying connection in {self.sleep_time}s")
#                     await asyncio.sleep(self.sleep_time)
#                     continue
#                 except websockets.exceptions.InvalidStatusCode as e:
#                     logger.error(f"Connection Refused - {e}")
#                     await asyncio.sleep(self.sleep_time)
#                     continue
#
#         except asyncio.CancelledError:
#             pass
