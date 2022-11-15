import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed

from freqtrade.enums import RPCMessageType, RPCRequestType
from freqtrade.rpc.api_server.api_auth import validate_ws_token
from freqtrade.rpc.api_server.deps import get_message_stream, get_rpc
from freqtrade.rpc.api_server.ws import WebSocketChannel
from freqtrade.rpc.api_server.ws.message_stream import MessageStream
from freqtrade.rpc.api_server.ws_schemas import (WSAnalyzedDFMessage, WSMessageSchema,
                                                 WSRequestSchema, WSWhitelistMessage)
from freqtrade.rpc.rpc import RPC


logger = logging.getLogger(__name__)

# Private router, protected by API Key authentication
router = APIRouter()


# async def is_websocket_alive(ws: WebSocket) -> bool:
#     """
#     Check if a FastAPI Websocket is still open
#     """
#     if (
#         ws.application_state == WebSocketState.CONNECTED and
#         ws.client_state == WebSocketState.CONNECTED
#     ):
#         return True
#     return False


class WebSocketChannelClosed(Exception):
    """
    General WebSocket exception to signal closing the channel
    """
    pass


async def channel_reader(channel: WebSocketChannel, rpc: RPC):
    """
    Iterate over the messages from the channel and process the request
    """
    try:
        async for message in channel:
            await _process_consumer_request(message, channel, rpc)
    except (
        RuntimeError,
        WebSocketDisconnect,
        ConnectionClosed
    ):
        raise WebSocketChannelClosed
    except asyncio.CancelledError:
        return


async def channel_broadcaster(channel: WebSocketChannel, message_stream: MessageStream):
    """
    Iterate over messages in the message stream and send them
    """
    try:
        async for message in message_stream:
            await channel.send(message)
    except (
        RuntimeError,
        WebSocketDisconnect,
        ConnectionClosed
    ):
        raise WebSocketChannelClosed
    except asyncio.CancelledError:
        return


async def _process_consumer_request(
    request: Dict[str, Any],
    channel: WebSocketChannel,
    rpc: RPC
):
    """
    Validate and handle a request from a websocket consumer
    """
    # Validate the request, makes sure it matches the schema
    try:
        websocket_request = WSRequestSchema.parse_obj(request)
    except ValidationError as e:
        logger.error(f"Invalid request from {channel}: {e}")
        return

    type, data = websocket_request.type, websocket_request.data
    response: WSMessageSchema

    logger.debug(f"Request of type {type} from {channel}")

    # If we have a request of type SUBSCRIBE, set the topics in this channel
    if type == RPCRequestType.SUBSCRIBE:
        # If the request is empty, do nothing
        if not data:
            return

        # If all topics passed are a valid RPCMessageType, set subscriptions on channel
        if all([any(x.value == topic for x in RPCMessageType) for topic in data]):
            channel.set_subscriptions(data)

        # We don't send a response for subscriptions
        return

    elif type == RPCRequestType.WHITELIST:
        # Get whitelist
        whitelist = rpc._ws_request_whitelist()

        # Format response
        response = WSWhitelistMessage(data=whitelist)
        # Send it back
        await channel.send(response.dict(exclude_none=True))

    elif type == RPCRequestType.ANALYZED_DF:
        limit = None

        if data:
            # Limit the amount of candles per dataframe to 'limit' or 1500
            limit = max(data.get('limit', 1500), 1500)

        # For every pair in the generator, send a separate message
        for message in rpc._ws_request_analyzed_df(limit):
            # Format response
            response = WSAnalyzedDFMessage(data=message)
            await channel.send(response.dict(exclude_none=True))


@router.websocket("/message/ws")
async def message_endpoint(
    websocket: WebSocket,
    token: str = Depends(validate_ws_token),
    rpc: RPC = Depends(get_rpc),
    message_stream: MessageStream = Depends(get_message_stream)
):
    async with WebSocketChannel(websocket).connect() as channel:
        try:
            logger.info(f"Channel connected - {channel}")

            channel_tasks = asyncio.gather(
                channel_reader(channel, rpc),
                channel_broadcaster(channel, message_stream)
            )
            await channel_tasks

        finally:
            logger.info(f"Channel disconnected - {channel}")
            channel_tasks.cancel()


# @router.websocket("/message/ws")
# async def message_endpoint(
#     ws: WebSocket,
#     rpc: RPC = Depends(get_rpc),
#     channel_manager=Depends(get_channel_manager),
#     token: str = Depends(validate_ws_token)
# ):
#     """
#     Message WebSocket endpoint, facilitates sending RPC messages
#     """
#     try:
#         channel = await channel_manager.on_connect(ws)
#         if await is_websocket_alive(ws):

#             logger.info(f"Consumer connected - {channel}")

#             # Keep connection open until explicitly closed, and process requests
#             try:
#                 while not channel.is_closed():
#                     request = await channel.recv()

#                     # Process the request here
#                     await _process_consumer_request(request, channel, rpc, channel_manager)

#             except (WebSocketDisconnect, WebSocketException):
#                 # Handle client disconnects
#                 logger.info(f"Consumer disconnected - {channel}")
#             except RuntimeError:
#                 # Handle cases like -
#                 # RuntimeError('Cannot call "send" once a closed message has been sent')
#                 pass
#             except Exception as e:
#                 logger.info(f"Consumer connection failed - {channel}: {e}")
#                 logger.debug(e, exc_info=e)

#     except RuntimeError:
#         # WebSocket was closed
#         # Do nothing
#         pass
#     except Exception as e:
#         logger.error(f"Failed to serve - {ws.client}")
#         # Log tracebacks to keep track of what errors are happening
#         logger.exception(e)
#     finally:
#         if channel:
#             await channel_manager.on_disconnect(ws)
