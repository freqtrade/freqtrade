import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from freqtrade.enums import RPCMessageType, RPCRequestType
from freqtrade.rpc.api_server.deps import get_channel_manager, get_rpc
from freqtrade.rpc.api_server.ws.channel import WebSocketChannel
from freqtrade.rpc.api_server.ws.utils import is_websocket_alive
from freqtrade.rpc.rpc import RPC


# from typing import Any, Dict


logger = logging.getLogger(__name__)

# Private router, protected by API Key authentication
router = APIRouter()


async def _process_consumer_request(
    request: Dict[str, Any],
    channel: WebSocketChannel,
    rpc: RPC
):
    type, data = request.get('type'), request.get('data')

    # If we have a request of type SUBSCRIBE, set the topics in this channel
    if type == RPCRequestType.SUBSCRIBE:
        # If the request is empty, do nothing
        if not data:
            return

        if not isinstance(data, list):
            logger.error(f"Improper request from channel: {channel} - {request}")
            return

        # If all topics passed are a valid RPCMessageType, set subscriptions on channel
        if all([any(x.value == topic for x in RPCMessageType) for topic in data]):

            logger.debug(f"{channel} subscribed to topics: {data}")
            channel.set_subscriptions(data)

    elif type == RPCRequestType.INITIAL_DATA:
        # Acquire the data
        initial_data = rpc._ws_initial_data()

        # We now loop over it sending it in pieces
        whitelist_data, analyzed_df = initial_data.get('whitelist'), initial_data.get('analyzed_df')

        if whitelist_data:
            await channel.send({"type": RPCMessageType.WHITELIST, "data": whitelist_data})

        if analyzed_df:
            for pair, message in analyzed_df.items():
                await channel.send({"type": RPCMessageType.ANALYZED_DF, "data": message})


@router.websocket("/message/ws")
async def message_endpoint(
    ws: WebSocket,
    rpc: RPC = Depends(get_rpc),
    channel_manager=Depends(get_channel_manager),
):
    try:
        if is_websocket_alive(ws):
            logger.info(f"Consumer connected - {ws.client}")

            # TODO:
            # Return a channel ID, pass that instead of ws to the rest of the methods
            channel = await channel_manager.on_connect(ws)

            # Keep connection open until explicitly closed, and process requests
            try:
                while not channel.is_closed():
                    request = await channel.recv()

                    # Process the request here. Should this be a method of RPC?
                    logger.info(f"Request: {request}")
                    await _process_consumer_request(request, channel, rpc)

            except WebSocketDisconnect:
                # Handle client disconnects
                logger.info(f"Consumer disconnected - {ws.client}")
                await channel_manager.on_disconnect(ws)
            except Exception as e:
                logger.info(f"Consumer connection failed - {ws.client}")
                logger.exception(e)
                # Handle cases like -
                # RuntimeError('Cannot call "send" once a closed message has been sent')
                await channel_manager.on_disconnect(ws)

    except Exception:
        logger.error(f"Failed to serve - {ws.client}")
        await channel_manager.on_disconnect(ws)
