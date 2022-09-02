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

    logger.debug(f"Request of type {type} from {channel}")

    # If we have a request of type SUBSCRIBE, set the topics in this channel
    if type == RPCRequestType.SUBSCRIBE:
        # If the request is empty, do nothing
        if not data:
            return

        if not isinstance(data, list):
            logger.error(f"Improper subscribe request from channel: {channel} - {request}")
            return

        # If all topics passed are a valid RPCMessageType, set subscriptions on channel
        if all([any(x.value == topic for x in RPCMessageType) for topic in data]):

            logger.debug(f"{channel} subscribed to topics: {data}")
            channel.set_subscriptions(data)

    elif type == RPCRequestType.WHITELIST:
        # They requested the whitelist
        whitelist = rpc._ws_request_whitelist()

        await channel.send({"type": RPCMessageType.WHITELIST, "data": whitelist})

    elif type == RPCRequestType.ANALYZED_DF:
        limit = None

        if data:
            # Limit the amount of candles per dataframe to 'limit' or 1500
            limit = max(data.get('limit', 500), 1500)

        # They requested the full historical analyzed dataframes
        analyzed_df = rpc._ws_request_analyzed_df(limit)

        logger.debug(f"ANALYZED_DF RESULT: {analyzed_df}")

        # For every dataframe, send as a separate message
        for _, message in analyzed_df.items():
            await channel.send({"type": RPCMessageType.ANALYZED_DF, "data": message})


@router.websocket("/message/ws")
async def message_endpoint(
    ws: WebSocket,
    rpc: RPC = Depends(get_rpc),
    channel_manager=Depends(get_channel_manager),
):
    try:
        if is_websocket_alive(ws):
            # TODO:
            # Return a channel ID, pass that instead of ws to the rest of the methods
            channel = await channel_manager.on_connect(ws)

            logger.info(f"Consumer connected - {channel}")

            # Keep connection open until explicitly closed, and process requests
            try:
                while not channel.is_closed():
                    request = await channel.recv()

                    # Process the request here
                    await _process_consumer_request(request, channel, rpc)

            except WebSocketDisconnect:
                # Handle client disconnects
                logger.info(f"Consumer disconnected - {channel}")
                await channel_manager.on_disconnect(ws)
            except Exception as e:
                logger.info(f"Consumer connection failed - {channel}")
                logger.exception(e)
                # Handle cases like -
                # RuntimeError('Cannot call "send" once a closed message has been sent')
                await channel_manager.on_disconnect(ws)

    except Exception as e:
        logger.error(f"Failed to serve - {ws.client}")
        # Log tracebacks to keep track of what errors are happening
        logger.exception(e)
        await channel_manager.on_disconnect(ws)
