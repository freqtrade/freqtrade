import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from freqtrade.rpc.api_server.deps import get_channel_manager
from freqtrade.rpc.api_server.ws.utils import is_websocket_alive


logger = logging.getLogger(__name__)

# Private router, protected by API Key authentication
router = APIRouter()


@router.websocket("/message/ws")
async def message_endpoint(
    ws: WebSocket,
    channel_manager=Depends(get_channel_manager)
):
    try:
        if is_websocket_alive(ws):
            logger.info(f"Consumer connected - {ws.client}")

            # TODO:
            # Return a channel ID, pass that instead of ws to the rest of the methods
            channel = await channel_manager.on_connect(ws)

            # Keep connection open until explicitly closed, and sleep
            try:
                while not channel.is_closed():
                    request = await channel.recv()

                    # This is where we'd parse the request. For now this should only
                    # be a list of topics to subscribe too. List[str]
                    # Maybe allow the consumer to update the topics subscribed
                    # during runtime?
                    logger.info(f"Consumer request - {request}")

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
