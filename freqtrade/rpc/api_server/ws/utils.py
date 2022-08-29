from fastapi import WebSocket
# fastapi does not make this available through it, so import directly from starlette
from starlette.websockets import WebSocketState


async def is_websocket_alive(ws: WebSocket) -> bool:
    if (
        ws.application_state == WebSocketState.CONNECTED and
        ws.client_state == WebSocketState.CONNECTED
    ):
        return True
    return False
