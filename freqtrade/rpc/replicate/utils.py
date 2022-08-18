from starlette.websockets import WebSocket, WebSocketState


async def is_websocket_alive(ws: WebSocket) -> bool:
    if (
        ws.application_state == WebSocketState.CONNECTED and
        ws.client_state == WebSocketState.CONNECTED
    ):
        return True
    return False
