from pandas import DataFrame
from starlette.websockets import WebSocket, WebSocketState

from freqtrade.enums.signaltype import SignalTagType, SignalType


async def is_websocket_alive(ws: WebSocket) -> bool:
    if (
        ws.application_state == WebSocketState.CONNECTED and
        ws.client_state == WebSocketState.CONNECTED
    ):
        return True
    return False


def remove_entry_exit_signals(dataframe: DataFrame):
    dataframe[SignalType.ENTER_LONG.value] = 0
    dataframe[SignalType.EXIT_LONG.value] = 0
    dataframe[SignalType.ENTER_SHORT.value] = 0
    dataframe[SignalType.EXIT_SHORT.value] = 0
    dataframe[SignalTagType.ENTER_TAG.value] = None
    dataframe[SignalTagType.EXIT_TAG.value] = None
