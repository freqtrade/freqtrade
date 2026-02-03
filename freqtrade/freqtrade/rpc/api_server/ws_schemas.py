from datetime import datetime
from typing import Any, TypedDict

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict

from freqtrade.constants import PairWithTimeframe
from freqtrade.enums import RPCMessageType, RPCRequestType


class BaseArbitraryModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WSRequestSchema(BaseArbitraryModel):
    type: RPCRequestType
    data: Any | None = None


class WSMessageSchemaType(TypedDict):
    # Type for typing to avoid doing pydantic typechecks.
    type: RPCMessageType
    data: dict[str, Any] | None


class WSMessageSchema(BaseArbitraryModel):
    type: RPCMessageType
    data: Any | None = None
    model_config = ConfigDict(extra="allow")


# ------------------------------ REQUEST SCHEMAS ----------------------------


class WSSubscribeRequest(WSRequestSchema):
    type: RPCRequestType = RPCRequestType.SUBSCRIBE
    data: list[RPCMessageType]


class WSWhitelistRequest(WSRequestSchema):
    type: RPCRequestType = RPCRequestType.WHITELIST
    data: None = None


class WSAnalyzedDFRequest(WSRequestSchema):
    type: RPCRequestType = RPCRequestType.ANALYZED_DF
    data: dict[str, Any] = {"limit": 1500, "pair": None}


# ------------------------------ MESSAGE SCHEMAS ----------------------------


class WSWhitelistMessage(WSMessageSchema):
    type: RPCMessageType = RPCMessageType.WHITELIST
    data: list[str]


class WSAnalyzedDFMessage(WSMessageSchema):
    class AnalyzedDFData(BaseArbitraryModel):
        key: PairWithTimeframe
        df: DataFrame
        la: datetime

    type: RPCMessageType = RPCMessageType.ANALYZED_DF
    data: AnalyzedDFData


class WSErrorMessage(WSMessageSchema):
    type: RPCMessageType = RPCMessageType.EXCEPTION
    data: str


# --------------------------------------------------------------------------
