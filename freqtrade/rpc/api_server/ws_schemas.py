from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from pandas import DataFrame
from pydantic import BaseModel

from freqtrade.constants import PairWithTimeframe
from freqtrade.enums.rpcmessagetype import RPCMessageType, RPCRequestType


class BaseArbitraryModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class WSRequestSchema(BaseArbitraryModel):
    type: RPCRequestType
    data: Optional[Any] = None


class WSMessageSchemaType(TypedDict):
    # Type for typing to avoid doing pydantic typechecks.
    type: RPCMessageType
    data: Optional[Dict[str, Any]]


class WSMessageSchema(BaseArbitraryModel):
    type: RPCMessageType
    data: Optional[Any] = None

    class Config:
        extra = 'allow'


# ------------------------------ REQUEST SCHEMAS ----------------------------


class WSSubscribeRequest(WSRequestSchema):
    type: RPCRequestType = RPCRequestType.SUBSCRIBE
    data: List[RPCMessageType]


class WSWhitelistRequest(WSRequestSchema):
    type: RPCRequestType = RPCRequestType.WHITELIST
    data: None = None


class WSAnalyzedDFRequest(WSRequestSchema):
    type: RPCRequestType = RPCRequestType.ANALYZED_DF
    data: Dict[str, Any] = {"limit": 1500, "pair": None}


# ------------------------------ MESSAGE SCHEMAS ----------------------------

class WSWhitelistMessage(WSMessageSchema):
    type: RPCMessageType = RPCMessageType.WHITELIST
    data: List[str]


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
