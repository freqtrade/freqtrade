import logging
from copy import deepcopy

from fastapi import APIRouter, Depends, HTTPException

from freqtrade.configuration import validate_config_consistency
from freqtrade.rpc.api_server.api_pairlists import handleExchangePayload
from freqtrade.rpc.api_server.api_schemas import PairHistory, PairHistoryRequest
from freqtrade.rpc.api_server.deps import get_config, get_exchange
from freqtrade.rpc.rpc import RPC


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/pair_history", response_model=PairHistory, tags=["Candle data"])
def pair_history(
    pair: str,
    timeframe: str,
    timerange: str,
    strategy: str,
    freqaimodel: str | None = None,
    config=Depends(get_config),
    exchange=Depends(get_exchange),
):
    # The initial call to this endpoint can be slow, as it may need to initialize
    # the exchange class.
    config_loc = deepcopy(config)
    config_loc.update(
        {
            "timeframe": timeframe,
            "strategy": strategy,
            "timerange": timerange,
            "freqaimodel": freqaimodel if freqaimodel else config_loc.get("freqaimodel"),
        }
    )
    validate_config_consistency(config_loc)
    try:
        return RPC._rpc_analysed_history_full(config_loc, pair, timeframe, exchange, None, False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/pair_history", response_model=PairHistory, tags=["Candle data"])
def pair_history_filtered(payload: PairHistoryRequest, config=Depends(get_config)):
    # The initial call to this endpoint can be slow, as it may need to initialize
    # the exchange class.
    config_loc = deepcopy(config)
    config_loc.update(
        {
            "timeframe": payload.timeframe,
            "strategy": payload.strategy,
            "timerange": payload.timerange,
            "freqaimodel": (
                payload.freqaimodel if payload.freqaimodel else config_loc.get("freqaimodel")
            ),
        }
    )
    handleExchangePayload(payload, config_loc)
    exchange = get_exchange(config_loc)

    validate_config_consistency(config_loc)

    try:
        return RPC._rpc_analysed_history_full(
            config_loc,
            payload.pair,
            payload.timeframe,
            exchange,
            payload.columns,
            payload.live_mode,
        )
    except Exception as e:
        logger.exception("Error in pair_history_filtered")
        raise HTTPException(status_code=502, detail=str(e))
