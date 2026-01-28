import logging

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException

from freqtrade.enums import TradingMode
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server.api_schemas import (
    Balances,
    BlacklistPayload,
    BlacklistResponse,
    Count,
    DailyWeeklyMonthly,
    DeleteLockRequest,
    DeleteTrade,
    Entry,
    Exit,
    ForceEnterPayload,
    ForceEnterResponse,
    ForceExitPayload,
    ListCustomData,
    Locks,
    LocksPayload,
    MixTag,
    OpenTradeSchema,
    PairCandlesRequest,
    PairHistory,
    PerformanceEntry,
    Profit,
    ProfitAll,
    ResultMsg,
    Stats,
    StatusMsg,
    WhitelistResponse,
)
from freqtrade.rpc.api_server.deps import get_config, get_rpc
from freqtrade.rpc.rpc import RPCException


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/balance", response_model=Balances, tags=["Trading-info"])
def balance(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    """Account Balances"""
    return rpc._rpc_balance(
        config["stake_currency"],
        config.get("fiat_display_currency", ""),
    )


@router.get("/count", response_model=Count, tags=["Trading-info"])
def count(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_count()


@router.get("/entries", response_model=list[Entry], tags=["Trading-info"])
def entries(pair: str | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_enter_tag_performance(pair)


@router.get("/exits", response_model=list[Exit], tags=["Trading-info"])
def exits(pair: str | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_exit_reason_performance(pair)


@router.get("/mix_tags", response_model=list[MixTag], tags=["Trading-info"])
def mix_tags(pair: str | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_mix_tag_performance(pair)


@router.get("/performance", response_model=list[PerformanceEntry], tags=["Trading-info"])
def performance(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_performance()


@router.get("/profit", response_model=Profit, tags=["Trading-info"])
def profit(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_trade_statistics(config["stake_currency"], config.get("fiat_display_currency"))


@router.get("/profit_all", response_model=ProfitAll, tags=["Trading-info"])
def profit_all(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    response = {
        "all": rpc._rpc_trade_statistics(
            config["stake_currency"], config.get("fiat_display_currency")
        ),
    }
    if config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
        response["long"] = rpc._rpc_trade_statistics(
            config["stake_currency"], config.get("fiat_display_currency"), direction="long"
        )
        response["short"] = rpc._rpc_trade_statistics(
            config["stake_currency"], config.get("fiat_display_currency"), direction="short"
        )

    return response


@router.get("/stats", response_model=Stats, tags=["Trading-info"])
def stats(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stats()


@router.get("/daily", response_model=DailyWeeklyMonthly, tags=["Trading-info"])
def daily(
    timescale: int = Query(7, ge=1, description="Number of days to fetch data for"),
    rpc: RPC = Depends(get_rpc),
    config=Depends(get_config),
):
    return rpc._rpc_timeunit_profit(
        timescale, config["stake_currency"], config.get("fiat_display_currency", "")
    )


@router.get("/weekly", response_model=DailyWeeklyMonthly, tags=["Trading-info"])
def weekly(
    timescale: int = Query(4, ge=1, description="Number of weeks to fetch data for"),
    rpc: RPC = Depends(get_rpc),
    config=Depends(get_config),
):
    return rpc._rpc_timeunit_profit(
        timescale, config["stake_currency"], config.get("fiat_display_currency", ""), "weeks"
    )


@router.get("/monthly", response_model=DailyWeeklyMonthly, tags=["Trading-info"])
def monthly(
    timescale: int = Query(3, ge=1, description="Number of months to fetch data for"),
    rpc: RPC = Depends(get_rpc),
    config=Depends(get_config),
):
    return rpc._rpc_timeunit_profit(
        timescale, config["stake_currency"], config.get("fiat_display_currency", ""), "months"
    )


@router.get("/status", response_model=list[OpenTradeSchema], tags=["Trading-info"])
def status(rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status()
    except RPCException:
        return []


# Using the responsemodel here will cause a ~100% increase in response time (from 1s to 2s)
# on big databases. Correct response model: response_model=TradeResponse,
@router.get("/trades", tags=["Trading-info", "Trades"])
def trades(
    limit: int = Query(500, ge=1, description="Maximum number of different trades to return data"),
    offset: int = Query(0, ge=0, description="Number of trades to skip for pagination"),
    order_by_id: bool = Query(
        True, description="Sort trades by id (default: True). If False, sorts by latest timestamp"
    ),
    rpc: RPC = Depends(get_rpc),
):
    return rpc._rpc_trade_history(limit, offset=offset, order_by_id=order_by_id)


@router.get("/trade/{tradeid}", response_model=OpenTradeSchema, tags=["Trades"])
def trade(tradeid: int = 0, rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status([tradeid])[0]
    except (RPCException, KeyError):
        raise HTTPException(status_code=404, detail="Trade not found.")


@router.delete("/trades/{tradeid}", response_model=DeleteTrade, tags=["Trades"])
def trades_delete(tradeid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete(tradeid)


@router.delete("/trades/{tradeid}/open-order", response_model=OpenTradeSchema, tags=["Trades"])
def trade_cancel_open_order(tradeid: int, rpc: RPC = Depends(get_rpc)):
    rpc._rpc_cancel_open_order(tradeid)
    return rpc._rpc_trade_status([tradeid])[0]


@router.post("/trades/{tradeid}/reload", response_model=OpenTradeSchema, tags=["Trades"])
def trade_reload(tradeid: int, rpc: RPC = Depends(get_rpc)):
    rpc._rpc_reload_trade_from_exchange(tradeid)
    return rpc._rpc_trade_status([tradeid])[0]


@router.get("/trades/open/custom-data", response_model=list[ListCustomData], tags=["Trades"])
def list_open_trades_custom_data(
    key: str | None = Query(None, description="Optional key to filter data"),
    limit: int = Query(100, ge=1, description="Maximum number of different trades to return data"),
    offset: int = Query(0, ge=0, description="Number of trades to skip for pagination"),
    rpc: RPC = Depends(get_rpc),
):
    """
    Fetch custom data for all open trades.
    If a key is provided, it will be used to filter data accordingly.
    Pagination is implemented via the `limit` and `offset` parameters.
    """
    try:
        return rpc._rpc_list_custom_data(key=key, limit=limit, offset=offset)
    except RPCException as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/trades/{trade_id}/custom-data", response_model=list[ListCustomData], tags=["Trades"])
def list_custom_data(trade_id: int, key: str | None = Query(None), rpc: RPC = Depends(get_rpc)):
    """
    Fetch custom data for a specific trade.
    If a key is provided, it will be used to filter data accordingly.
    """
    try:
        return rpc._rpc_list_custom_data(trade_id, key=key)
    except RPCException as e:
        raise HTTPException(status_code=404, detail=str(e))


# /forcebuy is deprecated with short addition. use /forceentry instead
@router.post("/forceenter", response_model=ForceEnterResponse, tags=["Trades"])
@router.post(
    "/forcebuy",
    response_model=ForceEnterResponse,
    tags=["Trades"],
    summary="(deprecated) Please use /forceenter instead",
)
def force_entry(payload: ForceEnterPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None

    trade = rpc._rpc_force_entry(
        payload.pair,
        payload.price,
        order_side=payload.side,
        order_type=ordertype,
        stake_amount=payload.stakeamount,
        enter_tag=payload.entry_tag or "force_entry",
        leverage=payload.leverage,
    )

    if trade:
        return ForceEnterResponse.model_validate(trade.to_json())
    else:
        return ForceEnterResponse.model_validate(
            {"status": f"Error entering {payload.side} trade for pair {payload.pair}."}
        )


# /forcesell is deprecated with short addition. use /forceexit instead
@router.post("/forceexit", response_model=ResultMsg, tags=["Trades"])
@router.post(
    "/forcesell",
    response_model=ResultMsg,
    tags=["Trades"],
    summary="(deprecated) Please use /forceexit instead",
)
def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None
    return rpc._rpc_force_exit(
        str(payload.tradeid), ordertype, amount=payload.amount, price=payload.price
    )


@router.get("/blacklist", response_model=BlacklistResponse, tags=["Trading-info", "Pairlist"])
def blacklist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist()


@router.post("/blacklist", response_model=BlacklistResponse, tags=["Pairlist"])
def blacklist_post(payload: BlacklistPayload, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist(payload.blacklist)


@router.delete("/blacklist", response_model=BlacklistResponse, tags=["Pairlist"])
def blacklist_delete(pairs_to_delete: list[str] = Query([]), rpc: RPC = Depends(get_rpc)):
    """Provide a list of pairs to delete from the blacklist"""

    return rpc._rpc_blacklist_delete(pairs_to_delete)


@router.get("/whitelist", response_model=WhitelistResponse, tags=["Trading-info", "Pairlist"])
def whitelist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_whitelist()


@router.get("/locks", response_model=Locks, tags=["Trading-info", "Locks"])
def locks(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_locks()


@router.delete("/locks/{lockid}", response_model=Locks, tags=["Locks"])
def delete_lock(lockid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=lockid)


@router.post("/locks/delete", response_model=Locks, tags=["Locks"])
def delete_lock_pair(payload: DeleteLockRequest, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=payload.lockid, pair=payload.pair)


@router.post("/locks", response_model=Locks, tags=["Locks"])
def add_locks(payload: list[LocksPayload], rpc: RPC = Depends(get_rpc)):
    for lock in payload:
        rpc._rpc_add_lock(lock.pair, lock.until, lock.reason, lock.side)
    return rpc._rpc_locks()


@router.post("/start", response_model=StatusMsg, tags=["Bot-control"])
def start(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_start()


@router.post("/stop", response_model=StatusMsg, tags=["Bot-control"])
def stop(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stop()


@router.post("/pause", response_model=StatusMsg, tags=["Bot-control"])
@router.post("/stopentry", response_model=StatusMsg, tags=["Bot-control"])
@router.post("/stopbuy", response_model=StatusMsg, tags=["Bot-control"])
def pause(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_pause()


@router.post("/reload_config", response_model=StatusMsg, tags=["Bot-control"])
def reload_config(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_reload_config()


@router.get("/pair_candles", response_model=PairHistory, tags=["Candle data"])
def pair_candles(pair: str, timeframe: str, limit: int | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_analysed_dataframe(pair, timeframe, limit, None)


@router.post("/pair_candles", response_model=PairHistory, tags=["Candle data"])
def pair_candles_filtered(payload: PairCandlesRequest, rpc: RPC = Depends(get_rpc)):
    # Advanced pair_candles endpoint with column filtering
    return rpc._rpc_analysed_dataframe(
        payload.pair, payload.timeframe, payload.limit, payload.columns
    )
