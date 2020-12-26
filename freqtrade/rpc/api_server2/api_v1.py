from typing import List, Optional

from fastapi import APIRouter, Depends

from freqtrade import __version__
from freqtrade.rpc import RPC
from freqtrade.rpc.rpc import RPCException

from .api_models import (Balances, BlacklistPayload, BlacklistResponse, Count, Daily, DeleteTrade, ForceBuyPayload, ForceSellPayload, Locks, Logs, PerformanceEntry, Ping, Profit, ResultMsg, Stats,
                         StatusMsg, Version, WhitelistResponse)
from .deps import get_config, get_rpc


# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


@router_public.get('/ping', response_model=Ping)
def ping():
    """simple ping version"""
    return {"status": "pong"}


@router.get('/version', response_model=Version, tags=['info'])
def version():
    return {"version": __version__}


@router.get('/balance', response_model=Balances, tags=['info'])
def balance(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_balance(config['stake_currency'], config.get('fiat_display_currency', ''),)


@router.get('/count', response_model=Count, tags=['info'])
def count(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_count()


@router.get('/performance', response_model=List[PerformanceEntry], tags=['info'])
def performance(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_performance()


@router.get('/profit', response_model=Profit, tags=['info'])
def profit(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_trade_statistics(config['stake_currency'],
                                     config.get('fiat_display_currency')
                                     )


@router.get('/stats', response_model=Stats, tags=['info'])
def stats(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stats()


@router.get('/daily', response_model=Daily, tags=['info'])
def daily(timescale: int = 7, rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_daily_profit(timescale, config['stake_currency'],
                                 config.get('fiat_display_currency', ''))


# TODO: Missing response model
@router.get('/status', tags=['info'])
def status(rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status()
    except RPCException:
        return []


# TODO: Missing response model
@router.get('/trades', tags=['info'])
def trades(limit: Optional[int] = 0, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_trade_history(limit)


@router.delete('/trades/{tradeid}', response_model=DeleteTrade, tags=['info', 'trading'])
def trades_delete(tradeid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete(tradeid)


# TODO: Missing response model
@router.get('/edge', tags=['info'])
def edge(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_edge()


# TODO: Missing response model
@router.get('/show_config', tags=['info'])
def show_config(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return RPC._rpc_show_config(config, rpc._freqtrade.state)


# TODO: Missing response model
@router.post('/forcebuy', tags=['trading'])
def forcebuy(payload: ForceBuyPayload, rpc: RPC = Depends(get_rpc)):
    trade = rpc._rpc_forcebuy(payload.pair, payload.price)

    if trade:
        return trade.to_json()
    else:
        return {"status": f"Error buying pair {payload.pair}."}


@router.post('/forcesell', response_model=ResultMsg, tags=['trading'])
def forcesell(payload: ForceSellPayload, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_forcesell(payload.tradeid)


@router.get('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist()


@router.post('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist_post(payload: BlacklistPayload, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist(payload.blacklist)


@router.get('/whitelist', response_model=WhitelistResponse, tags=['info', 'pairlist'])
def whitelist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_whitelist()


@router.get('/locks', response_model=Locks, tags=['info'])
def locks(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_locks()


@router.get('/logs', response_model=Logs, tags=['info'])
def logs(limit: Optional[int] = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_get_logs(limit)


@router.post('/start', response_model=StatusMsg, tags=['botcontrol'])
def start(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_start()


@router.post('/stop', response_model=StatusMsg, tags=['botcontrol'])
def stop(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stop()


@router.post('/stopbuy', response_model=StatusMsg, tags=['botcontrol'])
def stop_buy(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stopbuy()


@router.post('/reload_config', response_model=StatusMsg, tags=['botcontrol'])
def reload_config(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_reload_config()
