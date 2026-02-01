import logging
from copy import deepcopy
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException

from freqtrade import __version__
from freqtrade.enums import RunMode, State
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server.api_pairlists import handleExchangePayload
from freqtrade.rpc.api_server.api_schemas import (
    Health,
    Logs,
    MarketRequest,
    MarketResponse,
    Ping,
    PlotConfig,
    ShowConfig,
    SysInfo,
    Version,
)
from freqtrade.rpc.api_server.deps import get_config, get_exchange, get_rpc, get_rpc_optional
from freqtrade.rpc.rpc import RPCException


logger = logging.getLogger(__name__)

# API version
# Pre-1.1, no version was provided
# Version increments should happen in "small" steps (1.1, 1.12, ...) unless big changes happen.
# 1.11: forcebuy and forcesell accept ordertype
# 1.12: add blacklist delete endpoint
# 1.13: forcebuy supports stake_amount
# versions 2.xx -> futures/short branch
# 2.14: Add entry/exit orders to trade response
# 2.15: Add backtest history endpoints
# 2.16: Additional daily metrics
# 2.17: Forceentry - leverage, partial force_exit
# 2.20: Add websocket endpoints
# 2.21: Add new_candle messagetype
# 2.22: Add FreqAI to backtesting
# 2.23: Allow plot config request in webserver mode
# 2.24: Add cancel_open_order endpoint
# 2.25: Add several profit values to /status endpoint
# 2.26: increase /balance output
# 2.27: Add /trades/<id>/reload endpoint
# 2.28: Switch reload endpoint to Post
# 2.29: Add /exchanges endpoint
# 2.30: new /pairlists endpoint
# 2.31: new /backtest/history/ delete endpoint
# 2.32: new /backtest/history/ patch endpoint
# 2.33: Additional weekly/monthly metrics
# 2.34: new entries/exits/mix_tags endpoints
# 2.35: pair_candles and pair_history endpoints as Post variant
# 2.40: Add hyperopt-loss endpoint
# 2.41: Add download-data endpoint
# 2.42: Add /pair_history endpoint with live data
# 2.43: Add /profit_all endpoint
# 2.44: Add candle_types parameter to download-data endpoint
# 2.45: Add price to forceexit endpoint
API_VERSION = 2.45

# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


@router_public.get("/ping", response_model=Ping, tags=["Info"])
def ping():
    """simple ping"""
    return {"status": "pong"}


@router.get("/version", response_model=Version, tags=["Info"])
def version():
    """Bot Version info"""
    return {"version": __version__}


@router.get("/show_config", response_model=ShowConfig, tags=["Info"])
def show_config(rpc: RPC | None = Depends(get_rpc_optional), config=Depends(get_config)):
    state: State | str = ""
    strategy_version = None
    if rpc:
        state = rpc._freqtrade.state
        strategy_version = rpc._freqtrade.strategy.version()
    resp = RPC._rpc_show_config(config, state, strategy_version)
    resp["api_version"] = API_VERSION
    return resp


@router.get("/logs", response_model=Logs, tags=["Info"])
def logs(limit: int | None = None):
    return RPC._rpc_get_logs(limit)


@router.get("/plot_config", response_model=PlotConfig, tags=["Candle data"])
def plot_config(
    strategy: str | None = None,
    config=Depends(get_config),
    rpc: RPC | None = Depends(get_rpc_optional),
):
    if not strategy:
        if not rpc:
            raise RPCException("Strategy is mandatory in webserver mode.")
        return PlotConfig.model_validate(rpc._rpc_plot_config())
    else:
        config1 = deepcopy(config)
        config1.update({"strategy": strategy})
    try:
        return PlotConfig.model_validate(RPC._rpc_plot_config_with_strategy(config1))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/markets", response_model=MarketResponse, tags=["Candle data"])
def markets(
    query: Annotated[MarketRequest, Query()],
    config=Depends(get_config),
    rpc: RPC | None = Depends(get_rpc_optional),
):
    if not rpc or config["runmode"] == RunMode.WEBSERVER:
        # webserver mode
        config_loc = deepcopy(config)
        handleExchangePayload(query, config_loc)
        exchange = get_exchange(config_loc)
    else:
        exchange = rpc._freqtrade.exchange

    return {
        "markets": exchange.get_markets(
            base_currencies=[query.base] if query.base else None,
            quote_currencies=[query.quote] if query.quote else None,
        ),
        "exchange_id": exchange.id,
    }


@router.get("/sysinfo", response_model=SysInfo, tags=["Info"])
def sysinfo():
    return RPC._rpc_sysinfo()


@router.get("/health", response_model=Health, tags=["Info"])
def health(rpc: RPC = Depends(get_rpc)):
    return rpc.health()
