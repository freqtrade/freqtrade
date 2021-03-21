import asyncio
import logging
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade import __version__
from freqtrade.constants import USERPATH_STRATEGIES
from freqtrade.data.history import get_datahandler
from freqtrade.exceptions import OperationalException
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server.api_schemas import (AvailablePairs, BacktestRequest, BacktestResponse,
                                                  Balances, BlacklistPayload, BlacklistResponse,
                                                  Count, Daily, DeleteLockRequest, DeleteTrade,
                                                  ForceBuyPayload, ForceBuyResponse,
                                                  ForceSellPayload, Locks, Logs, OpenTradeSchema,
                                                  PairHistory, PerformanceEntry, Ping, PlotConfig,
                                                  Profit, ResultMsg, ShowConfig, Stats, StatusMsg,
                                                  StrategyListResponse, StrategyResponse, Version,
                                                  WhitelistResponse)
from freqtrade.rpc.api_server.deps import get_config, get_rpc, get_rpc_optional
from freqtrade.rpc.rpc import RPCException
from freqtrade.state import BacktestState


logger = logging.getLogger(__name__)

# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


@router_public.get('/ping', response_model=Ping)
def ping():
    """simple ping"""
    return {"status": "pong"}


@router.get('/version', response_model=Version, tags=['info'])
def version():
    """ Bot Version info"""
    return {"version": __version__}


@router.get('/balance', response_model=Balances, tags=['info'])
def balance(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    """Account Balances"""
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


@router.get('/status', response_model=List[OpenTradeSchema], tags=['info'])
def status(rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status()
    except RPCException:
        return []


# Using the responsemodel here will cause a ~100% increase in response time (from 1s to 2s)
# on big databases. Correct response model: response_model=TradeResponse,
@router.get('/trades', tags=['info', 'trading'])
def trades(limit: int = 500, offset: int = 0, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_trade_history(limit, offset=offset, order_by_id=True)


@router.get('/trade/{tradeid}', response_model=OpenTradeSchema, tags=['info', 'trading'])
def trade(tradeid: int = 0, rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status([tradeid])[0]
    except (RPCException, KeyError):
        raise HTTPException(status_code=404, detail='Trade not found.')


@router.delete('/trades/{tradeid}', response_model=DeleteTrade, tags=['info', 'trading'])
def trades_delete(tradeid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete(tradeid)


# TODO: Missing response model
@router.get('/edge', tags=['info'])
def edge(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_edge()


@router.get('/show_config', response_model=ShowConfig, tags=['info'])
def show_config(rpc: Optional[RPC] = Depends(get_rpc_optional), config=Depends(get_config)):
    state = ''
    if rpc:
        state = rpc._freqtrade.state
    return RPC._rpc_show_config(config, state)


@router.post('/forcebuy', response_model=ForceBuyResponse, tags=['trading'])
def forcebuy(payload: ForceBuyPayload, rpc: RPC = Depends(get_rpc)):
    trade = rpc._rpc_forcebuy(payload.pair, payload.price)

    if trade:
        return ForceBuyResponse.parse_obj(trade.to_json())
    else:
        return ForceBuyResponse.parse_obj({"status": f"Error buying pair {payload.pair}."})


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


@router.get('/locks', response_model=Locks, tags=['info', 'locks'])
def locks(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_locks()


@router.delete('/locks/{lockid}', response_model=Locks, tags=['info', 'locks'])
def delete_lock(lockid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=lockid)


@router.post('/locks/delete', response_model=Locks, tags=['info', 'locks'])
def delete_lock_pair(payload: DeleteLockRequest, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=payload.lockid, pair=payload.pair)


@router.get('/logs', response_model=Logs, tags=['info'])
def logs(limit: Optional[int] = None):
    return RPC._rpc_get_logs(limit)


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


@router.get('/pair_candles', response_model=PairHistory, tags=['candle data'])
def pair_candles(pair: str, timeframe: str, limit: Optional[int], rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_analysed_dataframe(pair, timeframe, limit)


@router.get('/pair_history', response_model=PairHistory, tags=['candle data'])
def pair_history(pair: str, timeframe: str, timerange: str, strategy: str,
                 config=Depends(get_config)):
    config = deepcopy(config)
    config.update({
            'strategy': strategy,
        })
    return RPC._rpc_analysed_history_full(config, pair, timeframe, timerange)


@router.get('/plot_config', response_model=PlotConfig, tags=['candle data'])
def plot_config(rpc: RPC = Depends(get_rpc)):
    return PlotConfig.parse_obj(rpc._rpc_plot_config())


@router.get('/strategies', response_model=StrategyListResponse, tags=['strategy'])
def list_strategies(config=Depends(get_config)):
    directory = Path(config.get(
        'strategy_path', config['user_data_dir'] / USERPATH_STRATEGIES))
    from freqtrade.resolvers.strategy_resolver import StrategyResolver
    strategies = StrategyResolver.search_all_objects(directory, False)
    strategies = sorted(strategies, key=lambda x: x['name'])

    return {'strategies': [x['name'] for x in strategies]}


@router.get('/strategy/{strategy}', response_model=StrategyResponse, tags=['strategy'])
def get_strategy(strategy: str, config=Depends(get_config)):

    config = deepcopy(config)
    from freqtrade.resolvers.strategy_resolver import StrategyResolver
    try:
        strategy_obj = StrategyResolver._load_strategy(strategy, config,
                                                       extra_dir=config.get('strategy_path'))
    except OperationalException:
        raise HTTPException(status_code=404, detail='Strategy not found')

    return {
        'strategy': strategy_obj.get_strategy_name(),
        'code': strategy_obj.__source__,
    }


@router.get('/available_pairs', response_model=AvailablePairs, tags=['candle data'])
def list_available_pairs(timeframe: Optional[str] = None, stake_currency: Optional[str] = None,
                         config=Depends(get_config)):

    dh = get_datahandler(config['datadir'], config.get('dataformat_ohlcv', None))

    pair_interval = dh.ohlcv_get_available_data(config['datadir'])

    if timeframe:
        pair_interval = [pair for pair in pair_interval if pair[1] == timeframe]
    if stake_currency:
        pair_interval = [pair for pair in pair_interval if pair[0].endswith(stake_currency)]
    pair_interval = sorted(pair_interval, key=lambda x: x[0])

    pairs = list({x[0] for x in pair_interval})

    result = {
        'length': len(pairs),
        'pairs': pairs,
        'pair_interval': pair_interval,
    }
    return result


@router.post('/backtest', response_model=BacktestResponse, tags=['webserver', 'backtest'])
async def api_start_backtest(bt_settings: BacktestRequest, background_tasks: BackgroundTasks,
                             config=Depends(get_config)):
    """Start backtesting if not done so already"""
    if ApiServer._bgtask_running:
        raise RPCException('Bot Background task already running')

    btconfig = deepcopy(config)
    settings = dict(bt_settings)
    # Pydantic models will contain all keys, but non-provided ones are None
    for setting in settings.keys():
        if settings[setting] is not None:
            btconfig[setting] = settings[setting]

    # Start backtesting
    # Initialize backtesting object
    def run_backtest():
        from freqtrade.optimize.optimize_reports import generate_backtest_stats
        from freqtrade.resolvers import StrategyResolver
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            # Reload strategy
            lastconfig = ApiServer._lastbacktestconfig
            strat = StrategyResolver.load_strategy(btconfig)

            if (not ApiServer._bt
                    or lastconfig.get('timeframe') != strat.timeframe
                    or lastconfig.get('enable_protections') != btconfig.get('enable_protections')
                    or lastconfig.get('protections') != btconfig.get('protections', [])
                    or lastconfig.get('dry_run_wallet') != btconfig.get('dry_run_wallet', 0)):
                # TODO: Investigate if enabling protections can be dynamically ingested from here...
                from freqtrade.optimize.backtesting import Backtesting
                ApiServer._bt = Backtesting(btconfig)
                # Reset data if backtesting is reloaded
                # TODO: is this always necessary??
                ApiServer._backtestdata = None

            if (not ApiServer._backtestdata or not ApiServer._bt_timerange
                    or lastconfig.get('timerange') != btconfig['timerange']):
                lastconfig['timerange'] = btconfig['timerange']
                lastconfig['protections'] = btconfig.get('protections', [])
                lastconfig['enable_protections'] = btconfig.get('enable_protections')
                lastconfig['dry_run_wallet'] = btconfig.get('dry_run_wallet')
                lastconfig['timeframe'] = strat.timeframe
                ApiServer._backtestdata, ApiServer._bt_timerange = ApiServer._bt.load_bt_data()

            min_date, max_date = ApiServer._bt.backtest_one_strategy(
                strat, ApiServer._backtestdata,
                ApiServer._bt_timerange)
            ApiServer._bt.results = generate_backtest_stats(
                ApiServer._backtestdata, ApiServer._bt.all_results,
                min_date=min_date, max_date=max_date)
            logger.info("Backtesting finished.")

        finally:
            ApiServer._bgtask_running = False

    background_tasks.add_task(run_backtest)
    ApiServer._bgtask_running = True

    return {
        "status": "running",
        "running": True,
        "progress": 0,
        "step": str(BacktestState.STARTUP),
        "status_msg": "Backtest started",
    }


@router.get('/backtest', response_model=BacktestResponse, tags=['webserver', 'backtest'])
def api_get_backtest():
    """
    Get backtesting result.
    Returns Result after backtesting has been ran.
    """
    from freqtrade.persistence import LocalTrade
    if ApiServer._bgtask_running:
        return {
            "status": "running",
            "running": True,
            "step": ApiServer._bt.get_action() if ApiServer._bt else str(BacktestState.STARTUP),
            "progress": ApiServer._bt.get_progress() if ApiServer._bt else 0,
            "trade_count": len(LocalTrade.trades),
            "status_msg": "Backtest running",
        }

    if not ApiServer._bt:
        return {
            "status": "not_started",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": "Backtesting not yet executed"
        }

    return {
        "status": "ended",
        "running": False,
        "status_msg": "Backtest ended",
        "step": "finished",
        "progress": 1,
        "backtest_result": ApiServer._bt.results,
    }


@router.delete('/backtest', response_model=BacktestResponse, tags=['webserver', 'backtest'])
def api_delete_backtest():
    """Reset backtesting"""
    if ApiServer._bgtask_running:
        return {
            "status": "running",
            "running": True,
            "progress": 0,
            "status_msg": "Backtest running",
        }
    if ApiServer._bt:
        del ApiServer._bt
        ApiServer._bt = None
        del ApiServer._backtestdata
        ApiServer._backtestdata = None
        logger.info("Backtesting reset")
    return {
        "status": "reset",
        "running": False,
        "progress": 0,
        "status_msg": "Backtesting reset",
    }
