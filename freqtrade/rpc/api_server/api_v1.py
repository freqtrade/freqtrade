import logging
from copy import deepcopy
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.exceptions import HTTPException

from freqtrade import __version__
from freqtrade.constants import Config
from freqtrade.data.history import get_datahandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server.api_schemas import (AvailablePairs, BackgroundTaskStatus, Balances,
                                                  BgJobStarted, BlacklistPayload, BlacklistResponse,
                                                  Count, Daily, DeleteLockRequest, DeleteTrade,
                                                  ForceEnterPayload, ForceEnterResponse,
                                                  ForceExitPayload, FreqAIModelListResponse, Health,
                                                  Locks, Logs, OpenTradeSchema, PairHistory,
                                                  PairListsPayload, PairListsResponse,
                                                  PerformanceEntry, Ping, PlotConfig, Profit,
                                                  ResultMsg, ShowConfig, Stats, StatusMsg,
                                                  StrategyListResponse, StrategyResponse, SysInfo,
                                                  Version, WhitelistEvaluateResponse,
                                                  WhitelistResponse)
from freqtrade.rpc.api_server.deps import get_config, get_exchange, get_rpc, get_rpc_optional
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
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
# 2.29: new /pairlists endpoint
API_VERSION = 2.29

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
    return rpc._rpc_timeunit_profit(timescale, config['stake_currency'],
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


@router.delete('/trades/{tradeid}/open-order', response_model=OpenTradeSchema,  tags=['trading'])
def trade_cancel_open_order(tradeid: int, rpc: RPC = Depends(get_rpc)):
    rpc._rpc_cancel_open_order(tradeid)
    return rpc._rpc_trade_status([tradeid])[0]


@router.post('/trades/{tradeid}/reload', response_model=OpenTradeSchema,  tags=['trading'])
def trade_reload(tradeid: int, rpc: RPC = Depends(get_rpc)):
    rpc._rpc_reload_trade_from_exchange(tradeid)
    return rpc._rpc_trade_status([tradeid])[0]


# TODO: Missing response model
@router.get('/edge', tags=['info'])
def edge(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_edge()


@router.get('/show_config', response_model=ShowConfig, tags=['info'])
def show_config(rpc: Optional[RPC] = Depends(get_rpc_optional), config=Depends(get_config)):
    state = ''
    strategy_version = None
    if rpc:
        state = rpc._freqtrade.state
        strategy_version = rpc._freqtrade.strategy.version()
    resp = RPC._rpc_show_config(config, state, strategy_version)
    resp['api_version'] = API_VERSION
    return resp


# /forcebuy is deprecated with short addition. use /forceentry instead
@router.post('/forceenter', response_model=ForceEnterResponse, tags=['trading'])
@router.post('/forcebuy', response_model=ForceEnterResponse, tags=['trading'])
def force_entry(payload: ForceEnterPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None

    trade = rpc._rpc_force_entry(payload.pair, payload.price, order_side=payload.side,
                                 order_type=ordertype, stake_amount=payload.stakeamount,
                                 enter_tag=payload.entry_tag or 'force_entry',
                                 leverage=payload.leverage)

    if trade:
        return ForceEnterResponse.parse_obj(trade.to_json())
    else:
        return ForceEnterResponse.parse_obj(
            {"status": f"Error entering {payload.side} trade for pair {payload.pair}."})


# /forcesell is deprecated with short addition. use /forceexit instead
@router.post('/forceexit', response_model=ResultMsg, tags=['trading'])
@router.post('/forcesell', response_model=ResultMsg, tags=['trading'])
def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None
    return rpc._rpc_force_exit(payload.tradeid, ordertype, amount=payload.amount)


@router.get('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist()


@router.post('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist_post(payload: BlacklistPayload, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist(payload.blacklist)


@router.delete('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist_delete(pairs_to_delete: List[str] = Query([]), rpc: RPC = Depends(get_rpc)):
    """Provide a list of pairs to delete from the blacklist"""

    return rpc._rpc_blacklist_delete(pairs_to_delete)


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


@router.post('/stopentry', response_model=StatusMsg, tags=['botcontrol'])
@router.post('/stopbuy', response_model=StatusMsg, tags=['botcontrol'])
def stop_buy(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stopentry()


@router.post('/reload_config', response_model=StatusMsg, tags=['botcontrol'])
def reload_config(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_reload_config()


@router.get('/pair_candles', response_model=PairHistory, tags=['candle data'])
def pair_candles(
        pair: str, timeframe: str, limit: Optional[int] = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_analysed_dataframe(pair, timeframe, limit)


@router.get('/pair_history', response_model=PairHistory, tags=['candle data'])
def pair_history(pair: str, timeframe: str, timerange: str, strategy: str,
                 freqaimodel: Optional[str] = None,
                 config=Depends(get_config), exchange=Depends(get_exchange)):
    # The initial call to this endpoint can be slow, as it may need to initialize
    # the exchange class.
    config = deepcopy(config)
    config.update({
        'strategy': strategy,
        'timerange': timerange,
        'freqaimodel': freqaimodel if freqaimodel else config.get('freqaimodel'),
    })
    return RPC._rpc_analysed_history_full(config, pair, timeframe, exchange)


@router.get('/plot_config', response_model=PlotConfig, tags=['candle data'])
def plot_config(strategy: Optional[str] = None, config=Depends(get_config),
                rpc: Optional[RPC] = Depends(get_rpc_optional)):
    if not strategy:
        if not rpc:
            raise RPCException("Strategy is mandatory in webserver mode.")
        return PlotConfig.parse_obj(rpc._rpc_plot_config())
    else:
        config1 = deepcopy(config)
        config1.update({
            'strategy': strategy
        })
        return PlotConfig.parse_obj(RPC._rpc_plot_config_with_strategy(config1))


@router.get('/strategies', response_model=StrategyListResponse, tags=['strategy'])
def list_strategies(config=Depends(get_config)):
    from freqtrade.resolvers.strategy_resolver import StrategyResolver
    strategies = StrategyResolver.search_all_objects(
        config, False, config.get('recursive_strategy_search', False))
    strategies = sorted(strategies, key=lambda x: x['name'])

    return {'strategies': [x['name'] for x in strategies]}


@router.get('/strategy/{strategy}', response_model=StrategyResponse, tags=['strategy'])
def get_strategy(strategy: str, config=Depends(get_config)):
    if ":" in strategy:
        raise HTTPException(status_code=500, detail="base64 encoded strategies are not allowed.")

    config_ = deepcopy(config)
    from freqtrade.resolvers.strategy_resolver import StrategyResolver
    try:
        strategy_obj = StrategyResolver._load_strategy(strategy, config_,
                                                       extra_dir=config_.get('strategy_path'))
    except OperationalException:
        raise HTTPException(status_code=404, detail='Strategy not found')

    return {
        'strategy': strategy_obj.get_strategy_name(),
        'code': strategy_obj.__source__,
    }


@router.get('/pairlists/available',
            response_model=PairListsResponse, tags=['pairlists', 'webserver'])
def list_pairlists(config=Depends(get_config)):
    from freqtrade.resolvers import PairListResolver
    pairlists = PairListResolver.search_all_objects(
        config, False)
    pairlists = sorted(pairlists, key=lambda x: x['name'])

    return {'pairlists': [{
        "name": x['name'],
        "is_pairlist_generator": x['class'].is_pairlist_generator,
        "params": x['class'].available_parameters(),
        "description": x['class'].description(),
         } for x in pairlists
    ]}


def __run_pairlist(job_id: str, config_loc: Config):
    try:

        ApiBG.jobs[job_id]['is_running'] = True
        from freqtrade.plugins.pairlistmanager import PairListManager

        exchange = get_exchange(config_loc)
        pairlists = PairListManager(exchange, config_loc)
        pairlists.refresh_pairlist()
        ApiBG.jobs[job_id]['result'] = {
                'method': pairlists.name_list,
                'length': len(pairlists.whitelist),
                'whitelist': pairlists.whitelist
            }
        ApiBG.jobs[job_id]['status'] = 'success'
    except (OperationalException, Exception) as e:
        logger.exception(e)
        ApiBG.jobs[job_id]['error'] = str(e)
    finally:
        ApiBG.jobs[job_id]['is_running'] = False
        ApiBG.pairlist_running = False


@router.post('/pairlists/evaluate', response_model=BgJobStarted, tags=['pairlists'])
def pairlists_evaluate(payload: PairListsPayload, background_tasks: BackgroundTasks,
                       config=Depends(get_config)):
    if ApiBG.pairlist_running:
        raise HTTPException(status_code=400, detail='Pairlist evaluation is already running.')

    config_loc = deepcopy(config)
    config_loc['stake_currency'] = payload.stake_currency
    config_loc['pairlists'] = payload.pairlists
    # TODO: overwrite blacklist? make it optional and fall back to the one in config?
    # Outcome depends on the UI approach.
    config_loc['exchange']['pair_blacklist'] = payload.blacklist
    # Random job id
    job_id = ApiBG.get_job_id()

    ApiBG.jobs[job_id] = {
        'category': 'pairlist',
        'status': 'pending',
        'progress': None,
        'is_running': False,
        'result': {},
        'error': None,
    }
    ApiBG.running_jobs.append(job_id)
    background_tasks.add_task(__run_pairlist, job_id, config_loc)
    ApiBG.pairlist_running = True

    return {
        'status': 'Pairlist evaluation started in background.',
        'job_id': job_id,
    }


@router.get('/pairlists/evaluate/{jobid}', response_model=WhitelistEvaluateResponse,
            tags=['pairlists'])
def pairlists_evaluate_get(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail='Job not found.')

    if job['is_running']:
        raise HTTPException(status_code=400, detail='Job not finished yet.')

    if error := job['error']:
        return {
            'status': 'failed',
            'error': error,
        }

    return {
        'status': 'success',
        'result': job['result'],
    }


@router.get('/background/{jobid}', response_model=BackgroundTaskStatus, tags=['webserver'])
def background_job(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail='Job not found.')

    return {
        'job_id': jobid,
        # 'type': job['job_type'],
        'status': job['status'],
        'running': job['is_running'],
        'progress': job.get('progress'),
        # 'job_error': job['error'],
    }


@router.get('/freqaimodels', response_model=FreqAIModelListResponse, tags=['freqai'])
def list_freqaimodels(config=Depends(get_config)):
    from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver
    models = FreqaiModelResolver.search_all_objects(
        config, False)
    models = sorted(models, key=lambda x: x['name'])

    return {'freqaimodels': [x['name'] for x in models]}


@router.get('/available_pairs', response_model=AvailablePairs, tags=['candle data'])
def list_available_pairs(timeframe: Optional[str] = None, stake_currency: Optional[str] = None,
                         candletype: Optional[CandleType] = None, config=Depends(get_config)):

    dh = get_datahandler(config['datadir'], config.get('dataformat_ohlcv'))
    trading_mode: TradingMode = config.get('trading_mode', TradingMode.SPOT)
    pair_interval = dh.ohlcv_get_available_data(config['datadir'], trading_mode)

    if timeframe:
        pair_interval = [pair for pair in pair_interval if pair[1] == timeframe]
    if stake_currency:
        pair_interval = [pair for pair in pair_interval if pair[0].endswith(stake_currency)]
    if candletype:
        pair_interval = [pair for pair in pair_interval if pair[2] == candletype]
    else:
        candle_type = CandleType.get_default(trading_mode)
        pair_interval = [pair for pair in pair_interval if pair[2] == candle_type]

    pair_interval = sorted(pair_interval, key=lambda x: x[0])

    pairs = list({x[0] for x in pair_interval})
    pairs.sort()
    result = {
        'length': len(pairs),
        'pairs': pairs,
        'pair_interval': pair_interval,
    }
    return result


@router.get('/sysinfo', response_model=SysInfo, tags=['info'])
def sysinfo():
    return RPC._rpc_sysinfo()


@router.get('/health', response_model=Health, tags=['info'])
def health(rpc: RPC = Depends(get_rpc)):
    return rpc.health()
