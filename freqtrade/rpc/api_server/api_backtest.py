import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade.configuration.config_validation import validate_config_consistency
from freqtrade.data.btanalysis import get_backtest_resultlist, load_and_merge_backtest_result
from freqtrade.enums import BacktestState
from freqtrade.exceptions import DependencyException
from freqtrade.misc import deep_merge_dicts
from freqtrade.rpc.api_server.api_schemas import (BacktestHistoryEntry, BacktestRequest,
                                                  BacktestResponse)
from freqtrade.rpc.api_server.deps import get_config, is_webserver_mode
from freqtrade.rpc.api_server.webserver import ApiServer
from freqtrade.rpc.rpc import RPCException


logger = logging.getLogger(__name__)

# Private API, protected by authentication
router = APIRouter()


@router.post('/backtest', response_model=BacktestResponse, tags=['webserver', 'backtest'])
# flake8: noqa: C901
async def api_start_backtest(bt_settings: BacktestRequest, background_tasks: BackgroundTasks,
                             config=Depends(get_config), ws_mode=Depends(is_webserver_mode)):
    """Start backtesting if not done so already"""
    if ApiServer._bgtask_running:
        raise RPCException('Bot Background task already running')

    if ':' in bt_settings.strategy:
        raise HTTPException(status_code=500, detail="base64 encoded strategies are not allowed.")

    btconfig = deepcopy(config)
    settings = dict(bt_settings)
    if settings.get('freqai', None) is not None:
        settings['freqai'] = dict(settings['freqai'])
    # Pydantic models will contain all keys, but non-provided ones are None

    btconfig = deep_merge_dicts(settings, btconfig, allow_null_overrides=False)
    try:
        btconfig['stake_amount'] = float(btconfig['stake_amount'])
    except ValueError:
        pass

    # Force dry-run for backtesting
    btconfig['dry_run'] = True

    # Start backtesting
    # Initialize backtesting object
    def run_backtest():
        from freqtrade.optimize.optimize_reports import (generate_backtest_stats,
                                                         store_backtest_stats)
        from freqtrade.resolvers import StrategyResolver
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            # Reload strategy
            lastconfig = ApiServer._bt_last_config
            strat = StrategyResolver.load_strategy(btconfig)
            validate_config_consistency(btconfig)

            if (
                not ApiServer._bt
                or lastconfig.get('timeframe') != strat.timeframe
                or lastconfig.get('timeframe_detail') != btconfig.get('timeframe_detail')
                or lastconfig.get('timerange') != btconfig['timerange']
            ):
                from freqtrade.optimize.backtesting import Backtesting
                ApiServer._bt = Backtesting(btconfig)
                ApiServer._bt.load_bt_data_detail()
            else:
                ApiServer._bt.config = btconfig
                ApiServer._bt.init_backtest()
            # Only reload data if timeframe changed.
            if (
                not ApiServer._bt_data
                or not ApiServer._bt_timerange
                or lastconfig.get('timeframe') != strat.timeframe
                or lastconfig.get('timerange') != btconfig['timerange']
            ):
                ApiServer._bt_data, ApiServer._bt_timerange = ApiServer._bt.load_bt_data()

            lastconfig['timerange'] = btconfig['timerange']
            lastconfig['timeframe'] = strat.timeframe
            lastconfig['protections'] = btconfig.get('protections', [])
            lastconfig['enable_protections'] = btconfig.get('enable_protections')
            lastconfig['dry_run_wallet'] = btconfig.get('dry_run_wallet')

            ApiServer._bt.enable_protections = btconfig.get('enable_protections', False)
            ApiServer._bt.strategylist = [strat]
            ApiServer._bt.results = {}
            ApiServer._bt.load_prior_backtest()

            ApiServer._bt.abort = False
            if (ApiServer._bt.results and
                    strat.get_strategy_name() in ApiServer._bt.results['strategy']):
                # When previous result hash matches - reuse that result and skip backtesting.
                logger.info(f'Reusing result of previous backtest for {strat.get_strategy_name()}')
            else:
                min_date, max_date = ApiServer._bt.backtest_one_strategy(
                    strat, ApiServer._bt_data, ApiServer._bt_timerange)

                ApiServer._bt.results = generate_backtest_stats(
                    ApiServer._bt_data, ApiServer._bt.all_results,
                    min_date=min_date, max_date=max_date)

            if btconfig.get('export', 'none') == 'trades':
                store_backtest_stats(
                    btconfig['exportfilename'], ApiServer._bt.results,
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    )

            logger.info("Backtest finished.")

        except DependencyException as e:
            logger.info(f"Backtesting caused an error: {e}")
            pass
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
def api_get_backtest(ws_mode=Depends(is_webserver_mode)):
    """
    Get backtesting result.
    Returns Result after backtesting has been ran.
    """
    from freqtrade.persistence import LocalTrade
    if ApiServer._bgtask_running:
        return {
            "status": "running",
            "running": True,
            "step": ApiServer._bt.progress.action if ApiServer._bt else str(BacktestState.STARTUP),
            "progress": ApiServer._bt.progress.progress if ApiServer._bt else 0,
            "trade_count": len(LocalTrade.trades),
            "status_msg": "Backtest running",
        }

    if not ApiServer._bt:
        return {
            "status": "not_started",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": "Backtest not yet executed"
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
def api_delete_backtest(ws_mode=Depends(is_webserver_mode)):
    """Reset backtesting"""
    if ApiServer._bgtask_running:
        return {
            "status": "running",
            "running": True,
            "step": "",
            "progress": 0,
            "status_msg": "Backtest running",
        }
    if ApiServer._bt:
        ApiServer._bt.cleanup()
        del ApiServer._bt
        ApiServer._bt = None
        del ApiServer._bt_data
        ApiServer._bt_data = None
        logger.info("Backtesting reset")
    return {
        "status": "reset",
        "running": False,
        "step": "",
        "progress": 0,
        "status_msg": "Backtest reset",
    }


@router.get('/backtest/abort', response_model=BacktestResponse, tags=['webserver', 'backtest'])
def api_backtest_abort(ws_mode=Depends(is_webserver_mode)):
    if not ApiServer._bgtask_running:
        return {
            "status": "not_running",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": "Backtest ended",
        }
    ApiServer._bt.abort = True
    return {
        "status": "stopping",
        "running": False,
        "step": "",
        "progress": 0,
        "status_msg": "Backtest ended",
    }


@router.get('/backtest/history', response_model=List[BacktestHistoryEntry], tags=['webserver', 'backtest'])
def api_backtest_history(config=Depends(get_config), ws_mode=Depends(is_webserver_mode)):
    # Get backtest result history, read from metadata files
    return get_backtest_resultlist(config['user_data_dir'] / 'backtest_results')


@router.get('/backtest/history/result', response_model=BacktestResponse, tags=['webserver', 'backtest'])
def api_backtest_history_result(filename: str, strategy: str, config=Depends(get_config), ws_mode=Depends(is_webserver_mode)):
    # Get backtest result history, read from metadata files
    fn = config['user_data_dir'] / 'backtest_results' / filename
    results: Dict[str, Any] = {
        'metadata': {},
        'strategy': {},
        'strategy_comparison': [],
    }

    load_and_merge_backtest_result(strategy, fn, results)
    return {
        "status": "ended",
        "running": False,
        "step": "",
        "progress": 1,
        "status_msg": "Historic result",
        "backtest_result": results,
    }
