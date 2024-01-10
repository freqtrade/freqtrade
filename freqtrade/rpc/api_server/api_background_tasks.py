import logging
from copy import deepcopy

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade.constants import Config
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import FtNoDBContext
from freqtrade.rpc.api_server.api_schemas import (BackgroundTaskStatus, BgJobStarted,
                                                  ExchangeModePayloadMixin, PairListsPayload,
                                                  PairListsResponse, WhitelistEvaluateResponse)
from freqtrade.rpc.api_server.deps import get_config, get_exchange
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG


logger = logging.getLogger(__name__)

# Private API, protected by authentication and webserver_mode dependency
router = APIRouter()


@router.get('/background/{jobid}', response_model=BackgroundTaskStatus, tags=['webserver'])
def background_job(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail='Job not found.')

    return {
        'job_id': jobid,
        'job_category': job['category'],
        'status': job['status'],
        'running': job['is_running'],
        'progress': job.get('progress'),
        # 'job_error': job['error'],
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
        with FtNoDBContext():
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
        ApiBG.jobs[job_id]['status'] = 'failed'
    finally:
        ApiBG.jobs[job_id]['is_running'] = False
        ApiBG.pairlist_running = False


@router.post('/pairlists/evaluate', response_model=BgJobStarted, tags=['pairlists', 'webserver'])
def pairlists_evaluate(payload: PairListsPayload, background_tasks: BackgroundTasks,
                       config=Depends(get_config)):
    if ApiBG.pairlist_running:
        raise HTTPException(status_code=400, detail='Pairlist evaluation is already running.')

    config_loc = deepcopy(config)
    config_loc['stake_currency'] = payload.stake_currency
    config_loc['pairlists'] = payload.pairlists
    handleExchangePayload(payload, config_loc)
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
    background_tasks.add_task(__run_pairlist, job_id, config_loc)
    ApiBG.pairlist_running = True

    return {
        'status': 'Pairlist evaluation started in background.',
        'job_id': job_id,
    }


def handleExchangePayload(payload: ExchangeModePayloadMixin, config_loc: Config):
    """
    Handle exchange and trading mode payload.
    Updates the configuration with the payload values.
    """
    if payload.exchange:
        config_loc['exchange']['name'] = payload.exchange
    if payload.trading_mode:
        config_loc['trading_mode'] = payload.trading_mode
        config_loc['candle_type_def'] = CandleType.get_default(
            config_loc.get('trading_mode', 'spot') or 'spot')
    if payload.margin_mode:
        config_loc['margin_mode'] = payload.margin_mode


@router.get('/pairlists/evaluate/{jobid}', response_model=WhitelistEvaluateResponse,
            tags=['pairlists', 'webserver'])
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
