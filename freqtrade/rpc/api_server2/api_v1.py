from fastapi import APIRouter, Depends

from freqtrade import __version__

from .deps import get_config, get_rpc
from .models import Balances, Ping, Version


# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


@router_public.get('/ping', response_model=Ping)
def ping():
    """simple ping version"""
    return {"status": "pong"}


@router.get('/version', response_model=Version)
def version():
    return {"version": __version__}


@router.get('/balance', response_model=Balances)
def balance(rpc=Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_balance(config['stake_currency'], config.get('fiat_display_currency', ''),)


