from typing import Dict

from fastapi import APIRouter, Depends

from .deps import get_rpc, get_config
from .models import Balances, Ping

# Public API, requires no auth.
router_public = APIRouter()
router = APIRouter()


@router_public.get('/ping', response_model=Ping)
def ping():
    """simple ping version"""
    return {"status": "pong"}


@router.get('/balance', response_model=Balances)
def balance(rpc=Depends(get_rpc), config=Depends(get_config)) -> Dict[str, str]:
    return rpc._rpc_balance(config['stake_currency'], config.get('fiat_display_currency', ''),)
