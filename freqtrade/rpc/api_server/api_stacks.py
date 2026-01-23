"""
Stacks blockchain API endpoints for ApexTrader-Stacks.
Provides escrow balance and status information.
"""

import logging

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from freqtrade.rpc.api_server.deps import get_config, get_exchange


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stacks", tags=["stacks"])


class EscrowBalance(BaseModel):
    """Escrow balance response schema."""
    balance: float
    currency: str
    contract_id: str
    network: str
    dry_run: bool
    is_configured: bool


class StacksStatus(BaseModel):
    """Stacks integration status response schema."""
    enabled: bool
    network: str
    escrow_contract: str | None
    wallet_address: str | None
    stx_balance: float
    escrow_balance: float
    is_bot_authorized: bool
    is_paused: bool


@router.get("/escrow/balance", response_model=EscrowBalance)
def get_escrow_balance(config=Depends(get_config)):
    """
    Get current escrow balance from Stacks contract.

    Returns the user's USDCx balance held in the escrow smart contract,
    along with contract and network details.
    """
    stacks_config = config.get("stacks", {})

    if not stacks_config.get("enabled", False):
        raise HTTPException(
            status_code=400,
            detail="Stacks integration is not enabled"
        )

    # Check if exchange is Stacks
    exchange_config = config.get("exchange", {})
    if exchange_config.get("name", "").lower() != "stacks":
        raise HTTPException(
            status_code=400,
            detail="Exchange is not configured as Stacks"
        )

    try:
        # Try to get exchange and escrow manager
        from freqtrade.exchange.stacks import Stacks

        # Create temporary exchange instance for balance check
        exchange = Stacks(config, validate=False)
        escrow = exchange.escrow

        status = escrow.get_status()

        return EscrowBalance(
            balance=status.get("balance", 0.0),
            currency="USDCx",
            contract_id=status.get("contract_id", ""),
            network=status.get("network", "testnet"),
            dry_run=status.get("dry_run", True),
            is_configured=status.get("is_configured", False),
        )
    except Exception as e:
        logger.error(f"Error fetching escrow balance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch escrow balance: {str(e)}"
        )


@router.get("/status", response_model=StacksStatus)
def get_stacks_status(config=Depends(get_config)):
    """
    Get full Stacks integration status.

    Returns comprehensive status including wallet, escrow, and
    contract state information.
    """
    stacks_config = config.get("stacks", {})
    exchange_config = config.get("exchange", {})

    if not stacks_config.get("enabled", False):
        return StacksStatus(
            enabled=False,
            network="unknown",
            escrow_contract=None,
            wallet_address=None,
            stx_balance=0.0,
            escrow_balance=0.0,
            is_bot_authorized=False,
            is_paused=True,
        )

    try:
        from freqtrade.exchange.stacks import Stacks

        exchange = Stacks(config, validate=False)
        escrow = exchange.escrow
        client = exchange.stacks_client

        return StacksStatus(
            enabled=True,
            network=stacks_config.get("network", "testnet"),
            escrow_contract=stacks_config.get("escrow_contract"),
            wallet_address=exchange_config.get("wallet_address"),
            stx_balance=client.get_stx_balance(),
            escrow_balance=escrow.get_escrow_balance(),
            is_bot_authorized=escrow.is_bot_authorized(),
            is_paused=escrow.is_paused(),
        )
    except Exception as e:
        logger.error(f"Error fetching Stacks status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Stacks status: {str(e)}"
        )


@router.post("/escrow/deposit")
def deposit_to_escrow(amount: float, config=Depends(get_config)):
    """
    Deposit funds to escrow (dry-run simulation only).

    In production, deposits must be done via Stacks wallet.
    This endpoint is for testing/demo purposes.
    """
    stacks_config = config.get("stacks", {})

    if not stacks_config.get("enabled", False):
        raise HTTPException(status_code=400, detail="Stacks not enabled")

    if not config.get("dry_run", True):
        raise HTTPException(
            status_code=400,
            detail="Deposits only allowed in dry-run mode via API"
        )

    try:
        from freqtrade.exchange.stacks import Stacks

        exchange = Stacks(config, validate=False)
        tx_id = exchange.escrow.deposit(amount)

        return {
            "status": "success",
            "amount": amount,
            "tx_id": tx_id,
            "new_balance": exchange.escrow.get_escrow_balance(),
        }
    except Exception as e:
        logger.error(f"Deposit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/escrow/withdraw")
def withdraw_from_escrow(amount: float, config=Depends(get_config)):
    """
    Withdraw funds from escrow (dry-run simulation only).

    In production, withdrawals must be done via Stacks wallet.
    This endpoint is for testing/demo purposes.
    """
    stacks_config = config.get("stacks", {})

    if not stacks_config.get("enabled", False):
        raise HTTPException(status_code=400, detail="Stacks not enabled")

    if not config.get("dry_run", True):
        raise HTTPException(
            status_code=400,
            detail="Withdrawals only allowed in dry-run mode via API"
        )

    try:
        from freqtrade.exchange.stacks import Stacks

        exchange = Stacks(config, validate=False)
        tx_id = exchange.escrow.withdraw(amount)

        if tx_id is None:
            raise HTTPException(
                status_code=400,
                detail="Withdrawal failed - insufficient balance"
            )

        return {
            "status": "success",
            "amount": amount,
            "tx_id": tx_id,
            "new_balance": exchange.escrow.get_escrow_balance(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Withdrawal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
