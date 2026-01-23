"""
Stacks blockchain client for ApexTrader-Stacks.
Handles wallet operations and transaction management.
"""

import hashlib
import logging
from typing import Any

from freqtrade.stacks.hiro_api import HiroAPI

logger = logging.getLogger(__name__)


class StacksClient:
    """
    Client for interacting with Stacks blockchain.
    Wraps Hiro API and provides wallet/transaction functionality.
    """

    def __init__(self, config: dict):
        """
        Initialize Stacks client from Freqtrade config.

        :param config: Freqtrade configuration dict
        """
        self.config = config
        stacks_config = config.get("stacks", {})
        exchange_config = config.get("exchange", {})

        self.network = stacks_config.get("network", "testnet")
        self.wallet_address = exchange_config.get("wallet_address", "")
        self.private_key = exchange_config.get("private_key", "")
        self.contract_address = stacks_config.get("escrow_contract", "")

        # Initialize Hiro API
        self.api = HiroAPI(
            network=self.network,
            api_key=stacks_config.get("hiro_api_key"),
        )

        logger.info(
            f"StacksClient initialized: network={self.network}, "
            f"wallet={self.wallet_address[:20]}..."
        )

    def get_stx_balance(self) -> float:
        """
        Get wallet STX balance in STX (not microSTX).

        :return: Balance in STX
        """
        if not self.wallet_address:
            return 0.0
        micro_stx = self.api.get_stx_balance(self.wallet_address)
        return micro_stx / 1_000_000

    def get_account_balances(self) -> dict:
        """
        Get all balances for wallet.

        :return: Balance dict with STX and tokens
        """
        if not self.wallet_address:
            return {}
        return self.api.get_account_balances(self.wallet_address)

    def get_nonce(self) -> int:
        """
        Get current account nonce for transaction building.

        :return: Account nonce
        """
        if not self.wallet_address:
            return 0
        account_info = self.api.get_account_info(self.wallet_address)
        return int(account_info.get("nonce", 0))

    def get_block_height(self) -> int:
        """Get current Stacks block height."""
        return self.api.get_block_height()

    def call_read_only(
        self,
        function_name: str,
        arguments: list[str] | None = None,
        contract_id: str | None = None,
    ) -> dict:
        """
        Call a read-only contract function.

        :param function_name: Function name
        :param arguments: Clarity-encoded arguments
        :param contract_id: Optional contract ID (defaults to escrow contract)
        :return: Function result
        """
        target_contract = contract_id or self.contract_address
        if not target_contract:
            raise ValueError("No contract address configured")

        return self.api.call_read_only(
            contract_id=target_contract,
            function_name=function_name,
            arguments=arguments,
            sender=self.wallet_address,
        )

    def get_transaction_status(self, tx_id: str) -> str:
        """
        Get transaction status.

        :param tx_id: Transaction ID
        :return: Status string
        """
        return self.api.get_transaction_status(tx_id)

    def broadcast_transaction(self, signed_tx_hex: str) -> str:
        """
        Broadcast a signed transaction.

        :param signed_tx_hex: Hex-encoded signed transaction
        :return: Transaction ID
        """
        result = self.api.broadcast_transaction(signed_tx_hex)
        tx_id = result.get("txid", "")
        logger.info(f"Transaction broadcast: {tx_id}")
        return tx_id

    def generate_mock_tx_id(self, prefix: str = "mock") -> str:
        """
        Generate a mock transaction ID for dry-run mode.

        :param prefix: Prefix for the mock ID
        :return: Mock transaction ID
        """
        import time

        data = f"{prefix}-{self.wallet_address}-{time.time()}"
        hash_hex = hashlib.sha256(data.encode()).hexdigest()
        return f"0x{hash_hex[:64]}"

    def is_configured(self) -> bool:
        """Check if client is properly configured."""
        return bool(self.wallet_address and self.contract_address)

    def get_network_info(self) -> dict:
        """Get network configuration info."""
        return {
            "network": self.network,
            "wallet_address": self.wallet_address,
            "contract_address": self.contract_address,
            "api_base_url": self.api.base_url,
        }
