"""
Hiro API wrapper for Stacks blockchain queries.
Documentation: https://docs.hiro.so/stacks/api
"""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class HiroAPI:
    """
    Wrapper for Hiro API endpoints.
    Used for blockchain state queries, account balances, and contract interactions.
    """

    MAINNET_URL = "https://api.mainnet.hiro.so"
    TESTNET_URL = "https://api.testnet.hiro.so"

    def __init__(self, network: str = "testnet", api_key: str | None = None):
        """
        Initialize Hiro API client.

        :param network: 'mainnet' or 'testnet'
        :param api_key: Optional Hiro API key for higher rate limits
        """
        self.network = network
        self.base_url = self.MAINNET_URL if network == "mainnet" else self.TESTNET_URL
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        if api_key:
            self._session.headers["x-api-key"] = api_key

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make GET request to Hiro API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Hiro API GET error: {e}")
            raise

    def _post(self, endpoint: str, data: dict | None = None) -> dict:
        """Make POST request to Hiro API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Hiro API POST error: {e}")
            raise

    def get_account_balances(self, address: str) -> dict:
        """
        Get STX and token balances for an address.

        :param address: Stacks address (e.g., 'ST1PQHQKV0RJXZFY1DGX8MNSNYVE3VGZJSRTPGZGM')
        :return: Balance information including STX and fungible tokens
        """
        return self._get(f"/extended/v1/address/{address}/balances")

    def get_stx_balance(self, address: str) -> int:
        """
        Get STX balance in microSTX.

        :param address: Stacks address
        :return: Balance in microSTX (1 STX = 1,000,000 microSTX)
        """
        data = self.get_account_balances(address)
        return int(data.get("stx", {}).get("balance", 0))

    def get_block_height(self) -> int:
        """Get current Stacks block height."""
        data = self._get("/extended/v1/block", params={"limit": 1})
        if data.get("results"):
            return data["results"][0].get("height", 0)
        return 0

    def get_account_info(self, address: str) -> dict:
        """Get account information including nonce."""
        return self._get(f"/v2/accounts/{address}")

    def get_contract_info(self, contract_id: str) -> dict:
        """
        Get contract interface.

        :param contract_id: Contract identifier (e.g., 'ST1PQHQKV0RJXZFY1DGX8MNSNYVE3VGZJSRTPGZGM.usdcx-escrow')
        :return: Contract interface including functions and maps
        """
        return self._get(f"/v2/contracts/interface/{contract_id}")

    def call_read_only(
        self,
        contract_id: str,
        function_name: str,
        arguments: list[str] | None = None,
        sender: str | None = None,
    ) -> dict:
        """
        Call a read-only contract function.

        :param contract_id: Contract identifier
        :param function_name: Function name to call
        :param arguments: List of Clarity value arguments (hex-encoded)
        :param sender: Optional sender address for context
        :return: Function result
        """
        # Split contract_id into address and name
        parts = contract_id.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid contract_id format: {contract_id}")

        contract_address, contract_name = parts

        payload = {
            "sender": sender or contract_address,
            "arguments": arguments or [],
        }

        return self._post(
            f"/v2/contracts/call-read/{contract_address}/{contract_name}/{function_name}",
            data=payload,
        )

    def broadcast_transaction(self, signed_tx_hex: str) -> dict:
        """
        Broadcast a signed transaction to the network.

        :param signed_tx_hex: Hex-encoded signed transaction
        :return: Transaction result with txid
        """
        url = f"{self.base_url}/v2/transactions"
        headers = {"Content-Type": "application/octet-stream"}
        try:
            response = self._session.post(
                url,
                data=bytes.fromhex(signed_tx_hex),
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Broadcast transaction error: {e}")
            raise

    def get_transaction(self, tx_id: str) -> dict:
        """
        Get transaction details by ID.

        :param tx_id: Transaction ID (with or without 0x prefix)
        :return: Transaction details including status
        """
        # Ensure tx_id has 0x prefix
        if not tx_id.startswith("0x"):
            tx_id = f"0x{tx_id}"
        return self._get(f"/extended/v1/tx/{tx_id}")

    def get_transaction_status(self, tx_id: str) -> str:
        """
        Get transaction status.

        :param tx_id: Transaction ID
        :return: Status string ('pending', 'success', 'failed', etc.)
        """
        tx = self.get_transaction(tx_id)
        return tx.get("tx_status", "unknown")
