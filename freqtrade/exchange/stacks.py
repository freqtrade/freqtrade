"""
Stacks DEX exchange wrapper for ApexTrader-Stacks.
Implements Exchange interface for Stacks blockchain trading with escrow integration.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas, Tickers
from freqtrade.stacks import EscrowManager, StacksClient

logger = logging.getLogger(__name__)


class Stacks(Exchange):
    """
    Stacks blockchain exchange wrapper.
    Integrates with Stacks DEXs and escrow contracts for automated trading.

    Note: This is a custom implementation that doesn't use CCXT.
    Market data is simulated for the hackathon demo.
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC"],
        "ohlcv_has_history": False,
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,
        "ohlcv_params": {},
        "ohlcv_volume_currency": "base",
        "trades_has_history": False,
        "trades_pagination": "time",
        "trades_pagination_arg": "since",
        "trades_limit": 1000,
        "tickers_have_bid_ask": False,
        "tickers_have_percentage": False,
        "tickers_have_quoteVolume": False,
        "tickers_have_price": True,
        "l2_limit_range": None,
        "l2_limit_range_required": False,
        "ws_enabled": False,
        "needs_trading_fees": False,
        "order_props_in_contracts": ["amount", "filled", "remaining"],
        "marketOrderRequiresPrice": False,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
    ]

    def __init__(self, config: dict, *, validate: bool = True, **kwargs) -> None:
        """
        Initialize Stacks exchange.

        :param config: Freqtrade configuration
        :param validate: Whether to validate configuration
        """
        self._config = config
        self._stacks_config = config.get("stacks", {})

        # Initialize Stacks clients
        self._stacks_client = StacksClient(config)
        self._escrow = EscrowManager(self._stacks_client)

        # Initialize simulated state
        self._markets: dict[str, dict] = {}
        self._simulated_orders: dict[str, dict] = {}
        self._order_counter = 0

        # Set exchange properties
        self.trading_mode = TradingMode.SPOT
        self.margin_mode = MarginMode.NONE
        self._dry_run = config.get("dry_run", True)

        # Initialize attributes expected by base Exchange class
        self._exchange_ws = None
        self._api = None
        self._api_async = None
        self._ws_async = None
        self._last_markets_refresh = 0
        self.loop = None
        self._precision_mode = 0  # Use underscore to avoid property conflict
        self._trades_pagination = "time"
        self._trades_pagination_arg = "since"
        self._ft_has_default = self._ft_has.copy()
        self._ft_has = {**self._ft_has_default, **self._ft_has}
        self.required_candle_call_count = 1
        self._markets_last_refresh_ts = 0

        # OHLCV cache
        self._klines: dict = {}
        self._pairs_last_refresh_time: dict = {}

        # Threading lock
        import threading
        self._loop_lock = threading.Lock()

        # Initialize demo markets
        self._init_demo_markets()

        logger.info(
            f"Stacks exchange initialized: network={self._stacks_client.network}, "
            f"dry_run={self._dry_run}"
        )

    def _init_demo_markets(self) -> None:
        """Initialize demo trading pairs for hackathon."""
        demo_pairs = [
            ("STX", "USDCx", 1.50),
            ("ALEX", "USDCx", 0.15),
            ("WELSH", "USDCx", 0.001),
        ]

        for base, quote, price in demo_pairs:
            symbol = f"{base}/{quote}"
            self._markets[symbol] = {
                "id": symbol.lower().replace("/", "-"),
                "symbol": symbol,
                "base": base,
                "quote": quote,
                "baseId": base.lower(),
                "quoteId": quote.lower(),
                "active": True,
                "spot": True,
                "type": "spot",
                "precision": {"amount": 6, "price": 6},
                "limits": {
                    "amount": {"min": 0.000001, "max": 1000000},
                    "price": {"min": 0.000001, "max": 1000000},
                    "cost": {"min": 0.01, "max": 1000000},
                },
                "info": {
                    "demo_price": price,
                    "stacks_contract": None,
                },
            }

    @property
    def precisionMode(self) -> int:
        """Override base class property - Stacks uses decimal precision."""
        return self._precision_mode

    @property
    def precision_mode_price(self) -> int:
        """Override base class property."""
        return self._precision_mode

    @property
    def timeframes(self) -> list[str]:
        """Supported timeframes for Stacks (simulated data)."""
        return ["1m", "5m", "15m", "1h", "4h", "1d"]

    @property
    def escrow(self) -> EscrowManager:
        """Access escrow manager."""
        return self._escrow

    @property
    def stacks_client(self) -> StacksClient:
        """Access Stacks client."""
        return self._stacks_client

    @property
    def markets(self) -> dict[str, Any]:
        """Return available markets."""
        return self._markets

    @property
    def name(self) -> str:
        """Exchange name."""
        return "Stacks"

    def validate_config(self, config: dict) -> None:
        """Validate Stacks configuration."""
        stacks_config = config.get("stacks", {})
        if not stacks_config.get("enabled", False):
            logger.warning("Stacks integration is not enabled in config")

        exchange_config = config.get("exchange", {})
        if not exchange_config.get("wallet_address"):
            logger.warning("No wallet_address configured for Stacks")

    def get_balances(self) -> dict[str, Any]:
        """
        Fetch balances from escrow contract.

        :return: Balance dict compatible with Freqtrade
        """
        escrow_balance = self._escrow.get_escrow_balance()
        stx_balance = self._stacks_client.get_stx_balance()

        return {
            "USDCx": {
                "free": escrow_balance,
                "used": 0.0,
                "total": escrow_balance,
            },
            "STX": {
                "free": stx_balance,
                "used": 0.0,
                "total": stx_balance,
            },
        }

    def get_balance(self, currency: str) -> dict[str, float]:
        """Get balance for specific currency."""
        balances = self.get_balances()
        return balances.get(currency, {"free": 0.0, "used": 0.0, "total": 0.0})

    def fetch_ticker(self, pair: str) -> dict[str, Any]:
        """
        Fetch ticker data for a pair.
        Returns simulated data for demo.
        """
        if pair not in self._markets:
            raise ValueError(f"Unknown pair: {pair}")

        market = self._markets[pair]
        base_price = market["info"].get("demo_price", 1.0)

        # Add small random variance for realism
        import random
        variance = random.uniform(-0.02, 0.02)
        price = base_price * (1 + variance)

        now = datetime.now(timezone.utc)
        return {
            "symbol": pair,
            "timestamp": int(now.timestamp() * 1000),
            "datetime": now.isoformat(),
            "last": price,
            "bid": price * 0.999,
            "ask": price * 1.001,
            "high": price * 1.05,
            "low": price * 0.95,
            "open": base_price,
            "close": price,
            "volume": 1000000.0,
            "quoteVolume": 1000000.0 * price,
        }

    def fetch_tickers(self, symbols: list[str] | None = None) -> Tickers:
        """Fetch tickers for multiple pairs."""
        target_symbols = symbols or list(self._markets.keys())
        return {s: self.fetch_ticker(s) for s in target_symbols if s in self._markets}

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float = 1.0,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create an order with escrow integration.

        :param pair: Trading pair
        :param ordertype: 'limit' or 'market'
        :param side: 'buy' or 'sell'
        :param amount: Order amount
        :param rate: Order price
        :return: Order dict
        """
        self._order_counter += 1
        order_id = f"stx-{self._order_counter}-{int(datetime.now(timezone.utc).timestamp())}"
        cost = amount * rate

        # Request trade approval from escrow
        tx_id = self._escrow.request_trade_approval(
            amount=cost,
            trade_id=order_id,
            expiry_blocks=100,
        )

        logger.info(f"Created Stacks order {order_id}, escrow tx: {tx_id}")

        now = datetime.now(timezone.utc)
        order = {
            "id": order_id,
            "clientOrderId": order_id,
            "symbol": pair,
            "type": ordertype,
            "side": side,
            "amount": amount,
            "price": rate,
            "cost": cost,
            "average": rate,
            "filled": amount,  # Immediately filled for demo
            "remaining": 0,
            "status": "closed",
            "fee": {"cost": cost * 0.003, "currency": "USDCx"},
            "timestamp": int(now.timestamp() * 1000),
            "datetime": now.isoformat(),
            "info": {
                "escrow_tx": tx_id,
                "stacks_network": self._stacks_config.get("network", "testnet"),
            },
        }

        self._simulated_orders[order_id] = order
        return order

    def fetch_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        """Fetch order by ID."""
        if order_id in self._simulated_orders:
            return self._simulated_orders[order_id]
        raise ValueError(f"Order {order_id} not found")

    def cancel_order(self, order_id: str, pair: str, params: dict | None = None) -> dict:
        """Cancel an order."""
        if order_id in self._simulated_orders:
            order = self._simulated_orders[order_id]
            order["status"] = "canceled"
            return order
        raise ValueError(f"Order {order_id} not found")

    def get_fee(
        self,
        symbol: str,
        type: str = "",
        side: str = "",
        amount: float = 1,
        price: float = 1,
        taker_or_maker: str = "maker",
    ) -> float:
        """Return trading fee - 0.3% for Stacks DEXs."""
        return 0.003

    def get_markets(
        self,
        base_currencies: list[str] | None = None,
        quote_currencies: list[str] | None = None,
        tradable_only: bool = True,
        active_only: bool = True,
        spot_only: bool = False,
    ) -> dict[str, Any]:
        """Get filtered markets."""
        result = {}
        for symbol, market in self._markets.items():
            if base_currencies and market["base"] not in base_currencies:
                continue
            if quote_currencies and market["quote"] not in quote_currencies:
                continue
            if active_only and not market.get("active", True):
                continue
            result[symbol] = market
        return result

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        """Check if market is tradable."""
        return market.get("active", False) and market.get("spot", False)

    def get_pair_quote_currency(self, pair: str) -> str:
        """Get quote currency for a pair."""
        if pair in self._markets:
            return self._markets[pair]["quote"]
        return pair.split("/")[1] if "/" in pair else ""

    def get_pair_base_currency(self, pair: str) -> str:
        """Get base currency for a pair."""
        if pair in self._markets:
            return self._markets[pair]["base"]
        return pair.split("/")[0] if "/" in pair else ""

    def reload_markets(self, force: bool = False) -> None:
        """Reload markets - no-op for Stacks."""
        self._last_markets_refresh = int(datetime.now(timezone.utc).timestamp())
        self._markets_last_refresh_ts = self._last_markets_refresh

    def close(self) -> None:
        """Close exchange connections."""
        pass  # No async connections to close

    def validate_stakecurrency(self, stake_currency: str) -> None:
        """Validate stake currency - allow USDCx."""
        valid = ["USDCx", "STX"]
        if stake_currency not in valid:
            logger.warning(f"Stake currency {stake_currency} may not be supported")

    def validate_pairs(self, pairs: list[str]) -> None:
        """Validate trading pairs exist in markets."""
        for pair in pairs:
            if pair not in self._markets:
                logger.warning(f"Pair {pair} not in Stacks markets")

    def validate_ordertypes(self, order_types: dict) -> None:
        """Validate order types - only limit/market supported."""
        pass

    def validate_order_time_in_force(self, order_time_in_force: dict) -> None:
        """Validate time in force - only GTC supported."""
        pass

    def validate_trading_mode_and_target(
        self, trading_mode: TradingMode, margin_mode: MarginMode
    ) -> None:
        """Validate trading mode - only spot supported."""
        if trading_mode != TradingMode.SPOT:
            raise ValueError("Stacks only supports spot trading")

    def exchange_has(self, feature: str) -> bool:
        """Check if exchange has feature."""
        return self._ft_has.get(feature, False)

    def refresh_latest_ohlcv(
        self,
        pair_list: list[tuple[str, str, CandleType]],
        *,
        since_ms: int | None = None,
        cache: bool = True,
        drop_incomplete: bool | None = None,
    ) -> dict:
        """
        Override to provide simulated OHLCV data for demo.
        Returns empty dataframes since we don't have real market data.
        """
        import pandas as pd

        result = {}
        for item in pair_list:
            if len(item) == 3:
                pair, timeframe, candle_type = item
            else:
                pair, timeframe = item
                candle_type = CandleType.SPOT

            key = (pair, timeframe, candle_type)

            # Create empty dataframe with correct structure
            df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            self._klines[key] = df
            result[key] = df

        return result

    # Stubs for required methods
    def fetch_ohlcv(self, pair: str, timeframe: str = "5m", **kwargs) -> list:
        """OHLCV not supported - return empty."""
        logger.warning("OHLCV data not available for Stacks demo")
        return []

    def fetch_trades(self, pair: str, **kwargs) -> list:
        """Trade history not supported."""
        return []

    def fetch_order_book(self, pair: str, limit: int = 100) -> dict:
        """Order book not supported - return simulated."""
        ticker = self.fetch_ticker(pair)
        return {
            "bids": [[ticker["bid"], 1000]],
            "asks": [[ticker["ask"], 1000]],
            "timestamp": ticker["timestamp"],
            "datetime": ticker["datetime"],
        }
