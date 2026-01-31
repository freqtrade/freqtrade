import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_FILE = Path("user_data/generated/runtime/order_id_cache.json")
MAX_CACHE_SIZE = 1000
CACHE_TTL_SEC = 86400  # 24 hours


class OrderIdempotency:
    def __init__(self):
        self._cache = {}  # {client_order_id: timestamp_sec}
        self._ensure_dir()
        self.load()

    def _ensure_dir(self):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create cache dir: {e}")

    def load(self):
        try:
            if CACHE_FILE.exists():
                with CACHE_FILE.open("r") as f:
                    self._cache = json.load(f)
                # Cleanup expired
                self._cleanup()
        except Exception as e:
            logger.error(f"Failed to load idempotency cache: {e}")
            self._cache = {}

    def persist(self):
        try:
            tmp_path = CACHE_FILE.with_suffix(".tmp")
            with tmp_path.open("w") as f:
                json.dump(self._cache, f, indent=2)
            tmp_path.rename(CACHE_FILE)
        except Exception as e:
            logger.error(f"Failed to persist idempotency cache: {e}")

    def _cleanup(self):
        now = time.time()
        initial_count = len(self._cache)
        # Remove old entries
        self._cache = {k: v for k, v in self._cache.items() if now - v < CACHE_TTL_SEC}
        # Enforce size limit (remove oldest)
        if len(self._cache) > MAX_CACHE_SIZE:
            # Sort by timestamp
            sorted_items = sorted(self._cache.items(), key=lambda item: item[1])
            # Keep newest
            self._cache = dict(sorted_items[-MAX_CACHE_SIZE:])

        if len(self._cache) < initial_count:
            pass  # Could log cleanup stats

    def make_client_order_id(self, fields: dict[str, Any]) -> str:
        """
        Generate a stable, deterministic client_order_id.
        Fields required: pair, side, amount, price, timeframe, candle_open_time
        """
        # Canonicalize inputs
        pair = str(fields.get("pair", "")).upper()
        side = str(fields.get("side", "")).lower()
        amount = f"{float(fields.get('amount', 0)):.8f}"
        raw_price = fields.get("price")
        if raw_price in [None, "None", ""]:
            price = "0.00000000"
        else:
            price = f"{float(raw_price):.8f}"

        # Optional context
        tf = fields.get("timeframe", "")
        candle = fields.get("candle_open_time", "")

        # Seed string
        seed = f"{pair}|{side}|{amount}|{price}|{tf}|{candle}"

        # Hash
        # Use MD5 for brevity (12 chars is a good balance for IDs)
        # Security not a concern for collision here vs randomness
        h = hashlib.md5(seed.encode()).hexdigest()

        return f"ft_{h[:12]}"

    def is_duplicate(self, client_order_id: str) -> bool:
        """
        Check if ID exists in cache.
        """
        duplicates = client_order_id in self._cache
        if duplicates:
            # Refresh timestamp? No, strict idempotency means original op matters.
            pass
        return duplicates

    def register(self, client_order_id: str):
        """
        Register a successfully submitted ID.
        """
        self._cache[client_order_id] = time.time()
        self.persist()
