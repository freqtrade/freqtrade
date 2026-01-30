import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HEALTH_FILE = Path("user_data/generated/runtime/health.json")


class HealthSnapshot:
    _instance = None

    def __init__(self):
        self._counters = {"policy_blocks": 0, "degraded_failures": 0}
        self._last_calls = {
            "fetch_ticker_utc": None,
            "fetch_ohlcv_utc": None,
            "create_order_utc": None,
        }
        self._last_error = {"code": None, "message": None}
        self._mode = {"breeze_mock": False, "paper_trading": False, "live_trading_enabled": False}
        self._circuit_breaker = {}
        self._ensure_dir()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_dir(self):
        try:
            HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create health dir: {e}")

    def update_mode(self, mock: bool, paper: bool, live: bool):
        self._mode = {"breeze_mock": mock, "paper_trading": paper, "live_trading_enabled": live}
        self.persist()

    def record_call(self, method_name: str):
        key = f"{method_name}_utc"
        if key in self._last_calls:
            self._last_calls[key] = datetime.now(timezone.utc).isoformat()
            self.persist()

    def increment_counter(self, counter_name: str):
        if counter_name in self._counters:
            self._counters[counter_name] += 1
            self.persist()

    def record_error(self, code: str, message: str):
        self._last_error = {
            "code": code,
            "message": str(message)[:200],  # Truncate for safety
        }
        self.persist()

    def update_circuit_breaker(self, data: dict):
        self._circuit_breaker = data
        self.persist()

    def persist(self):
        data = {
            "meta": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "commit": os.environ.get("GIT_COMMIT", "unknown"),
            },
            "runtime": {"mode": self._mode},
            "last_calls": self._last_calls,
            "counters": self._counters,
            "last_error": self._last_error,
            "circuit_breaker": self._circuit_breaker,
        }

        try:
            tmp_path = HEALTH_FILE.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.rename(HEALTH_FILE)
        except Exception as e:
            logger.error(f"Failed to write health snapshot: {e}")

    def load(self) -> dict:
        try:
            if not HEALTH_FILE.exists():
                return {}
            with open(HEALTH_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load health snapshot: {e}")
            return {}


# Global Helper functions for cleaner usage
def update(event: str, payload: dict | None = None) -> None:
    # Event mapping to internal methods
    instance = HealthSnapshot.get_instance()

    if payload is None:
        payload = {}

    if event == "call":
        method = payload.get("method")
        if method:
            instance.record_call(method)

    elif event == "mode":
        instance.update_mode(
            payload.get("breeze_mock", False),
            payload.get("paper_trading", False),
            payload.get("live_trading_enabled", False),
        )

    elif event == "policy_block":
        instance.increment_counter("policy_blocks")

    elif event == "degraded_failure":
        instance.increment_counter("degraded_failures")

    elif event == "error":
        instance.record_error(payload.get("code", "unknown"), payload.get("message", "unknown"))

    elif event == "circuit_breaker":
        instance.update_circuit_breaker(payload)


def load() -> dict:
    return HealthSnapshot.get_instance().load()
