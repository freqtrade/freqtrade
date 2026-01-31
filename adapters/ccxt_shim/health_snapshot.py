import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

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
        # Limit lists to 50 items
        self._durations = {
            "fetch_ticker": [],
            "fetch_ohlcv": [],
            "create_order": [],
        }
        self._ensure_dir()
        self.load_into_self()

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

    def record_call(self, method_name: str, duration_ms: int | None = None):
        key = f"{method_name}_utc"
        # Update timestamp if key exists (whitelist of tracked methods)
        if key in self._last_calls:
            self._last_calls[key] = datetime.now(timezone.utc).isoformat()

            # Update duration if provided and tracked
            if duration_ms is not None and method_name in self._durations:
                # Keep last 50
                self._durations[method_name].append(duration_ms)
                if len(self._durations[method_name]) > 50:
                    self._durations[method_name].pop(0)

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

    def get_p50_latency(self, method_name: str) -> int:
        durs = self._durations.get(method_name, [])
        if not durs:
            return 0
        sorted_durs = sorted(durs)
        return sorted_durs[len(sorted_durs) // 2]

    def get_counters(self) -> dict:
        return self._counters.copy()

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
            "durations": self._durations,
        }

        try:
            tmp_path = HEALTH_FILE.with_suffix(".tmp")
            with tmp_path.open("w") as f:
                json.dump(data, f, indent=2)
            tmp_path.rename(HEALTH_FILE)
        except Exception as e:
            logger.error(f"Failed to write health snapshot: {e}")

    def load_into_self(self):
        data = self.load()
        if not data:
            return
        self._counters = data.get("counters", self._counters)
        self._last_calls = data.get("last_calls", self._last_calls)
        self._last_error = data.get("last_error", self._last_error)
        self._mode = data.get("runtime", {}).get("mode", self._mode)
        self._circuit_breaker = data.get("circuit_breaker", self._circuit_breaker)
        self._durations = data.get("durations", self._durations)

    def load(self) -> dict:
        try:
            if not HEALTH_FILE.exists():
                return {}
            with HEALTH_FILE.open() as f:
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
        duration = payload.get("duration")  # Optional
        if method:
            instance.record_call(method, duration)

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


def get_p50_latency(method: str) -> int:
    return HealthSnapshot.get_instance().get_p50_latency(method)


def get_counters() -> dict:
    return HealthSnapshot.get_instance().get_counters()
