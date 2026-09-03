"""Repository boundary for trading universe persistence."""

from __future__ import annotations

import json

from sqlalchemy import select

from freqtrade_platform.storage.database import PlatformDatabase
from freqtrade_platform.storage.models import PlatformUniverseRecord
from freqtrade_platform.trading.universe import TradingUniverse


class TradingUniverseRepository:
    """Persistence interface for universe metadata."""

    def __init__(self, storage: PlatformDatabase | object | None = None) -> None:
        self._storage = storage
        self._universes: dict[str, TradingUniverse] | None = {} if storage is None else None

    @staticmethod
    def _serialize_list(values: list[str] | None) -> str | None:
        if not values:
            return None
        return json.dumps(values, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _deserialize_list(raw: str | None) -> list[str]:
        if raw in (None, ""):
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive against malformed storage
            raise ValueError(f"malformed universe serialization: {raw!r}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"expected list payload for universe serialization, got {type(parsed).__name__}")
        return [str(item).strip().upper() for item in parsed if str(item).strip()]

    @staticmethod
    def _metadata_to_json(metadata: dict[str, object] | None) -> str | None:
        if not metadata:
            return None
        return json.dumps(metadata, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _metadata_from_json(raw: str | None) -> dict[str, object]:
        if raw in (None, ""):
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive against malformed storage
            raise ValueError(f"malformed metadata serialization: {raw!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"expected dict payload for universe metadata, got {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _record_from_universe(universe: TradingUniverse) -> PlatformUniverseRecord:
        return PlatformUniverseRecord(
            universe_id=universe.universe_id,
            exchange=universe.exchange,
            market_type=universe.market_type,
            include_symbols=TradingUniverseRepository._serialize_list(universe.include_symbols),
            exclude_symbols=TradingUniverseRepository._serialize_list(universe.exclude_symbols),
            max_symbols=universe.max_symbols,
            enabled=universe.enabled,
            metadata_json=TradingUniverseRepository._metadata_to_json(universe.metadata),
        )

    @staticmethod
    def _universe_from_record(record: PlatformUniverseRecord) -> TradingUniverse:
        return TradingUniverse(
            universe_id=record.universe_id,
            exchange=record.exchange,
            market_type=record.market_type,
            include_symbols=TradingUniverseRepository._deserialize_list(record.include_symbols),
            exclude_symbols=TradingUniverseRepository._deserialize_list(record.exclude_symbols),
            max_symbols=record.max_symbols,
            enabled=record.enabled,
            metadata=TradingUniverseRepository._metadata_from_json(record.metadata_json),
        )

    def register(self, universe: TradingUniverse) -> TradingUniverse:
        return self.add(universe)

    def add(self, universe: TradingUniverse) -> TradingUniverse:
        if self._storage is None:
            self._universes[universe.universe_id] = universe
            return universe

        with self._storage.session() as session:
            record = session.scalar(
                select(PlatformUniverseRecord).where(PlatformUniverseRecord.universe_id == universe.universe_id)
            )
            if record is None:
                record = self._record_from_universe(universe)
                session.add(record)
            else:
                record.exchange = universe.exchange
                record.market_type = universe.market_type
                record.include_symbols = self._serialize_list(universe.include_symbols)
                record.exclude_symbols = self._serialize_list(universe.exclude_symbols)
                record.max_symbols = universe.max_symbols
                record.enabled = universe.enabled
                record.metadata_json = self._metadata_to_json(universe.metadata)
        return universe

    def get(self, universe_id: str) -> TradingUniverse | None:
        if self._storage is None:
            return self._universes.get(universe_id)

        with self._storage.session() as session:
            record = session.scalar(
                select(PlatformUniverseRecord).where(PlatformUniverseRecord.universe_id == universe_id)
            )
            if record is None:
                return None
            return self._universe_from_record(record)

    def list(self) -> list[TradingUniverse]:
        if self._storage is None:
            return list(self._universes.values())

        with self._storage.session() as session:
            records = session.scalars(select(PlatformUniverseRecord)).all()
            return [self._universe_from_record(record) for record in records]

    def remove(self, universe_id: str) -> None:
        if self._storage is None:
            self._universes.pop(universe_id, None)
            return

        with self._storage.session() as session:
            record = session.scalar(
                select(PlatformUniverseRecord).where(PlatformUniverseRecord.universe_id == universe_id)
            )
            if record is not None:
                session.delete(record)
