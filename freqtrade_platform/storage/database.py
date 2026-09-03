"""SQLite-backed storage foundation for platform metadata."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class PlatformBase(DeclarativeBase):
    """Declarative base for platform-owned tables only."""


class PlatformDatabase:
    """Thin wrapper around a SQLite database for platform-specific entities."""

    def __init__(self, database_url: str = "sqlite:///platform.db") -> None:
        self.database_url = database_url
        self.engine: Engine = create_engine(database_url)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        """Create known platform metadata tables."""
        PlatformBase.metadata.create_all(bind=self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def default_path() -> str:
        """Return the standard local SQLite path for platform metadata."""
        return str(Path("user_data") / "platform" / "platform.db")
