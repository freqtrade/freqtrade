"""SQLite-backed storage foundation for platform metadata."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects import sqlite
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
        """Create known platform metadata tables and migrate existing SQLite metadata."""
        import freqtrade_platform.storage.models  # noqa: F401
        PlatformBase.metadata.create_all(bind=self.engine, checkfirst=True)
        self.migrate_existing_database()

    def migrate_existing_database(self) -> None:
        """Add missing columns to legacy SQLite tables without a heavyweight migration tool."""
        if self.engine.url.get_backend_name() != "sqlite":
            return

        with self.engine.begin() as connection:
            inspector = inspect(connection)
            for table in PlatformBase.metadata.sorted_tables:
                if not inspector.has_table(table.name):
                    continue

                existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
                for column in table.columns:
                    if column.name in existing_columns:
                        continue
                    column_type = column.type.compile(dialect=sqlite.dialect())
                    sql = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column_type}"
                    if not column.nullable:
                        sql = f"{sql} NOT NULL"
                    connection.execute(text(sql))

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
