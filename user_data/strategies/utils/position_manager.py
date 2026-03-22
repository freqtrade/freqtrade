"""
Модуль для управления торговыми позициями.
Содержит классы для работы с позициями и их хранения в БД.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Константы
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "positions.db"


@dataclass
class TradePosition:
    """Класс для хранения информации о торговой позиции"""

    id: int = None
    pair: str = None
    entry_time: datetime = None
    entry_price: float = None
    amount: float = None
    stop_loss: float = None
    take_profit: float = None
    exit_time: datetime = None
    exit_price: float = None
    pnl: float = None
    status: str = "open"  # 'open' | 'closed' | 'canceled'

    def to_dict(self) -> dict:
        """Конвертирует позицию в словарь"""
        return {
            "id": self.id,
            "pair": self.pair,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradePosition":
        """Создает экземпляр из словаря"""
        pos = cls()
        pos.id = data.get("id")
        pos.pair = data.get("pair")

        # Преобразуем строки дат обратно в объекты datetime
        entry_time = data.get("entry_time")
        pos.entry_time = datetime.fromisoformat(entry_time) if entry_time else None

        pos.entry_price = data.get("entry_price")
        pos.amount = data.get("amount")
        pos.stop_loss = data.get("stop_loss")
        pos.take_profit = data.get("take_profit")

        exit_time = data.get("exit_time")
        pos.exit_time = datetime.fromisoformat(exit_time) if exit_time else None

        pos.exit_price = data.get("exit_price")
        pos.pnl = data.get("pnl")
        pos.status = data.get("status", "open")

        return pos


class PositionManager:
    """Класс для управления торговыми позициями в SQLite базе данных"""

    def __init__(self, db_path: str = None):
        """Инициализация менеджера позиций"""
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._init_db()

    def _init_db(self):
        """Инициализация базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Создаем таблицу позиций, если она не существует
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        entry_price REAL NOT NULL,
                        amount REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        exit_time TIMESTAMP,
                        exit_price REAL,
                        pnl REAL,
                        status TEXT NOT NULL DEFAULT 'open',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Создаем индексы для ускорения поиска
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_pair ON positions(pair)")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_time)"
                )

                conn.commit()
        except Exception as e:
            print(f"Ошибка при инициализации БД: {e}")
            raise

    def add_position(self, position: TradePosition) -> int:
        """Добавление новой позиции"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO positions
                    (pair, entry_time, entry_price, amount, stop_loss, take_profit, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        position.pair,
                        position.entry_time.isoformat(),
                        position.entry_price,
                        position.amount,
                        position.stop_loss,
                        position.take_profit,
                        position.status,
                    ),
                )

                position_id = cursor.lastrowid
                conn.commit()
                return position_id
        except Exception as e:
            print(f"Ошибка при добавлении позиции: {e}")
            raise

    def update_position(self, position: TradePosition) -> bool:
        """Обновление существующей позиции"""
        if position.id is None:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE positions SET
                        pair = ?,
                        entry_time = ?,
                        entry_price = ?,
                        amount = ?,
                        stop_loss = ?,
                        take_profit = ?,
                        exit_time = ?,
                        exit_price = ?,
                        pnl = ?,
                        status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (
                        position.pair,
                        position.entry_time.isoformat() if position.entry_time else None,
                        position.entry_price,
                        position.amount,
                        position.stop_loss,
                        position.take_profit,
                        position.exit_time.isoformat() if position.exit_time else None,
                        position.exit_price,
                        position.pnl,
                        position.status,
                        position.id,
                    ),
                )

                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Ошибка при обновлении позиции: {e}")
            return False

    def get_position(self, position_id: int) -> Optional[TradePosition]:
        """Получение позиции по ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_position(dict(row))
                return None
        except Exception as e:
            print(f"Ошибка при получении позиции: {e}")
            return None

    def get_open_positions(self, pair: str = None) -> List[TradePosition]:
        """Получение списка открытых позиций"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if pair:
                    cursor.execute(
                        """
                        SELECT * FROM positions
                        WHERE status = 'open' AND pair = ?
                        ORDER BY entry_time DESC
                    """,
                        (pair,),
                    )
                else:
                    cursor.execute("""
                        SELECT * FROM positions
                        WHERE status = 'open'
                        ORDER BY entry_time DESC
                    """)

                return [self._row_to_position(dict(row)) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Ошибка при получении открытых позиций: {e}")
            return []

    def close_position(
        self, position_id: int, exit_price: float, exit_time: datetime = None
    ) -> bool:
        """Закрытие позиции"""
        position = self.get_position(position_id)
        if not position or position.status != "open":
            return False

        position.exit_time = exit_time or datetime.utcnow()
        position.exit_price = exit_price
        position.pnl = (
            (exit_price - position.entry_price) / position.entry_price * 100
        )  # В процентах
        position.status = "closed"

        return self.update_position(position)

    def _row_to_position(self, row: Dict[str, Any]) -> TradePosition:
        """Конвертирует строку из БД в объект TradePosition"""
        return TradePosition(
            id=row["id"],
            pair=row["pair"],
            entry_time=datetime.fromisoformat(row["entry_time"]) if row["entry_time"] else None,
            entry_price=row["entry_price"],
            amount=row["amount"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            exit_price=row["exit_price"],
            pnl=row["pnl"],
            status=row["status"],
        )
