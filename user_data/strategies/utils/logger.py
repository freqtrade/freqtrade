import logging
from enum import Enum
from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime


class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SIGNAL = "SIGNAL"


class TradeResult(Enum):
    SUCCESS = "УСПЕШНО"
    REJECTED = "ОТКЛОНЕНО"


@dataclass
class TradeLogEntry:
    timestamp: datetime
    pair: str
    trade_type: Union[TradeType, str]
    price: float
    volume: float
    signal_value: str
    threshold: float
    result: TradeResult
    reason: str = ""


class TradeLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        self.logger = logging.getLogger("TradeLogger")
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        # Create file handler
        log_file = log_dir / "trades.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    @classmethod
    def log_trade(
        cls,
        pair: str,
        trade_type: Union[TradeType, str],
        price: float,
        volume: float,
        signal_value: str,
        threshold: float,
        result: TradeResult,
        reason: str = "",
    ):
        """
        Логирование торговой операции

        Args:
            pair: Торговая пара (например, 'BTC/USDT')
            trade_type: Тип операции (BUY/SELL/SIGNAL)
            price: Цена
            volume: Объем
            signal_value: Значение сигнала
            threshold: Пороговое значение
            result: Результат операции
            reason: Дополнительная информация
        """
        # Преобразуем строку в TradeType если нужно
        if isinstance(trade_type, str):
            try:
                trade_type = TradeType[trade_type.upper()]
            except KeyError:
                trade_type = TradeType.SIGNAL

        # Формируем сообщение
        msg = (
            f"{pair} | {trade_type.value} | "
            f"Цена: {price:.8f} | "
            f"Объем: {volume:.8f} | "
            f"Сигнал: {signal_value} | "
            f"Порог: {threshold:.2f} | "
            f"RESULT={result.value}"
        )

        if reason:
            msg += f" | Причина: {reason}"

        # Логируем
        cls().logger.info(msg)
