"""
Модуль для логирования торговых операций.
Содержит классы и перечисления для логирования торговых операций.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timezone

# Настройка логгера
logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Trade operation types"""

    BUY = "BUY"
    SELL = "SELL"
    SIGNAL = "SIGNAL"


class TradeResult(Enum):
    """Trade operation results"""

    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class TradeLogger:
    """Класс для логирования торговых операций"""

    @classmethod
    def log_trade(
        cls,
        pair: str,
        trade_type: TradeType,
        price: float,
        volume: float,
        signal_value: float = None,
        threshold: float = None,
        result: TradeResult = None,
        reason: str = None,
        details: dict = None,
        **kwargs,
    ) -> None:
        """
        Логирование торговой операции с расширенной информацией

        :param pair: Торговая пара (например, 'BTC/USDT')
        :param trade_type: Тип сделки (BUY/SELL/SIGNAL)
        :param price: Цена сделки
        :param volume: Объем сделки в базовой валюте
        :param signal_value: Значение сигнала (например, соотношение объемов)
        :param threshold: Пороговое значение для сигнала
        :param result: Результат операции (SUCCESS/REJECTED/CANCELLED)
        :param reason: Текстовое описание причины принятого решения
        :param details: Детальная информация о состоянии рынка и условиях
        :param kwargs: Дополнительные параметры для логирования

        Формат лога:
        TIMESTAMP | PAIR | TYPE | PRICE | VOLUME | SIGNAL=значение/порог | RESULT | REASON | ДОПОЛНИТЕЛЬНЫЕ_ПАРАМЕТРЫ
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # Основное сообщение лога (без временной метки, так как она добавляется логгером)
            log_parts = [f"{pair}", trade_type.value, f"PRICE={price:.8f}", f"VOLUME={volume:.8f}"]

            # Информация о сигнале больше не выводится в кратком логе
            # Детали сигнала доступны в сводке стакана

            # Добавляем результат и причину
            if result:
                log_parts.append(f"RESULT={result.value}")
            if reason:
                # Ограничиваем длину причины для основного лога
                short_reason = (reason[:97] + "...") if len(reason) > 100 else reason
                log_parts.append(f"REASON={short_reason}")

            # Добавляем сумму ордера для успешных сделок SELL
            if result == TradeResult.SUCCESS and trade_type == TradeType.SELL:
                amount = price * volume
                log_parts.append(f"AMOUNT={amount:.8f}")

            # Добавляем дополнительные параметры
            for key, value in kwargs.items():
                if value is not None:
                    log_parts.append(f"{key.upper()}={value}")

            # Собираем итоговое сообщение
            log_message = " | ".join(log_parts)

            # Логируем в зависимости от типа операции и результата
            if result == TradeResult.SUCCESS:
                logger.info(log_message)
            elif result == TradeResult.REJECTED:
                logger.warning(log_message)
            elif result == TradeResult.CANCELLED:
                logger.warning(log_message)
            else:
                logger.info(log_message)

            # Детальное логирование только для успешных сделок
            if result == TradeResult.SUCCESS and details:
                cls._log_detailed_info(
                    pair,
                    trade_type,
                    price,
                    volume,
                    signal_value,
                    threshold,
                    result,
                    reason,
                    details,
                )

        except Exception as e:
            logger.error(f"Ошибка при логировании сделки: {e}")

    @classmethod
    def _log_detailed_info(
        cls,
        pair: str,
        trade_type: TradeType,
        price: float,
        volume: float,
        signal_value: float,
        threshold: float,
        result: TradeResult,
        reason: str,
        details: dict,
    ) -> None:
        """
        Детальное логирование информации о сделке
        """
        try:
            # Собираем все строки лога в один буфер
            log_lines = []

            # Добавляем заголовок (без переноса строки в начале)
            log_lines.append(f"=== ДЕТАЛИ СДЕЛКИ {pair} ===")

            # Добавляем общую информацию
            log_lines.append(f"Тип: {trade_type.value}")
            log_lines.append(f"Цена: {price:.8f}")
            log_lines.append(f"Объем: {volume:.8f}")
            log_lines.append(f"Статус: {result.value}")
            log_lines.append(f"Причина: {reason}")

            # Добавляем информацию о сигнале
            if signal_value is not None and threshold is not None:
                signal_strength = signal_value / threshold if threshold != 0 else 0
                log_lines.append(f"\nСигнал: {signal_value:.8f} (порог: {threshold:.8f})")
                log_lines.append(
                    f"Соотношение: x{signal_strength:.4f} ({'Выше порога' if signal_strength >= 1.0 else 'Ниже порога'})"
                )

            # Добавляем детали сделки
            if details:
                log_lines.append("\nДетали:")
                for key, value in details.items():
                    if isinstance(value, dict):
                        logger.info(f"  {key}:")
                        for k, v in value.items():
                            logger.info(f"    {k}: {v}")
                    else:
                        logger.info(f"  {key}: {value}")

            # Причина и результат
            if reason:
                logger.info("\n📝 ПРИЧИНА:")
                for line in reason.split("\n"):
                    logger.info(f"  {line}")

            if result:
                logger.info(f"\n✅ РЕЗУЛЬТАТ: {result.value}")
                logger.info("\n".join(log_lines))

        except Exception as e:
            logger.error(f"Ошибка при детальном логировании: {e}")

    @staticmethod
    def filter_logs(
        log_file: str = None,
        pair: str = None,
        trade_type: str = None,
        date_from: str = None,
        date_to: str = None,
        result: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Фильтрация логов по заданным параметрам

        :param log_file: Путь к файлу лога
        :param pair: Фильтр по торговой паре
        :param trade_type: Фильтр по типу сделки
        :param date_from: Начальная дата (ГГГГ-ММ-ДД)
        :param date_to: Конечная дата (ГГГГ-ММ-ДД)
        :param result: Фильтр по результату
        :return: Список отфильтрованных записей
        """
        if not log_file or not Path(log_file).exists():
            return []

        try:
            result = []

            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Парсим строку лога
                        # Пример: 2023-01-01 12:00:00 | BTC/USDT | BUY | PRICE=10000.00 | VOLUME=0.01 | RESULT=SUCCESS
                        parts = line.strip().split(" | ")
                        if len(parts) < 3:
                            continue

                        # Парсим дату и время
                        log_time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")

                        # Парсим остальные параметры
                        log_data = {"timestamp": log_time}
                        for part in parts[1:]:
                            if "=" in part:
                                key, value = part.split("=", 1)
                                log_data[key.lower()] = value

                        # Применяем фильтры
                        if pair and log_data.get("pair") != pair:
                            continue

                        if trade_type and log_data.get("type") != trade_type.upper():
                            continue

                        if date_from:
                            filter_date = datetime.strptime(date_from, "%Y-%m-%d")
                            if log_time.date() < filter_date.date():
                                continue

                        if date_to:
                            filter_date = datetime.strptime(date_to, "%Y-%m-%d")
                            if log_time.date() > filter_date.date():
                                continue

                        if result and log_data.get("result") != result.upper():
                            continue

                        result.append(log_data)

                    except Exception as e:
                        logger.error(
                            f"Ошибка при парсинге строки лога: {line.strip()}, ошибка: {e}"
                        )

            return result

        except Exception as e:
            logger.error(f"Ошибка при чтении файла логов: {e}")
            return []
