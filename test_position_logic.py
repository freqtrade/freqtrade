#!/usr/bin/env python3
"""
Тестовый скрипт для проверки логики расчета размера позиции
"""

import logging
import os
import sys
from datetime import datetime


# Добавляем путь к freqtrade
sys.path.append("/home/dm/freqtrade")

from freqtrade.configuration import Configuration
from freqtrade.exchange import Exchange


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_position_size_logic():
    """Тестирует логику расчета размера позиции"""

    # Загружаем конфигурацию
    config = Configuration.from_files(["user_data/config_OrderBookDepth.json"])

    # Устанавливаем API ключи из переменных окружения
    config["exchange"]["key"] = os.getenv("BINANCE_API_KEY", "")
    config["exchange"]["secret"] = os.getenv("BINANCE_API_SECRET", "")

    # Инициализируем exchange
    exchange = Exchange(config)

    # Получаем stake_amount из конфига
    stake_amount = config.get("stake_amount", 100)
    logger.info(f"Stake amount из конфига: {stake_amount} USDT")

    # Тестовые пары
    test_pairs = ["BTC/USDT", "ETH/USDT"]

    for pair in test_pairs:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Тестирование пары: {pair}")

            # Получаем информацию о рынке
            market = exchange.market(pair)
            min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
            min_cost = market.get("limits", {}).get("cost", {}).get("min", 0)

            logger.info(f"Ограничения биржи для {pair}:")
            logger.info(f"  - Минимальное количество: {min_amount}")
            logger.info(f"  - Минимальная стоимость: {min_cost} USDT")

            # Получаем текущую цену
            ticker = exchange.fetch_ticker(pair)
            current_rate = ticker["last"]

            logger.info(f"Текущая цена: {current_rate}")

            # Рассчитываем минимальный размер в USDT
            exchange_min_stake = (
                max(min_cost, min_amount * current_rate)
                if min_amount > 0 and min_cost > 0
                else min_cost
            )

            logger.info(f"Рассчитанный минимальный размер: {exchange_min_stake:.2f} USDT")

            # Диапазон: от биржевого минимума до stake_amount
            min_position = exchange_min_stake
            max_position = stake_amount

            logger.info(f"Диапазон для {pair}:")
            logger.info(f"  - Минимум: {min_position:.2f} USDT (биржевой минимум)")
            logger.info(f"  - Максимум: {max_position:.2f} USDT (stake_amount)")

            # Тестируем фиксированный размер
            fixed_size = (min_position + max_position) / 2
            logger.info(f"  - Фиксированный размер: {fixed_size:.2f} USDT")

            # Симулируем ATR-расчеты для тестирования логики
            test_atr_values = [
                min_position * 0.5,  # Меньше минимума
                min_position,  # Равен минимуму
                (min_position + max_position) / 2,  # В середине диапазона
                max_position,  # Равен максимуму
                max_position * 1.5,  # Больше максимума
            ]

            logger.info(f"\nТестирование ATR логики:")
            for i, atr_size in enumerate(test_atr_values):
                if atr_size > max_position:
                    result = max_position
                    reason = f"ATR ({atr_size:.2f}) > max ({max_position:.2f})"
                elif atr_size < min_position:
                    result = min_position
                    reason = f"ATR ({atr_size:.2f}) < min ({min_position:.2f})"
                else:
                    result = atr_size
                    reason = f"ATR ({atr_size:.2f}) в диапазоне"

                logger.info(f"  Тест {i + 1}: {reason} → Результат: {result:.2f} USDT")

        except Exception as e:
            logger.error(f"Ошибка при тестировании {pair}: {e}")


if __name__ == "__main__":
    test_position_size_logic()
