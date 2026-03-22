#!/usr/bin/env python3
"""
Тестовый скрипт для проверки расчета размера позиции
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd


# Добавляем путь к freqtrade
sys.path.append("/home/dm/freqtrade")

from freqtrade.configuration import Configuration
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exchange import Exchange


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_position_size_calculation():
    """Тестирует расчет размера позиции"""

    # Загружаем конфигурацию
    config = Configuration.from_files(["user_data/config_DCA.json"])

    # Инициализируем exchange
    exchange = Exchange(config)

    # Тестовые пары
    test_pairs = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

    for pair in test_pairs:
        try:
            logger.info(f"\n{'=' * 50}")
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

            # Наши настройки
            config_stake_amount = 100
            min_position_size_pct = 50
            max_position_size_pct = 100

            our_min = config_stake_amount * (min_position_size_pct / 100)
            our_max = config_stake_amount * (max_position_size_pct / 100)

            # Финальный минимальный размер
            final_min = max(exchange_min_stake, our_min)

            logger.info(f"Наши настройки:")
            logger.info(f"  - Минимальный размер: {our_min:.2f} USDT")
            logger.info(f"  - Максимальный размер: {our_max:.2f} USDT")
            logger.info(f"  - Финальный минимальный размер: {final_min:.2f} USDT")

            # Тестируем фиксированный размер (75 USDT)
            fixed_size = (our_min + our_max) / 2
            logger.info(f"Фиксированный размер (среднее): {fixed_size:.2f} USDT")

        except Exception as e:
            logger.error(f"Ошибка при тестировании {pair}: {e}")


if __name__ == "__main__":
    test_position_size_calculation()
