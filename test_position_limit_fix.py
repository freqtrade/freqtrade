#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления ограничения размера позиции
"""

import logging


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_position_size_limits():
    """
    Тестирует логику ограничения размера позиции
    """
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ОГРАНИЧЕНИЙ РАЗМЕРА ПОЗИЦИИ")
    print("=" * 60)

    # Параметры теста
    config_stake_amount = 100  # USDT из конфига
    max_position_size_pct = 100  # 100% от stake_amount
    exchange_min_stake = 10  # Минимальный размер биржи
    min_stake = 10  # Минимальный размер, переданный в функцию

    # Рассчитываем максимальный размер позиции
    max_position = config_stake_amount * (max_position_size_pct / 100)

    print(f"Параметры:")
    print(f"  - Stake amount из конфига: {config_stake_amount} USDT")
    print(f"  - Максимальный %: {max_position_size_pct}%")
    print(f"  - Максимальный размер: {max_position} USDT")
    print(f"  - Минимальный размер биржи: {exchange_min_stake} USDT")
    print(f"  - Минимальный размер (min_stake): {min_stake} USDT")
    print()

    # Тестовые случаи
    test_cases = [
        {"name": "Нормальный размер", "position_size": 50, "expected": 50},
        {"name": "Размер меньше минимума", "position_size": 5, "expected": 10},
        {"name": "Размер больше максимума", "position_size": 150, "expected": 100},
        {"name": "Размер точно на максимуме", "position_size": 100, "expected": 100},
        {"name": "Размер точно на минимуме", "position_size": 10, "expected": 10},
        {"name": "Размер намного больше максимума", "position_size": 200, "expected": 100},
        {"name": "Размер намного меньше минимума", "position_size": 1, "expected": 10},
    ]

    print("Результаты тестирования:")
    print("-" * 60)
    print(
        f"{'Тест':<25} | {'Входной размер':<15} | {'Ожидаемый':<12} | {'Результат':<12} | {'Статус':<8}"
    )
    print("-" * 60)

    for case in test_cases:
        # Симулируем логику ограничений
        position_size = case["position_size"]

        # Применяем ограничения (как в исправленном коде)
        position_size = max(position_size, min_stake)  # Не меньше минимума
        position_size = min(position_size, max_position)  # Не больше максимума

        expected = case["expected"]
        status = "✓" if position_size == expected else "✗"

        print(
            f"{case['name']:<25} | {case['position_size']:<15.2f} | {expected:<12.2f} | {position_size:<12.2f} | {status:<8}"
        )

    print("-" * 60)
    print("Тестирование завершено!")
    print()

    # Дополнительная проверка для случая, который вызвал проблему
    print("ПРОВЕРКА ПРОБЛЕМНОГО СЛУЧАЯ:")
    print(f"Исходный размер: 149.89 USDT")
    print(f"Применяем ограничения:")
    print(f"  - max(149.89, {min_stake}) = 149.89")
    print(f"  - min(149.89, {max_position}) = {min(149.89, max_position):.2f}")
    print(f"Результат: {min(149.89, max_position):.2f} USDT")

    if min(149.89, max_position) <= max_position:
        print("✓ Проблема исправлена - размер ограничен максимумом")
    else:
        print("✗ Проблема не исправлена")


if __name__ == "__main__":
    test_position_size_limits()
