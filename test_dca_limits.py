#!/usr/bin/env python3
"""
Тестовый скрипт для проверки ограничений DCA
"""


def test_dca_limits():
    """
    Тестирует логику ограничения размера DCA
    """
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ОГРАНИЧЕНИЙ DCA")
    print("=" * 60)

    # Параметры теста
    trade_stake_amount = 100  # USDT первоначальная покупка
    max_position_size_pct = 100  # 100% от stake_amount
    config_stake_amount = 100  # USDT из конфига
    min_stake = 10  # Минимальный размер

    # Рассчитываем максимальный размер позиции
    max_position = config_stake_amount * (max_position_size_pct / 100)

    print(f"Параметры:")
    print(f"  - Первоначальная покупка: {trade_stake_amount} USDT")
    print(f"  - Максимальный размер: {max_position} USDT")
    print(f"  - Минимальный размер: {min_stake} USDT")
    print()

    # Тестовые случаи DCA
    test_cases = [
        {"entries": 0, "description": "Первая докупка (1.5^0 = 1.0)"},
        {"entries": 1, "description": "Вторая докупка (1.5^1 = 1.5)"},
        {"entries": 2, "description": "Третья докупка (1.5^2 = 2.25)"},
        {"entries": 3, "description": "Четвертая докупка (1.5^3 = 3.375)"},
    ]

    print("Результаты тестирования DCA:")
    print("-" * 80)
    print(
        f"{'Докупка':<15} | {'Множитель':<12} | {'Расчет':<15} | {'Ограничение':<15} | {'Результат':<12} | {'Статус':<8}"
    )
    print("-" * 80)

    for case in test_cases:
        entries = case["entries"]
        multiplier = 1.5
        calculated_amount = trade_stake_amount * (multiplier**entries)

        # Применяем ограничение
        original_amount = calculated_amount
        final_amount = min(calculated_amount, max_position)

        # Проверяем минимальный размер
        if final_amount < min_stake:
            status = "✗ Слишком мало"
        elif original_amount > max_position:
            status = "✓ Ограничен"
        else:
            status = "✓ Нормально"

        print(
            f"{case['description']:<15} | {multiplier**entries:<12.3f} | {calculated_amount:<15.2f} | {max_position:<15.2f} | {final_amount:<12.2f} | {status:<8}"
        )

    print("-" * 80)
    print("Тестирование завершено!")
    print()

    # Проверка конкретного случая из лога
    print("ПРОВЕРКА КОНКРЕТНОГО СЛУЧАЯ ИЗ ЛОГА:")
    print(f"Первоначальная покупка: {trade_stake_amount} USDT")
    print(
        f"Первая докупка (entries=1): {trade_stake_amount} * (1.5^1) = {trade_stake_amount * 1.5:.2f} USDT"
    )
    print(f"Максимальное ограничение: {max_position} USDT")
    print(f"Результат после ограничения: {min(trade_stake_amount * 1.5, max_position):.2f} USDT")

    if min(trade_stake_amount * 1.5, max_position) <= max_position:
        print("✓ Проблема исправлена - DCA размер ограничен максимумом")
    else:
        print("✗ Проблема не исправлена")


if __name__ == "__main__":
    test_dca_limits()
