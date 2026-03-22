#!/usr/bin/env python3
"""
Тестовый скрипт для проверки формулы расчета размера позиции на основе ATR
"""


def test_atr_position_calculation():
    """
    Тестирует новую формулу расчета размера позиции на основе ATR
    """

    # Тестовые данные
    balance = 1000  # USDT
    risk_per_trade = 0.03  # 3%
    current_rate = 50000  # USDT за BTC
    atr_position_multiplier = 2.0
    min_position = 10  # Биржевой минимум
    max_position = 100  # stake_amount из конфига

    # Различные значения ATR для тестирования
    test_cases = [
        {"atr": 100, "description": "Низкая волатильность (0.2%)"},
        {"atr": 250, "description": "Средняя волатильность (0.5%)"},
        {"atr": 500, "description": "Высокая волатильность (1.0%)"},
        {"atr": 1000, "description": "Очень высокая волатильность (2.0%)"},
    ]

    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ФОРМУЛЫ РАСЧЕТА РАЗМЕРА ПОЗИЦИИ НА ОСНОВЕ ATR")
    print("=" * 80)
    print(f"Баланс: {balance} USDT")
    print(f"Риск на сделку: {risk_per_trade * 100}%")
    print(f"Текущая цена: {current_rate} USDT")
    print(f"ATR множитель: {atr_position_multiplier}")
    print()

    for i, case in enumerate(test_cases, 1):
        atr = case["atr"]
        description = case["description"]

        # Рассчитываем ATR как процент от цены
        atr_pct = (atr / current_rate) * 100

        # Рассчитываем фактор волатильности
        volatility_factor = atr_pct * atr_position_multiplier

        # Рассчитываем размер позиции
        risk_amount = balance * risk_per_trade

        if volatility_factor > 0:
            atr_based_size = risk_amount / (volatility_factor / 100)
        else:
            atr_based_size = (
                min_position + max_position
            ) / 2  # Фиксированный размер при низкой волатильности

        # Применяем ограничения
        if atr_based_size > max_position:
            position_size = max_position
        elif atr_based_size < min_position:
            position_size = min_position
        else:
            position_size = atr_based_size

        print(f"Тест {i}: {description}")
        print(f"  ATR: {atr} USDT ({atr_pct:.2f}%)")
        print(f"  Фактор волатильности: {volatility_factor:.2f}%")
        print(f"  ATR-based size: {atr_based_size:.2f} USDT")
        print(f"  Final position size: {position_size:.2f} USDT")
        print(f"  Range: {min_position}-{max_position} USDT")
        print()

    print("=" * 80)
    print("ВЫВОДЫ:")
    print("- При низкой волатильности размер позиции увеличивается")
    print("- При высокой волатильности размер позиции уменьшается")
    print("- Это помогает управлять риском в зависимости от рыночных условий")
    print("=" * 80)


if __name__ == "__main__":
    test_atr_position_calculation()
