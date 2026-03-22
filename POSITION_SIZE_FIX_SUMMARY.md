# Исправление проблемы с превышением размера позиции

## Проблема
Была совершена покупка на сумму 149.89 USDT при том, что максимальный размер позиции ограничен 100 USDT.

## Причина
В функции `custom_stake_amount` в файле `user_data/strategies/OrderBookDepth.py` отсутствовало ограничение максимального размера позиции. Код содержал только проверку минимального размера:

```python
# Дополнительная проверка: не должен быть меньше min_stake
position_size = max(position_size, min_stake)
```

Эта строка гарантировала только то, что размер позиции не будет меньше `min_stake`, но **НЕ** ограничивала максимальный размер.

## Исправление
Добавлена проверка максимального размера позиции:

```python
# Дополнительная проверка: не должен быть меньше min_stake и больше max_position
original_position_size = position_size
position_size = max(position_size, min_stake)
if original_position_size < min_stake:
    self.logger.info(f"Position size increased from {original_position_size:.2f} to {position_size:.2f} (min_stake limit)")

position_size = min(position_size, max_position)
if original_position_size > max_position:
    self.logger.info(f"Position size reduced from {original_position_size:.2f} to {position_size:.2f} (max_position limit)")
```

## Логика работы
1. **Минимальное ограничение**: `position_size = max(position_size, min_stake)` - размер не может быть меньше минимального
2. **Максимальное ограничение**: `position_size = min(position_size, max_position)` - размер не может быть больше максимального
3. **Логирование**: Добавлено логирование для отслеживания случаев, когда происходит ограничение размера

## Параметры
- `max_position_size_pct = 100` - 100% от stake_amount из конфига
- `config_stake_amount = 100` - размер позиции из конфига
- `max_position = 100 * (100/100) = 100 USDT` - максимальный размер позиции

## Результат
Теперь размер позиции будет строго ограничен диапазоном от минимального размера биржи до максимального размера (100 USDT), что предотвратит превышение установленных лимитов.

## Тестирование
Создан тестовый скрипт `test_position_limit_fix.py`, который подтверждает корректность работы исправления. 
