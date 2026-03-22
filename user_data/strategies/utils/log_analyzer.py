"""
Утилита для анализа и фильтрации логов стратегии
"""

import os
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

# Путь к файлу логов по умолчанию
DEFAULT_LOG_FILE = Path("/user_data/logs/strategy.log")


def filter_logs(
    log_file: Path = None,
    pair: Optional[str] = None,
    trade_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    result: Optional[str] = None,
    min_volume: Optional[float] = None,
    min_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Фильтрация логов по заданным параметрам

    :param log_file: Путь к файлу лога
    :param pair: Фильтр по торговой паре
    :param trade_type: Filter by trade type (BUY/SELL/SIGNAL)
    :param date_from: Начальная дата (ГГГГ-ММ-ДД)
    :param date_to: Конечная дата (ГГГГ-ММ-ДД)
    :param result: Фильтр по результату (УСПЕШНО/ОТКЛОНЕНО/ОТМЕНЕНО)
    :param min_volume: Минимальный объем
    :param min_price: Минимальная цена
    :return: Список отфильтрованных записей
    """
    if log_file is None:
        log_file = DEFAULT_LOG_FILE

    if not log_file.exists():
        print(f"Файл лога {log_file} не найден")
        return []

    filtered_logs = []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if not line.strip():
                        continue

                    # Парсим строку лога
                    log_entry = {}
                    for part in line.strip().split(" | "):
                        if "=" in part:
                            key, value = part.split("=", 1)
                            log_entry[key] = value

                    # Применяем фильтры
                    if pair and log_entry.get("PAIR") != pair:
                        continue

                    if trade_type and log_entry.get("TYPE") != trade_type.upper():
                        continue

                    if result and log_entry.get("RESULT") != result.upper():
                        continue

                    # Фильтрация по дате
                    log_date = datetime.strptime(
                        log_entry.get("TIMESTAMP", ""), "%Y-%m-%d %H:%M:%S"
                    ).date()

                    if date_from:
                        from_date = datetime.strptime(date_from, "%Y-%m-%d").date()
                        if log_date < from_date:
                            continue

                    if date_to:
                        to_date = datetime.strptime(date_to, "%Y-%m-%d").date()
                        if log_date > to_date:
                            continue

                    # Фильтрация по объему и цене
                    if min_volume is not None:
                        volume = float(log_entry.get("VOLUME", 0))
                        if volume < min_volume:
                            continue

                    if min_price is not None:
                        price = float(log_entry.get("PRICE", 0))
                        if price < min_price:
                            continue

                    filtered_logs.append(log_entry)

                except Exception as e:
                    print(f"Ошибка при обработке строки лога: {line}. Ошибка: {e}")
                    continue

    except Exception as e:
        print(f"Ошибка при чтении файла лога {log_file}: {e}")

    return filtered_logs


def print_logs(logs: List[Dict[str, Any]]) -> None:
    """Красивая печать логов"""
    if not logs:
        print("Логи не найдены по заданным критериям")
        return

    # Определяем ширину колонок
    col_widths = {}
    for log in logs:
        for key, value in log.items():
            col_widths[key] = max(col_widths.get(key, 0), len(str(key)), len(str(value)) + 2)

    # Печатаем заголовок
    header = " | ".join(f"{key:<{col_widths[key]}}" for key in logs[0].keys())
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Печатаем строки
    for log in logs:
        row = " | ".join(f"{log.get(key, ''):<{col_widths[key]}}" for key in logs[0].keys())
        print(row)

    print(f"\nНайдено записей: {len(logs)}")


def main():
    """Точка входа для командной строки"""
    parser = argparse.ArgumentParser(description="Анализатор логов торговой стратегии")
    parser.add_argument("--file", type=Path, default=DEFAULT_LOG_FILE, help="Путь к файлу лога")
    parser.add_argument("--pair", type=str, help="Фильтр по торговой паре")
    parser.add_argument(
        "--type", type=str, choices=["BUY", "SELL", "SIGNAL"], help="Operation type"
    )
    parser.add_argument("--date-from", type=str, help="Начальная дата (ГГГГ-ММ-ДД)")
    parser.add_argument("--date-to", type=str, help="Конечная дата (ГГГГ-ММ-ДД)")
    parser.add_argument(
        "--result",
        type=str,
        choices=["УСПЕШНО", "ОТКЛОНЕНО", "ОТМЕНЕНО"],
        help="Результат операции",
    )
    parser.add_argument("--min-volume", type=float, help="Минимальный объем")
    parser.add_argument("--min-price", type=float, help="Минимальная цена")
    parser.add_argument(
        "--output", type=str, choices=["table", "json"], default="table", help="Формат вывода"
    )

    args = parser.parse_args()

    logs = filter_logs(
        log_file=args.file,
        pair=args.pair,
        trade_type=args.type,
        date_from=args.date_from,
        date_to=args.date_to,
        result=args.result,
        min_volume=args.min_volume,
        min_price=args.min_price,
    )

    if args.output == "json":
        print(json.dumps(logs, indent=2, ensure_ascii=False))
    else:
        print_logs(logs)


if __name__ == "__main__":
    main()
