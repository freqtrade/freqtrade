from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.persistence import Trade
from datetime import datetime, timezone
import logging
import os
import talib.abstract as ta

# ==================================================
# НАЧАЛЬНЫЕ ЗНАЧЕНИЯ ПАРАМЕТРОВ СТРАТЕГИИ
# ==================================================

# Базовые параметры стратегии
STRATEGY_PARAMS = {
    "timeframe": "5m",
    "order_book_depth": 50,
    "min_cluster_size": 0.1,
    "cluster_price_threshold": 0.0005,
    "min_orders_in_cluster": 2,
    "min_volume_ratio": 1.5,  # Используется для проверки volume_ratio в условиях входа/выхода
    "volume_threshold": 2.0,  # Используется для фильтрации кластеров по объему
    "cluster_price_distance": 0.001,  # Максимальное расстояние до кластера (0.1%)
    "stoploss": -0.02,
    "minimal_roi": {
        "0": 0.05,  # 5% прибыли сразу
        "30": 0.025,  # 2.5% после 30 минут
        "60": 0.015,  # 1.5% после 1 часа
        "120": 0.01,  # 1% после 2 часов
    },
    # Фильтры для входа и выхода
    # RSI фильтр для входа и выхода
    "rsi_filter_enabled": True,
    "rsi_period": 14,  # было 14
    "rsi_buy_threshold": 35,  # было 35
    "rsi_sell_threshold": 75,
    # EMA фильтр тренда
    "ema_filter_enabled": True,
    "ema_slow_period": 200,  # было 200
    # ATR фильтр волатильности
    "atr_filter_enabled": True,
    "atr_period": 14,
    "atr_min_volatility_pct": 0.3,  # было 0.5 Минимальная волатильность в % от цены
    # ATR для Stop-Loss и Take-Profit
    "atr_stoploss_enabled": True,
    "atr_stoploss_multiplier": 2.0,  # Множитель ATR для стоп-лосса
    "atr_takeprofit_enabled": True,
    "atr_takeprofit_multiplier": 3.0,  # Множитель ATR для тейк-профита
    "atr_takeprofit_min_pct": 1.0,  # Минимальный % прибыли для ATR тейк-профита
}

# ==================================================
# КОНЕЦ НАЧАЛЬНЫХ ЗНАЧЕНИЙ ПАРАМЕТРОВ
# ==================================================

# ==================================================
# НАСТРОЙКИ ДЛЯ КАЖДОЙ ТОРГОВОЙ ПАРЫ
# ==================================================

# Базовые настройки для всех пар
DEFAULT_SETTINGS = {
    "timeframe": STRATEGY_PARAMS["timeframe"],
    "order_book_depth": STRATEGY_PARAMS["order_book_depth"],
    "min_cluster_size": STRATEGY_PARAMS["min_cluster_size"],
    "cluster_price_threshold": STRATEGY_PARAMS["cluster_price_threshold"],
    "min_orders_in_cluster": STRATEGY_PARAMS["min_orders_in_cluster"],
    "min_volume_ratio": STRATEGY_PARAMS["min_volume_ratio"],
    "volume_threshold": STRATEGY_PARAMS["volume_threshold"],
    "stoploss": STRATEGY_PARAMS["stoploss"],
    "minimal_roi": STRATEGY_PARAMS["minimal_roi"],
    "rsi_filter_enabled": STRATEGY_PARAMS["rsi_filter_enabled"],
    "rsi_period": STRATEGY_PARAMS["rsi_period"],
    "rsi_buy_threshold": STRATEGY_PARAMS["rsi_buy_threshold"],
    "rsi_sell_threshold": STRATEGY_PARAMS["rsi_sell_threshold"],
    "ema_filter_enabled": STRATEGY_PARAMS["ema_filter_enabled"],
    "ema_slow_period": STRATEGY_PARAMS["ema_slow_period"],
    "atr_filter_enabled": STRATEGY_PARAMS["atr_filter_enabled"],
    "atr_period": STRATEGY_PARAMS["atr_period"],
    "atr_min_volatility_pct": STRATEGY_PARAMS["atr_min_volatility_pct"],
    "atr_stoploss_enabled": STRATEGY_PARAMS["atr_stoploss_enabled"],
    "atr_stoploss_multiplier": STRATEGY_PARAMS["atr_stoploss_multiplier"],
    "atr_takeprofit_enabled": STRATEGY_PARAMS["atr_takeprofit_enabled"],
    "atr_takeprofit_multiplier": STRATEGY_PARAMS["atr_takeprofit_multiplier"],
    "atr_takeprofit_min_pct": STRATEGY_PARAMS["atr_takeprofit_min_pct"],
}

# Настройки для конкретных пар
PAIR_SETTINGS = {
    "BTC/USDT": {
        "cluster_price_threshold": 0.0005,  # 0.05%
        "min_cluster_size": 0.05,
    },
    "ETH/USDT": {
        "cluster_price_threshold": 0.0008,  # 0.08%
        "min_cluster_size": 0.08,
    },
    # Добавьте здесь настройки для других пар
}


def get_pair_settings(pair: str) -> Dict[str, Any]:
    """Возвращает настройки для указанной пары"""
    settings = DEFAULT_SETTINGS.copy()
    if pair in PAIR_SETTINGS:
        settings.update(PAIR_SETTINGS[pair])
    return settings


# ==================================================
# КОНЕЦ НАСТРОЕК ДЛЯ КАЖДОЙ ПАРЫ
# ==================================================

# Настройка логирования
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# Настройка корневого логгера
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Удаляем все существующие обработчики
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Файловый обработчик (без цветов)
file_handler = logging.FileHandler(
    os.path.join(log_dir, "orderbook_strategy.log"), mode="a", encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)


# Кастомный фильтр для логирования
class LogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_warning = {}
        self.warning_interval = 10  # seconds

    def filter(self, record):
        import re
        from time import time

        # Пропускаем логи API
        if hasattr(record, "args") and record.args and "api/v1" in str(record.args):
            return False

        # Удаляем дублирующиеся предупреждения
        if record.levelno >= logging.WARNING:
            warning_key = (record.msg, record.args)
            current_time = time()

            if warning_key in self.last_warning:
                last_time = self.last_warning[warning_key]
                if current_time - last_time < self.warning_interval:
                    return False

            self.last_warning[warning_key] = current_time

            # Очищаем старые записи
            self.last_warning = {
                k: v
                for k, v in self.last_warning.items()
                if current_time - v < 3600  # Храним историю 1 час
            }

        # Удаляем ANSI коды
        ansi_escape = re.compile(r"\x1b\[[0-9;]*[mGKH]")

        def clean_text(text):
            if isinstance(text, str):
                text = ansi_escape.sub("", text)  # Удаляем ANSI коды
                text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)  # Удаляем управляющие символы
            return text

        # Обрабатываем сообщение
        if record.msg:
            if isinstance(record.msg, str):
                record.msg = clean_text(record.msg)
            else:
                record.msg = clean_text(str(record.msg))

        # Обрабатываем аргументы, если они есть
        if record.args:
            if isinstance(record.args, (list, tuple)):
                new_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        new_args.append(clean_text(arg))
                    elif isinstance(arg, dict):
                        new_args.append({k: clean_text(str(v)) for k, v in arg.items()})
                    else:
                        new_args.append(arg)
                record.args = tuple(new_args)
            elif isinstance(record.args, dict):
                # Если args - это словарь, преобразуем его в кортеж для форматирования
                record.args = (record.args,)

        # Пытаемся отформатировать сообщение, если есть аргументы
        if record.args and isinstance(record.msg, str) and "%" in record.msg:
            try:
                record.msg = record.msg % record.args
                record.args = None
            except (TypeError, ValueError):
                # Если не удалось отформатировать, оставляем как есть
                pass

        return True


# Создаем и настраиваем фильтр
log_filter = LogFilter()

# Настраиваем файловый обработчик
file_handler.addFilter(log_filter)
logger.addHandler(file_handler)

# Настраиваем консольный обработчик
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
console_handler.addFilter(log_filter)  # Применяем тот же фильтр
logger.addHandler(console_handler)

# Отключаем логирование для API и других шумных модулей
for module in ["freqtrade.rpc.api_server", "freqtrade.rpc.telegram", "urllib3"]:
    logging.getLogger(module).setLevel(logging.ERROR)


class OrderBookDepthStrategy(IStrategy):
    """
    Стратегия на основе анализа глубины стакана.

    Основные принципы работы:
    1. Анализ глубины стакана для поиска кластеров ликвидности
    2. Размещение ордеров рядом с этими кластерами
    3. Использование анализа объема для подтверждения силы уровней
    """

    # Параметры стратегии
    timeframe = STRATEGY_PARAMS["timeframe"]
    order_book_depth = IntParameter(
        20,
        100,
        default=STRATEGY_PARAMS["order_book_depth"],
        space="buy",
        description="Глубина анализа стакана",
    )
    min_cluster_size = DecimalParameter(
        0.01,
        0.5,
        default=STRATEGY_PARAMS["min_cluster_size"],
        space="buy",
        description="Минимальный размер кластера ликвидности",
    )
    cluster_price_threshold = DecimalParameter(
        0.0001,
        0.01,
        default=STRATEGY_PARAMS["cluster_price_threshold"],
        space="buy",
        description="Порог цены для объединения кластеров (0.01% - 1%)",
    )
    min_orders_in_cluster = IntParameter(
        2,
        10,
        default=STRATEGY_PARAMS["min_orders_in_cluster"],
        space="buy",
        description="Минимальное количество ордеров в кластере",
    )

    # Параметры анализа объема
    min_volume_ratio = DecimalParameter(
        0.5,
        2.0,
        default=STRATEGY_PARAMS["min_volume_ratio"],
        space="buy",
        description="Минимальное соотношение объемов покупки/продажи",
    )
    volume_threshold = DecimalParameter(
        1.0,
        5.0,
        default=STRATEGY_PARAMS["volume_threshold"],
        space="buy",
        description="Порог объема для фильтрации кластеров",
    )
    cluster_price_distance = DecimalParameter(
        0.0001,
        0.01,
        default=STRATEGY_PARAMS["cluster_price_distance"],
        space="buy",
        description="Максимальное расстояние до кластера (0.01% - 1%)",
    )

    # Параметры управления рисками
    stoploss = STRATEGY_PARAMS["stoploss"]
    minimal_roi = STRATEGY_PARAMS["minimal_roi"]

    # Фильтры
    rsi_filter_enabled = STRATEGY_PARAMS["rsi_filter_enabled"]
    rsi_period = STRATEGY_PARAMS["rsi_period"]
    rsi_buy_threshold = STRATEGY_PARAMS["rsi_buy_threshold"]
    rsi_sell_threshold = STRATEGY_PARAMS["rsi_sell_threshold"]
    ema_filter_enabled = STRATEGY_PARAMS["ema_filter_enabled"]
    ema_slow_period = STRATEGY_PARAMS["ema_slow_period"]
    atr_filter_enabled = STRATEGY_PARAMS["atr_filter_enabled"]
    atr_period = STRATEGY_PARAMS["atr_period"]
    atr_min_volatility_pct = STRATEGY_PARAMS["atr_min_volatility_pct"]
    atr_stoploss_enabled = STRATEGY_PARAMS["atr_stoploss_enabled"]
    atr_stoploss_multiplier = STRATEGY_PARAMS["atr_stoploss_multiplier"]
    atr_takeprofit_enabled = STRATEGY_PARAMS["atr_takeprofit_enabled"]
    atr_takeprofit_multiplier = STRATEGY_PARAMS["atr_takeprofit_multiplier"]
    atr_takeprofit_min_pct = STRATEGY_PARAMS["atr_takeprofit_min_pct"]

    def _get_param_value(self, param_name):
        """
        Безопасное получение значения параметра по имени

        Args:
            param_name: Имя параметра (str) или числовое значение

        Returns:
            Значение параметра или переданное значение, если это не имя параметра
        """
        # Если передан не строковый параметр, возвращаем как есть
        if not isinstance(param_name, str):
            return param_name

        # Получаем атрибут стратегии по имени
        param = getattr(self, param_name, None)

        # Если параметр не найден, возвращаем None
        if param is None:
            return None

        # Если параметр - это объект Parameter, получаем его значение
        if hasattr(param, "value"):
            return param.value

        # Если параметр - это словарь (как minimal_roi), возвращаем как есть
        if isinstance(param, dict):
            return param

        # Если это числовое значение, возвращаем как есть
        if isinstance(param, (int, float)):
            return param

        # Пытаемся преобразовать в число, если это строка с числом
        if isinstance(param, str):
            # Проверяем на целое число
            if param.isdigit():
                return int(param)
            # Проверяем на число с плавающей точкой
            try:
                return float(param)
            except (ValueError, TypeError):
                pass

        # Возвращаем как есть (строка или другой тип)
        return param

    def verify_pair_whitelist(self):
        """Проверяет и логирует список доступных для торговли пар"""
        try:
            if hasattr(self, "dp") and hasattr(self.dp, "current_whitelist"):
                whitelist = self.dp.current_whitelist()
                if whitelist:
                    self.logger.info(f"Доступные пары в whitelist: {whitelist}")
                    return True
                else:
                    self.logger.warning("Whitelist пар пуст. Проверьте настройки конфигурации.")
            else:
                self.logger.warning(
                    "Не удалось получить доступ к списку пар. Возможно, данные еще не загружены."
                )
        except Exception as e:
            self.logger.error(f"Ошибка при проверке whitelist: {str(e)}")
        return False

    def __init__(self, config: dict) -> None:
        """
        Инициализация стратегии с настройками для конкретной пары
        """
        # Инициализируем родительский класс в первую очередь
        super().__init__(config)

        # Получаем настройки для пары (если указана)
        pair = config.get("pair", "")
        self.pair_settings = get_pair_settings(pair)

        # Устанавливаем атрибуты стратегии из настроек
        for key, value in self.pair_settings.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Инициализация логгера
        self.logger = logging.getLogger("OrderBookDepth")
        self.logger.debug(f"Инициализация стратегии OrderBookDepth для пары {pair}")
        self.logger.debug(f"Настройки: {self.pair_settings}")

        # Проверяем whitelist пар
        self.verify_pair_whitelist()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавление индикаторов для анализа
        """
        try:
            # Получаем данные стакана
            pair = metadata.get("pair")
            if not pair:
                return dataframe

            exchange = self.dp._exchange
            if not exchange:
                logger.warning(f"Exchange not available for {pair}")
                return dataframe

            # Получаем стакан с заданной глубиной
            order_book_depth = self._get_param_value("order_book_depth")
            if not isinstance(order_book_depth, int):
                order_book_depth = int(order_book_depth)  # Ensure it's an integer
            orderbook = exchange.fetch_l2_order_book(pair, order_book_depth)

            # Логируем информацию о стакане
            logger.debug(f"Order book depth: {order_book_depth}")
            logger.debug(f"Bids count: {len(orderbook['bids']) if 'bids' in orderbook else 0}")
            logger.debug(f"Asks count: {len(orderbook['asks']) if 'asks' in orderbook else 0}")

            # Анализируем кластеры ликвидности
            logger.debug("Analyzing buy clusters...")
            buy_clusters = self._find_liquidity_clusters(orderbook["bids"], "bids")
            logger.debug(f"Buy clusters result: {buy_clusters}")

            logger.debug("Analyzing sell clusters...")
            sell_clusters = self._find_liquidity_clusters(orderbook["asks"], "asks")
            logger.debug(f"Sell clusters result: {sell_clusters}")

            # Берем самый большой кластер (первый в отсортированном списке)
            buy_cluster = (
                buy_clusters[0]
                if buy_clusters
                else {"price": 0, "volume": 0, "count": 0, "min_spread": 0, "max_spread": 0}
            )
            sell_cluster = (
                sell_clusters[0]
                if sell_clusters
                else {"price": 0, "volume": 0, "count": 0, "min_spread": 0, "max_spread": 0}
            )

            # Добавляем информацию о кластерах в dataframe
            dataframe["buy_cluster_price"] = buy_cluster["price"]
            dataframe["buy_cluster_volume"] = buy_cluster["volume"]
            dataframe["buy_cluster_orders"] = buy_cluster["count"]  # Добавляем количество ордеров
            dataframe["sell_cluster_price"] = sell_cluster["price"]
            dataframe["sell_cluster_volume"] = sell_cluster["volume"]
            dataframe["sell_cluster_orders"] = sell_cluster["count"]  # Добавляем количество ордеров

            # --- Расчет индикаторов для фильтров ---
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self._get_param_value("rsi_period"))
            dataframe["ema_slow"] = ta.EMA(
                dataframe, timeperiod=self._get_param_value("ema_slow_period")
            )
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=self._get_param_value("atr_period"))

            # Анализ объемов
            dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()

            # Логируем сигналы и информацию о стакане
            self._log_signals(
                pair,
                dataframe,
                {
                    "buy_cluster": buy_cluster,
                    "sell_cluster": sell_cluster,
                    "buy_clusters": buy_clusters,
                    "sell_clusters": sell_clusters,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

            return dataframe

        except Exception as e:
            logger.error(f"Ошибка в populate_indicators для {metadata.get('pair')}: {e}")
            return dataframe

    def _log_signals(self, pair: str, dataframe: DataFrame, ob_analysis: dict) -> None:
        """
        Логирование торговых сигналов и информации о стакане
        """
        try:
            last_row = dataframe.iloc[-1]
            current_price = last_row["close"]

            # Проверяем наличие открытой позиции и выполнение ROI
            has_position = self._has_open_position(pair)
            roi_condition_met = has_position and self._check_roi_condition(pair, current_price)

            # Получаем информацию о кластерах из анализа стакана
            buy_cluster = ob_analysis.get("buy_cluster", {})
            sell_cluster = ob_analysis.get("sell_cluster", {})

            # Получаем объемы для сравнения
            buy_volume = last_row.get("buy_cluster_volume", 0)
            sell_volume = last_row.get("sell_cluster_volume", 0)

            # Создаем краткое сообщение о состоянии
            status_msg = [
                f"[{pair}] Цена: {current_price:.2f} | Позиция: {'ДА' if has_position else 'НЕТ'} | "
                f"Покупатели: {buy_volume:.1f} | Продавцы: {sell_volume:.1f}"
            ]

            # Логируем только краткую информацию
            self.logger.info("".join(status_msg))

            # Возвращаем результаты для использования в других методах
            return {
                "buy_signal": False,  # Будет определено в populate_entry_trend
                "sell_signal": False,  # Будет определено в populate_exit_trend
                "has_position": has_position,
                "roi_ok": roi_condition_met,
            }

        except Exception as e:
            self.logger.error(f"Ошибка при логировании сигналов: {str(e)}", exc_info=True)
            return {
                "buy_signal": False,
                "sell_signal": False,
                "has_position": False,
                "roi_ok": False,
            }

    def _find_liquidity_clusters(
        self, order_book_side: List[List[float]], side: str = "bids"
    ) -> List[Dict]:
        """
        Находит кластеры ликвидности в стакане.

        Args:
            order_book_side: Список списков [цена, объем] для стакана (биды или аски)
            side: Сторона стакана ('bids' для покупок, 'asks' для продаж)

        Returns:
            Список словарей с информацией о найденных кластерах, отсортированный по объему (по убыванию)
        """
        if not order_book_side:
            self.logger.debug("Пустой стакан")
            return []

        # Сортируем стакан в зависимости от стороны
        if side == "bids":
            # Для бидов сортируем по убыванию цены
            order_book_side = sorted(order_book_side, key=lambda x: x[0], reverse=True)
        else:
            # Для асков сортируем по возрастанию цены
            order_book_side = sorted(order_book_side, key=lambda x: x[0])

        # Получаем значения параметров
        cluster_price_threshold = self._get_param_value("cluster_price_threshold")
        min_cluster_size = self._get_param_value("min_cluster_size")
        min_orders_in_cluster = self._get_param_value("min_orders_in_cluster")

        # Увеличиваем порог цены для лучшего разделения кластеров
        cluster_price_threshold = max(0.001, cluster_price_threshold)  # Минимум 0.1%

        self.logger.debug(f"\n{'=' * 50}")
        self.logger.debug(f"АНАЛИЗ {side.upper()}")
        self.logger.debug(f"Всего ордеров: {len(order_book_side)}")
        self.logger.debug(f"Параметры кластеризации:")
        self.logger.debug(f"  - Порог цены: {cluster_price_threshold * 100:.4f}%")
        self.logger.debug(f"  - Мин. размер кластера: {min_cluster_size}")
        self.logger.debug(f"  - Мин. ордеров в кластере: {min_orders_in_cluster}")

        # Анализ распределения цен
        prices = [price for price, _ in order_book_side]
        price_diff_pcts = []
        for i in range(1, len(prices)):
            diff_pct = abs(prices[i] - prices[i - 1]) / prices[i - 1] * 100
            price_diff_pcts.append(diff_pct)

        if price_diff_pcts:
            avg_diff = sum(price_diff_pcts) / len(price_diff_pcts)
            max_diff = max(price_diff_pcts)
            min_diff = min(price_diff_pcts)
            self.logger.debug(f"\nАнализ распределения цен:")
            self.logger.debug(f"  - Среднее изменение: {avg_diff:.6f}%")
            self.logger.debug(f"  - Макс. изменение: {max_diff:.6f}%")
            self.logger.debug(f"  - Мин. изменение: {min_diff:.6f}%")

            # Автоматическая настройка порога, если среднее изменение слишком мало
            if avg_diff < cluster_price_threshold * 0.5:
                new_threshold = avg_diff * 2
                self.logger.warning(
                    f"Среднее изменение цены {avg_diff:.6f}% слишком мало, увеличиваю порог до {new_threshold:.6f}%"
                )
                cluster_price_threshold = new_threshold / 100  # Конвертируем обратно в долю

        # Логируем первые 15 ордеров после сортировки
        self.logger.debug(f"\n=== ПЕРВЫЕ 15 ОРДЕРОВ {side.upper()} ===")
        self.logger.debug(
            f"{'#':>3} | {'Цена':>15} | {'Объем':>15} | {'Δ% от пред.':>12} | {'ΣΔ% от 1-го':>12}"
        )
        self.logger.debug("-" * 70)

        for i in range(min(15, len(order_book_side))):
            price, volume = order_book_side[i]
            if i == 0:
                price_diff_pct = 0.0
                total_diff_pct = 0.0
            else:
                prev_price = order_book_side[i - 1][0]
                price_diff = abs(price - prev_price)
                price_diff_pct = (price_diff / prev_price) * 100
                total_diff_pct = abs((price - order_book_side[0][0]) / order_book_side[0][0] * 100)

            self.logger.debug(
                f"{i + 1:>3} | {price:>15.8f} | {volume:>15.8f} | {price_diff_pct:>10.4f}% | {total_diff_pct:>10.4f}%"
            )

        if len(order_book_side) > 15:
            self.logger.debug(f"... и еще {len(order_book_side) - 15} ордеров")
        self.logger.debug("=" * 70 + "\n")

        clusters = []
        current_cluster = None

        for price, volume in order_book_side:
            if current_cluster is None:
                # Создаем новый кластер
                current_cluster = {
                    "price": price,
                    "volume": volume,
                    "count": 1,
                    "min_price": price,
                    "max_price": price,
                    "orders": [(price, volume)],
                }
            else:
                # Проверяем, входит ли ордер в текущий кластер
                price_diff = abs(price - current_cluster["price"]) / current_cluster["price"]
                if price_diff <= cluster_price_threshold:
                    # Добавляем ордер в текущий кластер
                    current_cluster["volume"] += volume
                    current_cluster["count"] += 1
                    current_cluster["min_price"] = min(current_cluster["min_price"], price)
                    current_cluster["max_price"] = max(current_cluster["max_price"], price)
                    current_cluster["orders"].append((price, volume))

                    # Пересчитываем средневзвешенную цену
                    total_volume = sum(v for _, v in current_cluster["orders"])
                    weighted_sum = sum(p * v for p, v in current_cluster["orders"])
                    current_cluster["price"] = (
                        weighted_sum / total_volume if total_volume > 0 else price
                    )
                else:
                    # Сохраняем текущий кластер, если он соответствует критериям
                    if (
                        current_cluster["count"] >= min_orders_in_cluster
                        and current_cluster["volume"] >= min_cluster_size
                    ):
                        # Рассчитываем спред цены в кластере
                        cluster_data = {
                            "price": current_cluster["price"],
                            "volume": current_cluster["volume"],
                            "count": current_cluster["count"],
                            "min_spread": current_cluster["min_price"],
                            "max_spread": current_cluster["max_price"],
                        }
                        clusters.append(cluster_data)

                    # Создаем новый кластер
                    current_cluster = {
                        "price": price,
                        "volume": volume,
                        "count": 1,
                        "min_price": price,
                        "max_price": price,
                        "orders": [(price, volume)],
                    }

        # Проверяем и добавляем последний кластер
        if (
            current_cluster
            and current_cluster["count"] >= self._get_param_value("min_orders_in_cluster")
            and current_cluster["volume"] >= self._get_param_value("min_cluster_size")
        ):
            cluster_data = {
                "price": current_cluster["price"],
                "volume": current_cluster["volume"],
                "count": current_cluster["count"],
                "min_spread": current_cluster["min_price"],
                "max_spread": current_cluster["max_price"],
            }
            clusters.append(cluster_data)

        # Сортируем кластеры по объему (по убыванию)
        clusters.sort(key=lambda x: x["volume"], reverse=True)

        # Логируем подробную информацию о найденных кластерах
        if clusters:
            self.logger.debug(f"Найдено кластеров {side}: {len(clusters)}")

            # Показываем только лучший кластер
            best_cluster = clusters[0]
            self.logger.debug(
                f"Лучший кластер: цена={best_cluster['price']:.8f}, объем={best_cluster['volume']:.4f}, ордеров={best_cluster['count']}"
            )

            # Анализ распределения объемов
            if len(clusters) > 1:
                volumes = [c["volume"] for c in clusters]
                total_volume = sum(volumes)
                avg_volume = total_volume / len(volumes)
                volume_std = (sum((v - avg_volume) ** 2 for v in volumes) / len(volumes)) ** 0.5

                # Проверяем, не слишком ли сильно отличаются объемы
                if volume_std > avg_volume * 0.5:  # Если отклонение больше 50% от среднего
                    self.logger.warning(
                        f"Высокое отклонение объемов между кластерами {side}: {volume_std:.4f} > {avg_volume * 0.5:.4f}"
                    )
        else:
            self.logger.warning(f"Не найдено кластеров для {side}")

        # Добавляем информацию о текущей цене для контекста
        if hasattr(self, "dp") and hasattr(self.dp, "_exchange"):
            try:
                # Получаем текущую цену для сравнения
                current_price = self.dp._exchange.fetch_ticker(side.split("_")[0] + "/USDT")["last"]

                if clusters:
                    best_cluster = clusters[0]  # Самый большой кластер
                    price_diff = abs(current_price - best_cluster["price"]) / current_price * 100

                    if price_diff > 1.0:  # Если расстояние больше 1%
                        self.logger.warning(
                            f"Большое расстояние между ценой и кластером {side}: {price_diff:.3f}%"
                        )
            except Exception as e:
                self.logger.debug(f"Не удалось получить текущую цену: {e}")

        self.logger.debug("=" * 50 + "\n")

        return clusters

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определение сигналов на вход в позицию на основе анализа кластера покупок
        """
        try:
            pair = metadata.get("pair")
            if not pair or len(dataframe) == 0:
                return dataframe

            # Получаем последнюю строку данных
            last_row = dataframe.iloc[-1]

            # Получаем параметры стратегии
            min_volume = self._get_param_value("min_cluster_size")
            min_orders = self._get_param_value("min_orders_in_cluster")

            # Получаем данные о кластере покупок
            buy_cluster_price = last_row.get("buy_cluster_price", 0)
            buy_cluster_volume = last_row.get("buy_cluster_volume", 0)
            buy_cluster_orders = last_row.get(
                "buy_cluster_orders", 0
            )  # Количество ордеров в кластере
            current_price = last_row["close"]

            # Получаем объемы для сравнения покупателей и продавцов
            buy_volume = last_row.get("buy_cluster_volume", 0)  # Используем объем кластера покупок
            sell_volume = last_row.get("sell_cluster_volume", 0)  # Используем объем кластера продаж

            # Проверяем условия для входа в позицию
            has_position = self._has_open_position(pair)

            # Определяем, находится ли цена рядом с кластером покупок (в пределах 0.1%)
            price_distance = (
                abs(current_price - buy_cluster_price) / buy_cluster_price
                if buy_cluster_price > 0
                else 1.0
            )
            cluster_distance_threshold = self._get_param_value("cluster_price_distance")

            # Проверяем, что получено числовое значение
            if (
                not isinstance(cluster_distance_threshold, (int, float))
                or cluster_distance_threshold is None
            ):
                self.logger.error(
                    f"Некорректное значение cluster_price_distance: {cluster_distance_threshold}"
                )
                cluster_distance_threshold = STRATEGY_PARAMS[
                    "cluster_price_distance"
                ]  # Значение по умолчанию

            price_near_cluster = price_distance <= cluster_distance_threshold

            # Основные условия для покупки
            conditions = {
                "volume_ok": buy_cluster_volume >= min_volume,
                "orders_ok": buy_cluster_orders >= min_orders,  # Проверяем количество ордеров
                "price_near_cluster": price_near_cluster,  # Цена рядом с кластером покупок
                "volume_ratio_ok": last_row.get("volume_ratio", 0)
                >= self._get_param_value("min_volume_ratio"),
                "buyers_stronger": buy_volume > sell_volume,
                "no_position": not has_position,
            }

            # Формируем сигнал на покупку
            cluster_enter_long = all(conditions.values())
            enter_long = cluster_enter_long

            # --- Применение фильтров ---
            if enter_long:  # Проверяем фильтры только если есть сигнал по кластерам
                # RSI фильтр
                rsi_filter_enabled = self._get_param_value("rsi_filter_enabled")
                if rsi_filter_enabled:
                    rsi_value = last_row.get("rsi")
                    rsi_threshold = self._get_param_value("rsi_buy_threshold")
                    if rsi_value is None or rsi_value >= rsi_threshold:
                        enter_long = False

                # EMA фильтр
                ema_filter_enabled = self._get_param_value("ema_filter_enabled")
                if ema_filter_enabled:
                    ema_slow_value = last_row.get("ema_slow")
                    if ema_slow_value is None or current_price <= ema_slow_value:
                        enter_long = False

                # ATR фильтр
                atr_filter_enabled = self._get_param_value("atr_filter_enabled")
                if atr_filter_enabled:
                    atr_value = last_row.get("atr")
                    min_vol_pct = self._get_param_value("atr_min_volatility_pct")
                    if atr_value is None or (atr_value / current_price) * 100 < min_vol_pct:
                        enter_long = False

            # Логируем условия входа всегда (даже при наличии позиции)
            self.logger.info(f"\n[{pair}] УСЛОВИЯ ДЛЯ ПОКУПКИ:")
            self.logger.info(
                f"1. Объем кластера {buy_cluster_volume:.4f} ≥ {min_volume}: {'✓' if conditions['volume_ok'] else '✗'}"
            )
            self.logger.info(
                f"2. Ордеров в кластере {buy_cluster_orders} ≥ {min_orders}: {'✓' if conditions['orders_ok'] else '✗'}"
            )
            self.logger.info(
                f"3. Цена {current_price:.8f} рядом с кластером {buy_cluster_price:.8f} (расстояние {price_distance * 100:.3f}% ≤ {cluster_distance_threshold * 100:.3f}%): {'✓' if conditions['price_near_cluster'] else '✗'}"
            )
            self.logger.info(
                f"4. Объем/Средний {last_row.get('volume_ratio', 0):.2f} ≥ {self._get_param_value('min_volume_ratio')}: {'✓' if conditions['volume_ratio_ok'] else '✗'}"
            )
            self.logger.info(
                f"5. Покупатели {buy_volume:.4f} > Продавцы {sell_volume:.4f}: {'✓' if conditions['buyers_stronger'] else '✗'}"
            )
            self.logger.info(
                f"6. Нет открытой позиции: {'✓' if conditions['no_position'] else '✗'}"
            )
            self.logger.info(f"-> Сигнал по кластерам: {'ДА' if cluster_enter_long else 'НЕТ'}")

            # Логирование фильтров
            self.logger.info("--- Фильтры для входа ---")
            rsi_filter_enabled = self._get_param_value("rsi_filter_enabled")
            if rsi_filter_enabled:
                rsi_value = last_row.get("rsi", 0)
                rsi_threshold = self._get_param_value("rsi_buy_threshold")
                self.logger.info(
                    f"RSI ({rsi_value:.1f}) < {rsi_threshold}: {'✓' if rsi_value < rsi_threshold else '✗'}"
                )
            else:
                self.logger.info("RSI Фильтр: Отключен")

            ema_filter_enabled = self._get_param_value("ema_filter_enabled")
            if ema_filter_enabled:
                ema_slow_value = last_row.get("ema_slow", 0)
                self.logger.info(
                    f"EMA Trend ({current_price:.2f} > {ema_slow_value:.2f}): {'✓' if current_price > ema_slow_value else '✗'}"
                )
            else:
                self.logger.info("EMA Фильтр: Отключен")

            atr_filter_enabled = self._get_param_value("atr_filter_enabled")
            if atr_filter_enabled:
                atr_value = last_row.get("atr", 0)
                atr_pct = (atr_value / current_price) * 100 if current_price > 0 else 0
                min_vol_pct = self._get_param_value("atr_min_volatility_pct")
                self.logger.info(
                    f"ATR Volatility ({atr_pct:.2f}%) > {min_vol_pct}%: {'✓' if atr_pct > min_vol_pct else '✗'}"
                )
            else:
                self.logger.info("ATR Фильтр: Отключен")

            if not has_position:
                if enter_long:
                    self.logger.info(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: СИГНАЛ НА ПОКУПКУ! 🚀")

                    # Логирование потенциальных уровней Stop-Loss и Take-Profit
                    atr_value = last_row.get("atr", 0)
                    if self._get_param_value("atr_stoploss_enabled"):
                        sl_multiplier = self._get_param_value("atr_stoploss_multiplier")
                        stop_price = current_price - (atr_value * sl_multiplier)
                        self.logger.info(f"  - Динамический Stop-Loss (ATR): {stop_price:.2f}")

                    if self._get_param_value("atr_takeprofit_enabled"):
                        tp_multiplier = self._get_param_value("atr_takeprofit_multiplier")
                        profit_price = current_price + (atr_value * tp_multiplier)
                        self.logger.info(f"  - Динамический Take-Profit (ATR): {profit_price:.2f}")
                else:
                    reason = "условия не выполнены"
                    if cluster_enter_long and not enter_long:
                        reason = "сигнал отфильтрован"
                    self.logger.info(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: ВХОД ЗАПРЕЩЕН ({reason})")
            else:
                self.logger.info(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: ВХОД ЗАПРЕЩЕН (есть открытая позиция)")

            # Устанавливаем сигналы
            dataframe.loc[:, "enter_long"] = 0
            dataframe.loc[dataframe.index[-1], "enter_long"] = 1 if enter_long else 0

            return dataframe

        except Exception as e:
            self.logger.error(f"Ошибка в populate_entry_trend: {e}", exc_info=True)
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определение сигналов на выход из позиции
        """
        pair = metadata.get("pair")
        if not pair or len(dataframe) == 0:
            return dataframe

        try:
            # Инициализируем колонки сигналов
            if "exit_long" not in dataframe.columns:
                dataframe["exit_long"] = 0

            # Получаем последнюю свечу
            last_candle = dataframe.iloc[-1].copy()

            # Получаем значения параметров
            volume_threshold = self._get_param_value("volume_threshold")

            # Проверяем, что получено числовое значение
            if not isinstance(volume_threshold, (int, float)) or volume_threshold is None:
                logger.error(f"Некорректное значение volume_threshold: {volume_threshold}")
                volume_threshold = STRATEGY_PARAMS["volume_threshold"]  # Значение по умолчанию

            # Проверяем наличие необходимых колонок
            required_columns = [
                "sell_cluster_volume",
                "sell_cluster_price",
                "close",
                "buy_cluster_volume",
            ]
            if not all(col in dataframe.columns for col in required_columns):
                logger.warning(f"Отсутствуют необходимые колонки в данных для {pair}")
                return dataframe

            # Преобразуем значения к числовому типу
            sell_volume = pd.to_numeric(dataframe["sell_cluster_volume"], errors="coerce")
            sell_price = pd.to_numeric(dataframe["sell_cluster_price"], errors="coerce")
            close_price = pd.to_numeric(dataframe["close"], errors="coerce")
            buy_volume = pd.to_numeric(dataframe["buy_cluster_volume"], errors="coerce")

            # Проверяем условия для выхода из длинной позиции
            cluster_exit_conditions = (
                (sell_volume > volume_threshold)  # Появился крупный кластер продавцов
                & (close_price < sell_price)  # Цена ниже кластера продавцов
                & (sell_volume > buy_volume)  # Преобладание продавцов
            )
            cluster_exit_long = (
                cluster_exit_conditions.iloc[-1]
                if hasattr(cluster_exit_conditions, "iloc")
                else cluster_exit_conditions
            )
            exit_long = cluster_exit_long

            # --- Применение фильтров ---
            rsi_filter_enabled = self._get_param_value("rsi_filter_enabled")
            rsi_value = last_candle.get("rsi")
            rsi_threshold = self._get_param_value("rsi_sell_threshold")
            rsi_condition = rsi_value > rsi_threshold if rsi_value is not None else False

            if cluster_exit_long and rsi_filter_enabled and not rsi_condition:
                exit_long = False  # Фильтр не пройден, отменяем сигнал

            # Проверяем, есть ли открытая позиция
            if self._has_open_position(pair):
                current_price = float(last_candle.get("close", 0))
                roi_condition_met = self._check_roi_condition(pair, current_price)

                # Логируем условия выхода
                self.logger.info(f"[{pair}] УСЛОВИЯ ДЛЯ ПРОДАЖИ:")
                self.logger.info(
                    f"1. Объем кластера {sell_volume.iloc[-1]:.4f} ≥ {volume_threshold}: {'✓' if sell_volume.iloc[-1] > volume_threshold else '✗'}"
                )
                self.logger.info(
                    f"2. Ордеров в кластере {last_candle.get('sell_cluster_orders', 0)} ≥ {self._get_param_value('min_orders_in_cluster')}: {'✓' if last_candle.get('sell_cluster_orders', 0) >= self._get_param_value('min_orders_in_cluster') else '✗'}"
                )
                self.logger.info(
                    f"3. Цена {close_price.iloc[-1]:.8f} < {sell_price.iloc[-1]:.8f}: {'✓' if close_price.iloc[-1] < sell_price.iloc[-1] else '✗'}"
                )
                self.logger.info(
                    f"4. Продавцы {sell_volume.iloc[-1]:.4f} > Покупатели {buy_volume.iloc[-1]:.4f}: {'✓' if sell_volume.iloc[-1] > buy_volume.iloc[-1] else '✗'}"
                )
                self.logger.info(
                    f"5. Есть открытая позиция: {'✓' if self._has_open_position(pair) else '✗'}"
                )
                self.logger.info(f"6. ROI ≥ минимального: {'✓' if roi_condition_met else '✗'}")
                self.logger.info(f"-> Сигнал по кластерам: {'ДА' if cluster_exit_long else 'НЕТ'}")

                # Логирование фильтров
                self.logger.info("--- Фильтры для выхода ---")
                if rsi_filter_enabled:
                    self.logger.info(
                        f"RSI ({rsi_value:.1f}) > {rsi_threshold}: {'✓' if rsi_condition else '✗'}"
                    )
                else:
                    self.logger.info("RSI Фильтр: Отключен")

                if roi_condition_met:
                    # Устанавливаем сигнал на выход
                    dataframe.loc[dataframe.index[-1], "exit_long"] = 1 if exit_long else 0

                    # Логируем сигнал на выход, если условия выполнены для последней свечи
                    if exit_long:
                        self.logger.info(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: СИГНАЛ НА ПРОДАЖУ! 📉")
                    else:
                        reason = (
                            "ROI не достигнут" if not roi_condition_met else "условия не выполнены"
                        )
                        if cluster_exit_long and not exit_long:
                            reason = "сигнал отфильтрован"
                        self.logger.info(f"ИТОГОВЫЙ РЕЗУЛЬТАТ: ВЫХОД НЕ РЕКОМЕНДОВАН ({reason})")
                else:
                    self.logger.info(
                        f"ИТОГОВЫЙ РЕЗУЛЬТАТ: ВЫХОД НЕ РЕКОМЕНДОВАН (ROI не достигнут)"
                    )
                self.logger.info("")

            return dataframe

        except Exception as e:
            logger.error(f"Ошибка в populate_exit_trend для {pair}: {str(e)}", exc_info=True)
            return dataframe

    def _has_open_position(self, pair: str) -> bool:
        """
        Проверяет наличие открытой позиции по указанной паре
        Args:
            pair: Торговая пара (например, 'BTC/USDT')
        Returns:
            bool: True если есть открытая позиция, иначе False
        """
        try:
            trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
            return len(trades) > 0
        except Exception as e:
            logger.error(f"Ошибка при проверке открытой позиции для {pair}: {e}")
            return False

    def _check_roi_condition(self, pair: str, current_price: float) -> bool:
        """
        Проверяет, выполняется ли условие ROI для текущей позиции
        Args:
            pair: Торговая пара
            current_price: Текущая цена актива
        Returns:
            bool: True если ROI >= минимального ROI, иначе False
        """
        try:
            trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
            if not trades:
                return False

            # Берем первую открытую сделку (должна быть только одна)
            trade = trades[0]

            # Рассчитываем текущий ROI
            profit_ratio = trade.calc_profit_ratio(current_price)

            # Получаем минимальный ROI из настроек
            min_roi_values = self.minimal_roi.values()
            if not min_roi_values:
                return False
            min_roi = min(float(roi) for roi in min_roi_values)

            return profit_ratio >= min_roi

        except Exception as e:
            logger.error(f"Ошибка при проверке ROI для {pair}: {e}")
            return False

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Настройка цены входа в позицию
        """
        try:
            # Получаем текущий стакан
            exchange = self.dp._exchange
            if not exchange:
                logger.warning(f"Exchange not available for {pair}, using proposed rate")
                return proposed_rate

            orderbook = exchange.fetch_l2_order_book(pair, 1)

            if side == "long":
                # Для длинной позиции ставим ордер чуть выше кластера покупок
                return proposed_rate * 1.001  # На 0.1% выше
            else:
                # Для короткой позиции ставим ордер чуть ниже кластера продаж
                return proposed_rate * 0.999  # На 0.1% ниже

        except Exception as e:
            logger.error(f"Error in custom_entry_price for {pair}: {e}")
            return proposed_rate

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Динамический стоп-лосс на основе ATR
        """
        if not self._get_param_value("atr_stoploss_enabled"):
            return self.stoploss  # Возвращаем статический стоп-лосс из параметров

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1].copy()
        atr_value = last_candle.get("atr")

        if not atr_value:
            return self.stoploss

        multiplier = self._get_param_value("atr_stoploss_multiplier")

        # Рассчитываем стоп-лосс как процентное отклонение от цены входа
        stop_price = trade.open_rate - (atr_value * multiplier)
        stop_loss_pct = (stop_price / trade.open_rate) - 1

        return stop_loss_pct

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Динамический тейк-профит на основе ATR
        """
        if not self._get_param_value("atr_takeprofit_enabled"):
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1].copy()
        atr_value = last_candle.get("atr")

        if not atr_value:
            return None

        # 1. Рассчитываем целевую цену по ATR
        multiplier = self._get_param_value("atr_takeprofit_multiplier")
        atr_take_profit_price = trade.open_rate + (atr_value * multiplier)

        # 2. Рассчитываем целевую цену по минимальному проценту
        min_profit_pct = self._get_param_value("atr_takeprofit_min_pct") / 100
        min_profit_price = trade.open_rate * (1 + min_profit_pct)

        # 3. Выбираем большую из двух цен
        final_take_profit_price = max(atr_take_profit_price, min_profit_price)

        # 4. Проверяем, достигнута ли цена
        if current_rate >= final_take_profit_price:
            # Логируем, какая именно цель сработала
            reason = (
                "ATR"
                if final_take_profit_price == atr_take_profit_price
                else f"Min Profit ({min_profit_pct * 100}%)"
            )

            self.logger.info(
                f"[{pair}] Динамический Take-Profit ({reason}) сработал. "
                f"Цена: {current_rate:.2f} >= Цель: {final_take_profit_price:.2f}"
            )
            return f"atr_take_profit_{reason.lower().replace(' ', '_').replace('%', '')}"

        return None
