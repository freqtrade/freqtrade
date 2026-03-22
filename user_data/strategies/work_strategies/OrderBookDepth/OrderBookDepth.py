from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from freqtrade.exchange.exchange_types import OrderBook
from freqtrade.persistence import Trade
from datetime import datetime, timezone, timedelta
import logging
import os
import talib.abstract as ta
import ccxt
from typing import Union, Dict, List, Optional

# ==================================================
# НАЧАЛЬНЫЕ ЗНАЧЕНИЯ ПАРАМЕТРОВ СТРАТЕГИИ
# ==================================================

# Базовые параметры стратегии
STRATEGY_PARAMS = {
    "timeframe": "1m",
    "order_book_depth": 50,
    "min_cluster_size": 0.08,  # 0.05% - настраиваемо
    "cluster_price_threshold": 0.0007,  # 0.0005
    "min_orders_in_cluster": 3,  # 2 - настраиваемо
    "min_volume_ratio": 0.85,  # 0.8 - настраиваемо
    "volume_threshold": 2.5,  # 2.0 - настраиваемо
    "cluster_price_distance": 0.0015,  # 0.002 - настраиваемо
    "stoploss": -0.03,  # 3%
    # min roi в зависимости от продолжительности позиции
    # Отключаем стандартный ROI, так как используем трейлинг-стоп
    "minimal_roi_settings": {"0": 100},  # Отключаем встроенный ROI
    # Используем для расчета трейлинг-стопа
    "roi_for_trailing": {
        "0": 0.045,  # 4.5%
        "15": 0.04,  # 4%
        "30": 0.03,  # 3%
        "60": 0.02,  # 2%
        "120": 0.01,  # 1%
    },
    # Фильтры для входа/выхода
    # --- ON/OFF фильтры (раздельно для buy/sell) ---
    "rsi_buy_enabled": False,
    "rsi_sell_enabled": False,
    "ema_buy_enabled": False,
    "ema_sell_enabled": False,
    "macd_buy_enabled": True,
    "macd_sell_enabled": False,
    "bbtrend_buy_enabled": False,
    "bbtrend_sell_enabled": False,
    "atr_filter_enabled": True,  # ATR фильтр волатильности
    "atr_stoploss_enabled": False,  # ATR SL
    "atr_takeprofit_enabled": False,  # ATR TP
    "anti_pump_filter_enabled": False,  # Anti-pump фильтр
    "three_bar_growth_filter_enabled": True,  # Anti-pump фильтр
    "pullback_filter_enabled": False,  # Pullback фильтр
    # --- Параметры настройки ---
    # --- RSI  ---
    "rsi_period": 14,
    "rsi_buy_threshold": 40,  # 45
    "rsi_sell_threshold": 70,  # 70
    # --- EMA фильтр тренда ---
    "ema_slow_period": 100,  # Снижено с 200 для более быстрой реакции
    # --- ATR фильтр волатильности ---
    "atr_period": 14,
    "atr_min_volatility_pct": 0.05,  # 0.15
    # --- ATR для Stop-Loss и Take-Profit ---
    "atr_stoploss_multiplier": 2.2,  # 2.0 Множитель ATR для стоп-лосса
    "atr_takeprofit_multiplier": 2.8,  # 3.0 Множитель ATR для тейк-профита
    "atr_takeprofit_min_pct": 1.25,  # 1.0 Минимальный % прибыли для ATR тейк-профита
    # --- MACD фильтр ---
    "macd_fastperiod": 12,
    "macd_slowperiod": 26,
    "macd_signalperiod": 9,
    # --- Anti-pump фильтр ---
    "max_candle_growth": 0.003,  # 0.3%
    "max_three_bar_growth": 0.02,  # 1.5%
    # --- Pullback фильтр ---
    "min_pullback": 0.6,  # 0.7% - минимальный откат от максимума
    "pullback_lookback": 5,  # Количество свечей для поиска максимума
    # --- BBTrend ---
    "bbtrend_period": 20,
    "bbtrend_stddev": 2.0,
    "bbtrend_buy_min": 0.02,  # Пример: минимальный BBTrend для входа
    "bbtrend_sell_max": 0.01,  # Пример: максимальный BBTrend для выхода
    "bbtrend_long_period": 50,  # длинный период для BBTrend (как в TV)
}

# =================================================
# КОНЕЦ НАЧАЛЬНЫХ ЗНАЧЕНИЙ ПАРАМЕТРОВ
# =================================================

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
    "minimal_roi": STRATEGY_PARAMS["minimal_roi_settings"],
    # ON/OFF фильтры (раздельно для buy/sell)
    "rsi_buy_enabled": STRATEGY_PARAMS["rsi_buy_enabled"],
    "rsi_sell_enabled": STRATEGY_PARAMS["rsi_sell_enabled"],
    "ema_buy_enabled": STRATEGY_PARAMS["ema_buy_enabled"],
    "ema_sell_enabled": STRATEGY_PARAMS["ema_sell_enabled"],
    "macd_buy_enabled": STRATEGY_PARAMS["macd_buy_enabled"],
    "macd_sell_enabled": STRATEGY_PARAMS["macd_sell_enabled"],
    "bbtrend_buy_enabled": STRATEGY_PARAMS["bbtrend_buy_enabled"],
    "bbtrend_sell_enabled": STRATEGY_PARAMS["bbtrend_sell_enabled"],
    # RSI и EMA
    "rsi_period": STRATEGY_PARAMS["rsi_period"],
    "rsi_buy_threshold": STRATEGY_PARAMS["rsi_buy_threshold"],
    "rsi_sell_threshold": STRATEGY_PARAMS["rsi_sell_threshold"],
    "ema_slow_period": STRATEGY_PARAMS["ema_slow_period"],
    # ATR
    "atr_filter_enabled": STRATEGY_PARAMS["atr_filter_enabled"],
    "atr_period": STRATEGY_PARAMS["atr_period"],
    "atr_min_volatility_pct": STRATEGY_PARAMS["atr_min_volatility_pct"],
    "atr_stoploss_enabled": STRATEGY_PARAMS["atr_stoploss_enabled"],
    "atr_stoploss_multiplier": STRATEGY_PARAMS["atr_stoploss_multiplier"],
    "atr_takeprofit_enabled": STRATEGY_PARAMS["atr_takeprofit_enabled"],
    "atr_takeprofit_multiplier": STRATEGY_PARAMS["atr_takeprofit_multiplier"],
    "atr_takeprofit_min_pct": STRATEGY_PARAMS["atr_takeprofit_min_pct"],
    # MACD
    "macd_fastperiod": STRATEGY_PARAMS["macd_fastperiod"],
    "macd_slowperiod": STRATEGY_PARAMS["macd_slowperiod"],
    "macd_signalperiod": STRATEGY_PARAMS["macd_signalperiod"],
    # Anti-pump и Pullback
    "anti_pump_filter_enabled": STRATEGY_PARAMS.get("anti_pump_filter_enabled", True),
    "max_candle_growth": STRATEGY_PARAMS.get("max_candle_growth", 0.3),
    "three_bar_growth_filter_enabled": STRATEGY_PARAMS.get("three_bar_growth_filter_enabled", True),
    "max_three_bar_growth": STRATEGY_PARAMS.get("max_three_bar_growth", 1.0),
    "pullback_filter_enabled": STRATEGY_PARAMS.get("pullback_filter_enabled", True),
    "min_pullback": STRATEGY_PARAMS.get("min_pullback", 0.7),
    "pullback_lookback": STRATEGY_PARAMS.get("pullback_lookback", 5),
    # BBTrend
    "bbtrend_period": STRATEGY_PARAMS["bbtrend_period"],
    "bbtrend_stddev": STRATEGY_PARAMS["bbtrend_stddev"],
    "bbtrend_buy_min": STRATEGY_PARAMS["bbtrend_buy_min"],
    "bbtrend_sell_max": STRATEGY_PARAMS["bbtrend_sell_max"],
    "bbtrend_long_period": STRATEGY_PARAMS["bbtrend_long_period"],
}

# ==================================================
# НАСТРОЙКИ ДЛЯ КАЖДОЙ ТОРГОВОЙ ПАРЫ
# ==================================================

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

        return True


# Создаем и настраиваем фильтр
log_filter = LogFilter()


# Кастомный фильтр для исключения WARNING из файла
class NoWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING  # Только ниже WARNING (INFO и ниже)


# Настраиваем файловый обработчик (без WARNING)
file_handler.addFilter(NoWarningFilter())
file_handler.addFilter(log_filter)
logger.addHandler(file_handler)

# Настраиваем консольный обработчик (оставляем WARNING)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
console_handler.addFilter(log_filter)  # Применяем тот же фильтр
logger.addHandler(console_handler)

# Отключаем логирование для API и других шумных модулей
for module in ["freqtrade.rpc.api_server", "freqtrade.rpc.telegram", "urllib3"]:
    logging.getLogger(module).setLevel(logging.ERROR)

# Создаём специальный логгер для стратегии
strategy_logger = logging.getLogger("OrderBookDepthStrategy")


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

    # Определяем minimal_roi как атрибут класса напрямую
    minimal_roi = STRATEGY_PARAMS.get(
        "minimal_roi_settings", {"0": 0.1, "30": 0.05, "60": 0.03, "120": 0.01}
    )

    # ROI для трейлинг-стопа (инициализируем пустым, будет установлен в update_roi_settings)
    roi_for_trailing = {}

    def __init__(self, *args, **kwargs):
        """
        Инициализация стратегии.
        """
        super().__init__(*args, **kwargs)

        # Обновляем roi_for_trailing при инициализации
        self.update_roi_settings()

    def update_roi_settings(self):
        """
        Устанавливает roi_for_trailing на основе значения из STRATEGY_PARAMS.
        """
        try:
            # Получаем настройки из STRATEGY_PARAMS
            roi_settings = STRATEGY_PARAMS.get("roi_for_trailing")

            if not roi_settings:
                raise ValueError("roi_for_trailing не найден в настройках стратегии")

            # Преобразуем ключи в int и значения в float
            self.roi_for_trailing = {int(k): float(v) for k, v in roi_settings.items()}

            # Устанавливаем trailing_stop_positive_offset на основе минимального ROI
            min_roi = min(self.roi_for_trailing.values())
            self.trailing_stop_positive_offset = min(min_roi + 0.001, 1.0)

            self.logger.info(
                f"Обновлены настройки ROI: {self.roi_for_trailing}, "
                f"trailing_stop_positive_offset: {self.trailing_stop_positive_offset}"
            )

        except Exception as e:
            self.logger.error(f"Ошибка при обновлении настроек ROI: {str(e)}")
            raise

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
    # minimal_roi = STRATEGY_PARAMS['minimal_roi_settings']

    # Фильтры
    rsi_period = STRATEGY_PARAMS["rsi_period"]
    rsi_buy_threshold = STRATEGY_PARAMS["rsi_buy_threshold"]
    rsi_sell_threshold = STRATEGY_PARAMS["rsi_sell_threshold"]
    rsi_buy_enabled = STRATEGY_PARAMS["rsi_buy_enabled"]
    rsi_sell_enabled = STRATEGY_PARAMS["rsi_sell_enabled"]
    ema_buy_enabled = STRATEGY_PARAMS["ema_buy_enabled"]
    ema_sell_enabled = STRATEGY_PARAMS["ema_sell_enabled"]
    ema_slow_period = STRATEGY_PARAMS["ema_slow_period"]
    macd_buy_enabled = STRATEGY_PARAMS["macd_buy_enabled"]
    macd_sell_enabled = STRATEGY_PARAMS["macd_sell_enabled"]
    bbtrend_buy_enabled = STRATEGY_PARAMS["bbtrend_buy_enabled"]
    bbtrend_sell_enabled = STRATEGY_PARAMS["bbtrend_sell_enabled"]
    atr_filter_enabled = STRATEGY_PARAMS["atr_filter_enabled"]
    atr_period = STRATEGY_PARAMS["atr_period"]
    atr_min_volatility_pct = STRATEGY_PARAMS["atr_min_volatility_pct"]
    atr_stoploss_enabled = STRATEGY_PARAMS["atr_stoploss_enabled"]
    atr_stoploss_multiplier = STRATEGY_PARAMS["atr_stoploss_multiplier"]
    atr_takeprofit_enabled = STRATEGY_PARAMS["atr_takeprofit_enabled"]
    atr_takeprofit_multiplier = STRATEGY_PARAMS["atr_takeprofit_multiplier"]
    atr_takeprofit_min_pct = STRATEGY_PARAMS["atr_takeprofit_min_pct"]
    macd_fastperiod = STRATEGY_PARAMS["macd_fastperiod"]
    macd_slowperiod = STRATEGY_PARAMS["macd_slowperiod"]
    macd_signalperiod = STRATEGY_PARAMS["macd_signalperiod"]
    bbtrend_period = STRATEGY_PARAMS["bbtrend_period"]
    bbtrend_stddev = STRATEGY_PARAMS["bbtrend_stddev"]
    bbtrend_buy_min = STRATEGY_PARAMS["bbtrend_buy_min"]
    bbtrend_sell_max = STRATEGY_PARAMS["bbtrend_sell_max"]
    bbtrend_long_period = STRATEGY_PARAMS["bbtrend_long_period"]
    # --- ДОБАВЛЯЕМ ОТСУТСТВУЮЩИЕ ПАРАМЕТРЫ ---
    min_volume_ratio = STRATEGY_PARAMS["min_volume_ratio"]
    volume_threshold = STRATEGY_PARAMS["volume_threshold"]
    cluster_price_distance = STRATEGY_PARAMS["cluster_price_distance"]
    stoploss = STRATEGY_PARAMS["stoploss"]
    # minimal_roi = STRATEGY_PARAMS['minimal_roi_settings']
    anti_pump_filter_enabled = STRATEGY_PARAMS.get("anti_pump_filter_enabled", True)
    max_candle_growth = STRATEGY_PARAMS.get("max_candle_growth", 0.3)
    three_bar_growth_filter_enabled = STRATEGY_PARAMS.get("three_bar_growth_filter_enabled", True)
    max_three_bar_growth = STRATEGY_PARAMS.get("max_three_bar_growth", 1.0)
    pullback_filter_enabled = STRATEGY_PARAMS.get("pullback_filter_enabled", True)
    min_pullback = STRATEGY_PARAMS.get("min_pullback", 0.7)
    pullback_lookback = STRATEGY_PARAMS.get("pullback_lookback", 5)

    # === Настройки усреднения для DCA ===
    position_adjustment_enable = True
    max_entry_position_adjustment = 3  # Максимальное количество уровней усреднения
    min_dca_time = 1  # Минимальный промежуток между докупками (в минутах)

    # === Параметры для управления рисками ===
    risk_per_trade = 0.03  # 3% риска на сделку

    # === Динамический размер позиции ===
    use_atr_calculation = (
        True  # Использовать ATR для расчета размера позиции (True = ATR, False = фиксированный)
    )
    atr_position_multiplier = 2.5  # 2.0 Множитель волатильности ATR для расчета размера позиции (чем выше, тем меньше позиция при высокой волатильности)
    # Диапазон: от минимального значения биржи до stake_amount (100 USDT)
    max_position_size_pct = 95  # % от stake_amount из конфига (100% = 100 USDT)

    # === Настройки для оптимизации производительности ===
    _last_ohlcv_update = {}
    _ohlcv_update_interval = 5  # 5 секунд вместо 1
    _last_orderbook_update = {}
    _orderbook_update_interval = 5  # 5 секунд вместо 1

    # Кэш для хранения данных книги ордеров
    _orderbook_cache = {}
    _orderbook_cache_time = {}
    _orderbook_cache_ttl = 5  # 5 секунд вместо 1
    exchange_instance = None  # Инициализация экземпляра биржи

    # === Настройки трейлинг-стопа (Take profit) ===
    trailing_stop = True
    trailing_stop_positive = 0.003  # отставание трейлинг-стопа на 0.3% от текущей цены
    # trailing_stop_positive_offset будет установлен в populate_trailing_data
    trailing_only_offset_is_reached = True  # Активация только после достижения смещения

    # === Настройки частичного закрытия позиции ===
    partial_profit_enabled = False
    partial_profit_levels = {
        0.01: 0.3,  # При 1% прибыли закрываем 30% позиции
        0.02: 0.3,  # При 2% прибыли закрываем еще 30%
        0.03: 0.4,  # При 3% прибыли закрываем оставшиеся 40%
    }

    def adjust_trade_position(
        self, trade, current_time, current_rate, current_profit, min_stake, max_stake, **kwargs
    ):
        """
        DCA усреднение и частичная фиксация прибыли
        """
        try:
            dca_levels = {0: -0.005, 1: -0.01, 2: -0.015}  # -0.5%, -1%, -1.5%
            current_entries = getattr(trade, "nr_of_successful_entries", 0)

            # --- Логирование попытки DCA ---
            # last_dca_time = getattr(trade, 'ft_last_dca_time', None)
            # minutes_since_last_dca = None
            # if last_dca_time is not None:
            #     minutes_since_last_dca = (current_time - last_dca_time).total_seconds() / 60
            # self.logger.info(
            #     f"DCA CHECK | pair={trade.pair} | entries={current_entries} | profit={current_profit:.4f} | "
            #     f"level={dca_levels.get(current_entries, 'NA')} | min_dca_time={self.min_dca_time} | "
            #     f"last_dca_time={last_dca_time} | minutes_since_last_dca={minutes_since_last_dca}"
            # )

            # --- Проверка минимального времени между докупками ---
            # if last_dca_time is not None and minutes_since_last_dca is not None:
            # if minutes_since_last_dca < self.min_dca_time:
            # self.logger.info(
            # f"DCA SKIP: прошло только {minutes_since_last_dca:.2f} мин, нужно минимум {self.min_dca_time} мин."
            # )
            # return None

            # --- DCA: докупка при просадке ---
            if current_entries < self.max_entry_position_adjustment:
                if current_profit <= dca_levels.get(current_entries, -999):
                    multiplier = 1.5
                    dca_amount = trade.stake_amount * (multiplier**current_entries)

                    config_stake_amount = self.config.get("stake_amount", 100)
                    max_position = config_stake_amount * (self.max_position_size_pct / 100)
                    original_dca_amount = dca_amount
                    dca_amount = min(dca_amount, max_position)

                    if dca_amount < min_stake:
                        self.logger.warning(
                            f"DCA amount {dca_amount} too small. Min required: {min_stake}"
                        )
                        return None

                    if original_dca_amount > max_position:
                        self.logger.info(
                            f"DCA amount reduced from {original_dca_amount:.2f} to {dca_amount:.2f} (max_position limit)"
                        )

                    self.logger.info(
                        f"DCA EXECUTE: Докупка для {trade.pair} на уровне {current_profit * 100:.2f}%: {dca_amount:.2f} USDT"
                    )
                    # --- Сохраняем время последней докупки ---
                    setattr(trade, "ft_last_dca_time", current_time)
                    return dca_amount
                else:
                    # self.logger.info(
                    #     f"DCA SKIP: profit {current_profit:.4f} выше порога {dca_levels.get(current_entries, -999)}"
                    # )
                    pass

            # --- Частичная фиксация прибыли (оставьте как было) ---
            if not getattr(trade, "has_open_orders", False) and self.partial_profit_enabled:
                for level, exit_percent in sorted(self.partial_profit_levels.items()):
                    if current_profit >= level:
                        level_key = f"partial_exit_{int(level * 100)}"
                        if not getattr(trade, level_key, False):
                            exit_amount = trade.amount * exit_percent
                            if exit_amount * current_rate >= min_stake:
                                setattr(trade, level_key, True)
                                self.logger.info(
                                    f"Частичная фиксация {trade.pair} на уровне {level * 100:.1f}%: "
                                    f"закрываем {exit_percent * 100:.0f}% позиции ({exit_amount:.8f})"
                                )
                                return -exit_amount
                            else:
                                self.logger.warning(
                                    f"Exit amount {exit_amount} too small for level {level * 100:.1f}%"
                                )
                                break

            return None
        except Exception as e:
            self.logger.error(f"Error in adjust_trade_position: {e}")
            return None

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ) -> float:
        """
        Динамический расчет размера позиции на основе ATR и управления риском

        Args:
            pair: Торговая пара
            current_time: Текущее время
            current_rate: Текущая цена
            proposed_stake: Предлагаемый размер позиции
            min_stake: Минимальный размер позиции
            max_stake: Максимальный размер позиции

        Returns:
            float: Рассчитанный размер позиции
        """
        try:
            if not self.use_atr_calculation:
                return proposed_stake

            # Получаем баланс счета
            exchange = self.dp._exchange
            if not exchange:
                logger.warning(f"Exchange not available for {pair}, using proposed stake")
                return proposed_stake

            # Получаем баланс аккаунта (исправлено)
            try:
                balances = exchange.get_balances()
                balance = balances.get("USDT", {}).get("free", 0)
            except Exception as e:
                logger.warning(f"Error getting balance: {e}, using proposed stake")
                return proposed_stake

            if balance is None or balance <= 0:
                logger.warning(f"Invalid balance: {balance}, using proposed stake")
                return proposed_stake

            # Получаем данные для расчета ATR
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            atr_period = self._get_param_value("atr_period")
            if dataframe.empty or len(dataframe) < atr_period:
                logger.warning("Not enough data for ATR calculation, using proposed stake")
                return proposed_stake

            atr = dataframe["atr"].iloc[-1]
            if pd.isna(atr) or atr <= 0:
                logger.warning(f"Invalid ATR value: {atr}, using proposed stake")
                return proposed_stake

            # Получаем базовые значения из конфига
            config_stake_amount = self.config.get("stake_amount", 100)  # Значение по умолчанию 100

            # Получаем минимальные значения биржи для данной пары
            try:
                market = exchange.market(pair)
                min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
                min_cost = market.get("limits", {}).get("cost", {}).get("min", 0)

                # Рассчитываем минимальный размер в USDT
                exchange_min_stake = (
                    max(min_cost, min_amount * current_rate)
                    if min_amount > 0 and min_cost > 0
                    else min_cost
                )

                self.logger.info(
                    f"Exchange limits for {pair}: min_amount={min_amount}, min_cost={min_cost}, calculated_min_stake={exchange_min_stake:.2f} USDT"
                )

            except Exception as e:
                self.logger.warning(f"Error getting exchange limits for {pair}: {e}")
                exchange_min_stake = min_stake  # Используем переданное значение как fallback

            # Рассчитываем минимальный и максимальный размер позиции
            # Минимальный размер = биржевой минимум (фиксированное значение)
            min_position = exchange_min_stake
            # Максимальный размер = процент от stake_amount из конфига
            max_position = config_stake_amount * (self.max_position_size_pct / 100)

            # Рассчитываем размер позиции
            if self.use_atr_calculation:
                # ATR-based расчет - исправленная формула
                risk_amount = balance * self.risk_per_trade

                # Рассчитываем ATR как процент от цены
                atr_pct = (atr / current_rate) * 100

                # Используем ATR как меру волатильности для расчета размера позиции
                # Чем выше волатильность (ATR), тем меньше размер позиции
                volatility_factor = atr_pct * self.atr_position_multiplier

                # Рассчитываем размер позиции на основе волатильности
                # При высокой волатильности размер позиции уменьшается
                if volatility_factor > 0:
                    atr_based_size = risk_amount / (volatility_factor / 100)
                else:
                    # Если волатильность слишком низкая, используем фиксированный размер
                    atr_based_size = (min_position + max_position) / 2

                # Проверяем валидность ATR-расчета
                if atr_based_size <= 0 or pd.isna(atr_based_size):
                    # Если ATR-расчет не работает, используем среднее значение диапазона
                    position_size = (min_position + max_position) / 2
                    self.logger.warning(
                        f"ATR calculation failed for {pair}, using fixed size: {position_size:.2f}"
                    )
                else:
                    # Применяем логику ограничений:
                    # ATR size > max → Размер позиции = max
                    # ATR size < min → Размер позиции = min
                    # min < ATR size < max → Размер позиции = ATR size
                    if atr_based_size > max_position:
                        position_size = max_position
                        self.logger.info(
                            f"ATR size ({atr_based_size:.2f}) > max ({max_position:.2f}), using max"
                        )
                    elif atr_based_size < min_position:
                        position_size = min_position
                        self.logger.info(
                            f"ATR size ({atr_based_size:.2f}) < min ({min_position:.2f}), using min"
                        )
                    else:
                        position_size = atr_based_size
                        self.logger.info(
                            f"ATR size ({atr_based_size:.2f}) within range, using ATR value"
                        )
            else:
                # Фиксированный размер без ATR-расчета
                # Используем среднее значение диапазона: (min + max) / 2
                position_size = (min_position + max_position) / 2
                self.logger.info(
                    f"Using fixed position size: {position_size:.2f} (average of min={min_position:.2f} and max={max_position:.2f})"
                )

            # Дополнительная проверка: не должен быть меньше min_stake и больше max_position
            original_position_size = position_size
            position_size = max(position_size, min_stake)
            if original_position_size < min_stake:
                self.logger.info(
                    f"Position size increased from {original_position_size:.2f} to {position_size:.2f} (min_stake limit)"
                )

            position_size = min(position_size, max_position)
            if original_position_size > max_position:
                self.logger.info(
                    f"Position size reduced from {original_position_size:.2f} to {position_size:.2f} (max_position limit)"
                )

            # Подробное логирование для отладки
            if self.use_atr_calculation:
                atr_pct = (atr / current_rate) * 100
                volatility_factor = atr_pct * self.atr_position_multiplier
                self.logger.info(
                    f"Position size calculation for {pair} (ATR-based): "
                    f"Balance={balance:.2f}, ATR={atr:.8f} ({atr_pct:.2f}%), "
                    f"Volatility factor={volatility_factor:.2f}%, "
                    f"Risk amount={risk_amount:.2f}, "
                    f"ATR-based size={atr_based_size:.2f}, "
                    f"Exchange min={exchange_min_stake:.2f}, Stake amount={config_stake_amount:.2f}, "
                    f"Max%={self.max_position_size_pct}%({config_stake_amount * (self.max_position_size_pct / 100):.2f}), "
                    f"Range: {min_position:.2f}-{max_position:.2f} USDT, "
                    f"Final size={position_size:.2f} USDT"
                )
            else:
                self.logger.info(
                    f"Position size calculation for {pair} (Fixed size): "
                    f"Exchange min={exchange_min_stake:.2f}, Stake amount={config_stake_amount:.2f}, "
                    f"Max%={self.max_position_size_pct}%({config_stake_amount * (self.max_position_size_pct / 100):.2f}), "
                    f"Range: {min_position:.2f}-{max_position:.2f} USDT, "
                    f"Fixed size={position_size:.2f} USDT"
                )

            return position_size

        except Exception as e:
            self.logger.error(f"Error in custom_stake_amount for {pair}: {str(e)}", exc_info=True)
            return min_stake

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

    def adjust_trailing_stop(
        self,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Динамически обновляет trailing_stop_positive_offset на основе времени удержания позиции
        Использует roi_for_trailing для определения уровней выхода
        Возвращает новый offset только если он изменился
        """
        # Check if we have a cached offset for this pair
        if trade.pair in self._last_offset:
            last_offset = self._last_offset[trade.pair]
            # If we have a recent update (within last 5 minutes), return the cached value
            if (
                current_time - trade.open_date_utc
            ).total_seconds() < 300 and last_offset is not None:
                return last_offset

        try:
            self.logger.debug(
                f"[DEBUG] Вход в adjust_trailing_stop. roi_for_trailing: {getattr(self, 'roi_for_trailing', 'не установлен')}"
            )

            # Получаем словарь ROI для трейлинг-стопа
            roi_dict = getattr(self, "roi_for_trailing", {})

            # Проверяем, что словарь не пустой
            if not roi_dict:
                self.logger.warning(
                    "roi_for_trailing пуст, используется значение по умолчанию 0.01"
                )
                return 0.01

            # Создаем копию словаря для безопасной работы
            roi_dict = {str(k): float(v) for k, v in roi_dict.items() if float(v) > 0}

            if not roi_dict:
                self.logger.error("Нет валидных значений ROI в roi_for_trailing")
                return None

            self.logger.debug(f"[DEBUG] Обрабатываемый roi_for_trailing: {roi_dict}")

            # Получаем и сортируем ключи времени по возрастанию
            try:
                time_keys = [int(float(k)) for k in roi_dict.keys()]
                time_keys_sorted = sorted(time_keys)
                self.logger.debug(f"Доступные временные ключи (мин): {time_keys_sorted}")
            except (ValueError, TypeError) as e:
                self.logger.error(f"Ошибка при обработке ключей времени: {e}")
                return None

            # Рассчитываем время удержания позиции
            trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
            trade_duration_mins = int(round(trade_duration))
            self.logger.debug(f"Время удержания позиции: {trade_duration_mins} минут")

            # Выбираем подходящий ROI на основе времени удержания
            selected_roi = None
            selected_time = 0

            # Идем от меньшего времени к большему
            for time_key in time_keys_sorted:
                time_key_str = str(time_key)
                if time_key_str in roi_dict and trade_duration_mins >= time_key:
                    selected_time = time_key
                    selected_roi = roi_dict[time_key_str]
                    self.logger.debug(
                        f"Выбран ROI {selected_roi * 100:.2f}% "
                        f"для времени удержания {time_key}+ минут"
                    )

            # Если не нашли подходящий ROI, используем минимальный ненулевой
            if selected_roi is None or selected_roi <= 0:
                valid_roi_values = [v for v in roi_dict.values() if v > 0]
                if valid_roi_values:
                    selected_roi = min(valid_roi_values)
                    selected_time = min(time_keys_sorted) if time_keys_sorted else 0
                    self.logger.debug(
                        f"Используем минимальный ROI {selected_roi * 100:.2f}% "
                        f"для времени {selected_time} минут "
                        f"(время удержания: {trade_duration_mins} минут)"
                    )
                else:
                    self.logger.error(f"Нет валидных значений ROI в словаре: {roi_dict}")
                    return None

            # Проверяем, достигли ли мы целевого ROI
            if current_profit < selected_roi:
                self.logger.debug(
                    f"[ТРЕЙЛИНГ-СТОП] ОЖИДАНИЕ ДОСТИЖЕНИЯ ЦЕЛЕВОГО ROI: {trade.pair}\n"
                    f"  Текущая прибыль: {current_profit * 100:.2f}% < Требуемый ROI: {selected_roi * 100:.2f}%\n"
                    f"  Время удержания: {trade_duration_mins} минут"
                )
                return None  # Не активируем трейлинг-стоп, пока не достигнут целевой ROI

            # Устанавливаем offset с небольшим запасом (0.1%)
            new_offset = float(selected_roi) + 0.001

            # Проверяем, изменился ли offset
            if (
                trade.pair not in self._last_offset
                or abs(self._last_offset.get(trade.pair, 0) - new_offset) > 0.0001
            ):
                logger_method = (
                    self.logger.info
                    if self.dp.runmode.value in ("live", "dry_run")
                    else self.logger.debug
                )
                logger_method(
                    f"[ТРЕЙЛИНГ-СТОП] АКТИВАЦИЯ ДЛЯ {trade.pair}:\n"
                    f"  Время удержания: {trade_duration_mins} минут\n"
                    f"  Текущая прибыль: {current_profit * 100:.2f}% >= Требуемый ROI: {selected_roi * 100:.2f}%\n"
                    f"  Установлен offset: {new_offset * 100:.2f}%"
                )
                # Обновляем кэшированный offset
                self._last_offset[trade.pair] = new_offset

            return new_offset

        except Exception as e:
            self.logger.error(f"Ошибка в adjust_trailing_stop: {str(e)}", exc_info=True)
            return None

    def __init__(self, config: dict) -> None:
        # Dictionary to store the last offset for each trading pair
        self._last_offset = {}

        """
        Инициализация стратегии с настройками для конкретной пары
        """
        # Инициализируем родительский класс в первую очередь
        super().__init__(config)

        # Инициализация логгера
        self.logger = logging.getLogger("OrderBookDepth")

        # Получаем настройки для пары (если указана)
        pair = config.get("pair", "")
        self.pair_settings = get_pair_settings(pair)

        # Инициализируем minimal_roi из настроек пары или используем значения по умолчанию
        if "minimal_roi" in self.pair_settings:
            self.minimal_roi = self.pair_settings["minimal_roi"]

        # Инициализируем roi_for_trailing
        if "roi_for_trailing" in self.pair_settings:
            self.roi_for_trailing = self.pair_settings["roi_for_trailing"]

        # Логируем начальные значения
        self.logger.debug(
            f"Начальное значение minimal_roi: {getattr(self, 'minimal_roi', 'не установлено')}"
        )
        self.logger.debug(
            f"Начальное значение roi_for_trailing: {getattr(self, 'roi_for_trailing', 'не установлено')}"
        )

        # Устанавливаем остальные атрибуты стратегии из настроек
        for key, value in self.pair_settings.items():
            if hasattr(self, key) and key not in ["minimal_roi", "roi_for_trailing"]:
                if not isinstance(getattr(type(self), key, None), property):
                    setattr(self, key, value)

        # Обновляем roi_for_trailing на основе minimal_roi
        self.update_roi_settings()
        self.logger.debug(
            f"Настройки ROI инициализированы. minimal_roi: {self.minimal_roi}, roi_for_trailing: {self.roi_for_trailing}"
        )

        # Убедимся, что roi_for_trailing установлен
        if not hasattr(self, "roi_for_trailing") or not self.roi_for_trailing:
            self.roi_for_trailing = {str(k): float(v) for k, v in self.minimal_roi.items()}
            self.logger.warning(
                f"[WARNING] roi_for_trailing не был установлен, используем значения из minimal_roi: {self.roi_for_trailing}"
            )

        # Настройки трейлинг-стопа
        self.trailing_stop = True
        self.trailing_stop_positive = 0.003  # 0.3% отставание
        self.trailing_only_offset_is_reached = True

        # Используем roi_for_trailing, определенный как атрибут класса
        self.logger.info(f"Используются настройки roi_for_trailing: {self.roi_for_trailing}")

        # Устанавливаем начальное значение offset из roi_for_trailing
        if "0" in self.roi_for_trailing:
            # Преобразуем ROI из процентов в доли (например, 100% -> 1.0)
            roi_value = float(self.roi_for_trailing["0"])
            # Если ROI больше 1, делим на 100 (предполагаем, что было введено в процентах)
            if roi_value > 1.0:
                roi_value = roi_value / 100.0
            self.trailing_stop_positive_offset = min(
                roi_value + 0.001, 1.0
            )  # Ограничиваем максимальное значение 1.0
        else:
            # Если нет значения для 0 минут, используем минимальное значение
            min_roi = min(self.roi_for_trailing.values()) if self.roi_for_trailing else 0.025
            # Преобразуем ROI из процентов в доли, если нужно
            if min_roi > 1.0:
                min_roi = min_roi / 100.0
            self.trailing_stop_positive_offset = min(
                min_roi + 0.001, 1.0
            )  # Ограничиваем максимальное значение 1.0

        self.logger.debug(f"Инициализация стратегии OrderBookDepth для пары {pair}")
        self.logger.debug(f"Настройки: {self.pair_settings}")
        self.logger.debug(
            f"Начальные настройки трейлинг-стопа: offset={self.trailing_stop_positive_offset * 100:.2f}%, "
            f"отставание={self.trailing_stop_positive * 100:.2f}%"
        )

        # Валидация критических параметров
        if not self._validate_parameters():
            self.logger.error("Критические параметры некорректны! Проверьте настройки стратегии.")

        # Проверяем whitelist пар
        self.verify_pair_whitelist()

        # Инициализация exchange с использованием переменных среды
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_API_SECRET")

        if api_key is None or secret_key is None:
            self.logger.warning(
                "Переменные окружения BINANCE_API_KEY и/или BINANCE_API_SECRET не установлены. "
                "Функциональность книги ордеров будет ограничена."
            )
        else:
            try:
                self.exchange_instance = ccxt.binance(
                    {
                        "apiKey": api_key,
                        "secret": secret_key,
                        "enableRateLimit": True,
                        "timeout": 30000,
                        "options": {
                            "defaultType": "spot",
                            "adjustForTimeDifference": True,
                            "recvWindow": 5000,
                            "warnOnFetchOHLCVLimitArgument": False,
                            "defaultTimeInForce": "GTC",
                            "createMarketBuyOrderRequiresPrice": False,
                            "fetchBalance": {"recvWindow": 10000},
                            "fetchOrderBook": {"limit": 100, "recvWindow": 10000},
                            "fetchTrades": {"limit": 1000},
                            "fetchOHLCV": {"limit": 1000},
                        },
                    }
                )

                # Предварительная загрузка рынков
                self.logger.info("Загрузка информации о рынках...")
                self.exchange_instance.load_markets()
                self.logger.info("Информация о рынках успешно загружена")

            except Exception as e:
                self.logger.error(f"Ошибка при инициализации биржевого подключения: {str(e)}")
                self.exchange_instance = None

    def get_order_book(self, pair: str, depth: int = 10) -> Optional[Dict[str, List[List[float]]]]:
        """
        Получает данные книги ордеров для указанной пары с кэшированием

        Args:
            pair: Торговая пара (например, 'BTC/USDT')
            depth: Глубина стакана (по умолчанию 10)

        Returns:
            Словарь с данными стакана (bids, asks) или None в случае ошибки
        """
        current_time = datetime.now(timezone.utc)

        # Проверяем кэш
        cache_key = f"{pair}_{depth}"
        if cache_key in self._orderbook_cache:
            cache_time = self._orderbook_cache_time.get(cache_key)
            if (
                cache_time
                and (current_time - cache_time).total_seconds() < self._orderbook_cache_ttl
            ):
                return self._orderbook_cache[cache_key]

        # Если нет экземпляра биржи, логируем предупреждение и возвращаем None
        if not self.exchange_instance:
            self.logger.warning(
                f"Не удалось получить стакан для {pair}: биржевое подключение не инициализировано"
            )
            return None

        try:
            # Получаем новые данные
            self.logger.debug(f"Запрос стакана для {pair} (глубина: {depth})")
            orderbook = self.exchange_instance.fetch_order_book(pair, depth)

            # Проверяем структуру данных
            if (
                not isinstance(orderbook, dict)
                or "bids" not in orderbook
                or "asks" not in orderbook
            ):
                self.logger.error(f"Некорректная структура стакана для {pair}")
                return None

            # Обновляем кэш
            self._orderbook_cache[cache_key] = {
                "bids": orderbook["bids"],
                "asks": orderbook["asks"],
                "timestamp": orderbook.get("timestamp", int(current_time.timestamp() * 1000)),
                "datetime": orderbook.get("datetime", current_time.isoformat()),
            }
            self._orderbook_cache_time[cache_key] = current_time

            # Логируем успешное обновление кэша
            self.logger.debug(
                f"Обновлен кэш стакана для {pair}, размер стакана: "
                f"{len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks"
            )

            return self._orderbook_cache[cache_key]

        except Exception as e:
            self.logger.error(f"Ошибка при получении стакана для {pair}: {str(e)}")
            return None

    def invalidate_orderbook_cache(self, pair: str = None, depth: int = None) -> None:
        """
        Инвалидирует кэш стакана для указанной пары и глубины

        Args:
            pair: Торговая пара (если None, очищается кэш для всех пар)
            depth: Глубина стакана (если None, очищаются все глубины для указанной пары)
        """
        if pair is None:
            # Очищаем весь кэш, если не указана пара
            self._orderbook_cache.clear()
            self._orderbook_cache_time.clear()
            self.logger.debug("Очищен кэш стакана для всех пар")
        else:
            cache_key = f"{pair}_{depth}" if depth is not None else pair
            keys_to_delete = [k for k in self._orderbook_cache.keys() if k.startswith(cache_key)]

            for key in keys_to_delete:
                if key in self._orderbook_cache:
                    del self._orderbook_cache[key]
                if key in self._orderbook_cache_time:
                    del self._orderbook_cache_time[key]

            if keys_to_delete:
                self.logger.debug(
                    f"Очищен кэш стакана для пары {pair} (глубина: {depth if depth else 'все'})"
                )

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
            # RSI - используем собственную реализацию для соответствия TradingView
            dataframe["rsi"] = self._calculate_rsi_tradingview(
                dataframe["close"], self._get_param_value("rsi_period")
            )
            dataframe["ema_slow"] = ta.EMA(
                dataframe, timeperiod=self._get_param_value("ema_slow_period")
            )
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=self._get_param_value("atr_period"))

            # --- MACD ---
            macd = ta.MACD(
                dataframe,
                fastperiod=self._get_param_value("macd_fastperiod"),
                slowperiod=self._get_param_value("macd_slowperiod"),
                signalperiod=self._get_param_value("macd_signalperiod"),
            )
            dataframe["macd"] = macd["macd"]
            dataframe["macdsignal"] = macd["macdsignal"]
            dataframe["macdhist"] = macd["macdhist"]

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

            # Дополнительное логирование RSI для отладки
            if len(dataframe) > 0:
                last_rsi = dataframe["rsi"].iloc[-1]
                if not pd.isna(last_rsi):
                    self.logger.debug(f"[{pair}] RSI (TradingView style): {last_rsi:.2f}")
                else:
                    self.logger.warning(f"[{pair}] RSI не рассчитан (NaN)")

            # --- BBTrend ---
            bb_short_period = self._get_param_value("bbtrend_period")
            bb_long_period = self._get_param_value("bbtrend_long_period")
            bb_stddev = self._get_param_value("bbtrend_stddev")

            # Короткие полосы
            bb_short = ta.BBANDS(
                dataframe, timeperiod=bb_short_period, nbdevup=bb_stddev, nbdevdn=bb_stddev
            )
            # Длинные полосы
            bb_long = ta.BBANDS(
                dataframe, timeperiod=bb_long_period, nbdevup=bb_stddev, nbdevdn=bb_stddev
            )

            # BBTrend = ширина короткой / ширину длинной
            bb_short_width = bb_short["upperband"] - bb_short["lowerband"]
            bb_long_width = bb_long["upperband"] - bb_long["lowerband"]
            # Чтобы избежать деления на 0
            dataframe["bbtrend"] = bb_short_width / bb_long_width.replace(0, float("nan"))

            # Логирование последних 5 значений bbtrend для проверки
            # self.logger.info(f"BBTrend (последние 5): {dataframe['bbtrend'].tail(5).tolist()}")

            return dataframe

        except Exception as e:
            logger.error(f"Ошибка в populate_indicators для {metadata.get('pair')}: {e}")
            # Возвращаем dataframe с базовыми индикаторами при ошибке
            try:
                dataframe["rsi"] = self._calculate_rsi_tradingview(dataframe["close"], 14)
                dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=100)
                dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
                # MACD fallback
                macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
                dataframe["macd"] = macd["macd"]
                dataframe["macdsignal"] = macd["macdsignal"]
                dataframe["macdhist"] = macd["macdhist"]
                dataframe["volume_ratio"] = (
                    dataframe["volume"] / dataframe["volume"].rolling(20).mean()
                )

                # Устанавливаем значения по умолчанию для кластеров
                dataframe["buy_cluster_price"] = 0
                dataframe["buy_cluster_volume"] = 0
                dataframe["buy_cluster_orders"] = 0
                dataframe["sell_cluster_price"] = 0
                dataframe["sell_cluster_volume"] = 0
                dataframe["sell_cluster_orders"] = 0
            except Exception as fallback_error:
                logger.error(f"Ошибка при установке базовых индикаторов: {fallback_error}")

            return dataframe

    def _log_signals(self, pair: str, dataframe: DataFrame, ob_analysis: dict) -> Dict:
        """
        Логирование торговых сигналов и информации о стакане
        """
        try:
            last_row = dataframe.iloc[-1]

            # Проверяем наличие открытой позиции и выполнение ROI
            has_position = self._has_open_position(pair)
            roi_condition_met = has_position and self._check_roi_condition(pair, last_row["close"])

            # Получаем информацию о кластерах из анализа стакана
            buy_cluster = ob_analysis.get("buy_cluster", {})
            sell_cluster = ob_analysis.get("sell_cluster", {})

            # Получаем объемы для сравнения
            buy_volume = last_row.get("buy_cluster_volume", 0)
            sell_volume = last_row.get("sell_cluster_volume", 0)

            # Получаем текущее значение minimal_roi
            try:
                # Инициализируем переменные
                current_roi = None
                trade_duration = 0

                if has_position and hasattr(self, "dp") and self.dp is not None:
                    try:
                        # Получаем открытые сделки
                        open_trades = Trade.get_trades_proxy(is_open=True)
                        for t in open_trades:
                            if t.pair == pair:
                                # Рассчитываем время удержания в минутах
                                current_time = datetime.now(timezone.utc)
                                trade_duration = (
                                    current_time - t.open_date_utc
                                ).total_seconds() / 60  # в минутах

                                # Детальное логирование времени (только в режиме DEBUG)
                                self.logger.debug(f"[{pair}] Текущее время: {current_time}")
                                self.logger.debug(
                                    f"[{pair}] Время открытия позиции: {t.open_date_utc}"
                                )
                                self.logger.debug(
                                    f"[{pair}] Время удержания позиции: {trade_duration:.1f} минут"
                                )

                                # Получаем ключи времени из minimal_roi и сортируем их по убыванию
                                time_keys = [
                                    int(k) for k in self.minimal_roi.keys() if str(k).isdigit()
                                ]
                                time_keys_sorted = sorted(
                                    time_keys, reverse=True
                                )  # Сортируем по убыванию

                                # Округляем время удержания до целого для сравнения с ключами
                                trade_duration_mins = int(round(trade_duration))

                                # Логируем информацию для отладки (только в режиме DEBUG)
                                self.logger.debug(f"[{pair}] Доступные ROI: {self.minimal_roi}")
                                self.logger.debug(
                                    f"[{pair}] Отсортированные ключи времени (по убыванию): {time_keys_sorted}"
                                )
                                self.logger.debug(
                                    f"[{pair}] Время удержания (целое): {trade_duration_mins} минут"
                                )
                                self.logger.debug(
                                    f"[{pair}] Точное время удержания: {trade_duration:.2f} минут"
                                )

                                if not time_keys_sorted:
                                    current_roi = 0.01  # 1% по умолчанию
                                    self.logger.error(
                                        f"[{pair}] ОШИБКА: Не найдено временных ключей в minimal_roi. Используем ROI по умолчанию: {current_roi * 100:.2f}%"
                                    )
                                else:
                                    # Устанавливаем ROI по умолчанию (минимальное ненулевое значение)
                                    non_zero_roi = [
                                        float(v) for v in self.minimal_roi.values() if float(v) > 0
                                    ]
                                    default_roi = min(non_zero_roi) if non_zero_roi else 0.01
                                    current_roi = default_roi
                                    self.logger.debug(
                                        f"[{pair}] ROI по умолчанию: {current_roi * 100:.2f}%"
                                    )

                                    # Ищем первый ключ, который меньше или равен времени удержания
                                    for time_key in time_keys_sorted:
                                        roi_value = float(self.minimal_roi[str(time_key)])
                                        if roi_value <= 0:
                                            self.logger.info(
                                                f"[{pair}] Пропускаем ключ {time_key} мин, так как ROI {roi_value * 100:.2f}% <= 0"
                                            )
                                            continue  # Пропускаем нулевые и отрицательные ROI

                                        self.logger.info(
                                            f"[{pair}] Проверяем ключ {time_key} минут: ROI = {roi_value * 100:.2f}%"
                                        )

                                        if trade_duration_mins >= time_key:
                                            current_roi = roi_value
                                            self.logger.info(
                                                f"[{pair}] ВЫБРАНО: ROI {current_roi * 100:.2f}% (время удержания {trade_duration_mins} мин >= {time_key} мин)"
                                            )
                                            break
                                        else:
                                            self.logger.info(
                                                f"[{pair}] Пропускаем ключ {time_key} мин, так как {trade_duration_mins} < {time_key}"
                                            )
                                    else:
                                        # Если не нашли подходящий ключ, используем минимальный ненулевой ROI
                                        if non_zero_roi:
                                            current_roi = min(non_zero_roi)
                                            self.logger.info(
                                                f"[{pair}] Не найден подходящий ключ. Используем минимальный ненулевой ROI: {current_roi * 100:.2f}%"
                                            )
                                        else:
                                            current_roi = 0.01  # 1% по умолчанию
                                            self.logger.warning(
                                                f"[{pair}] Все значения ROI равны 0! Используем ROI по умолчанию: {current_roi * 100:.2f}%"
                                            )

                                self.logger.debug(
                                    f"Текущий ROI: {current_roi * 100:.2f}% для времени удержания {trade_duration:.1f} минут"
                                )
                                break

                    except Exception as e:
                        self.logger.error(f"Ошибка при расчете ROI: {str(e)}")
                        current_roi = 0

                # Если не нашли подходящий ROI, используем значение по умолчанию
                if current_roi is None or current_roi == 0:
                    current_roi = 0.01  # 1% по умолчанию
                    self.logger.warning(
                        f"[{pair}] Не удалось определить ROI, используем значение по умолчанию: {current_roi * 100:.2f}%"
                    )

                # Форматируем ROI с двумя знаками после запятой
                roi_display = f"{current_roi * 100:.2f}%" if current_roi is not None else "N/A"

                # Добавляем время удержания в статус, если есть позиция
                status_extra = []
                if has_position and trade_duration > 0:
                    # Ограничиваем максимальное время удержания 24 часами (1440 минут)
                    trade_duration = min(trade_duration, 1440)
                    hours = int(trade_duration // 60)
                    minutes = int(trade_duration % 60)
                    status_extra.append(f"Время: {hours:02d}:{minutes:02d}")

                    # Добавляем отладочную информацию
                    self.logger.debug(
                        f"Форматированное время удержания: {hours:02d}:{minutes:02d} (из {trade_duration:.1f} мин)"
                    )

                # Создаем краткое сообщение о состоянии
                # Получаем текущий ROI для трейлинг-стопа, если позиция открыта
                roi_ts_display = ""
                time_display = ""
                if has_position and hasattr(trade, "open_date_utc"):
                    try:
                        # Рассчитываем время удержания
                        current_time = datetime.now(timezone.utc)
                        trade_duration = (
                            current_time - trade.open_date_utc
                        ).total_seconds() / 60  # в минутах

                        # Форматируем время удержания
                        hours = int(trade_duration // 60)
                        minutes = int(trade_duration % 60)
                        time_display = f"Время: {hours:02d}:{minutes:02d}"

                        # Получаем целевой ROI для трейлинг-стопа
                        if hasattr(self, "roi_for_trailing") and self.roi_for_trailing:
                            selected_roi = 0
                            # Сортируем ключи времени по возрастанию
                            time_keys = sorted([int(k) for k in self.roi_for_trailing.keys()])
                            for time_key in time_keys:
                                if trade_duration >= time_key:
                                    selected_roi = float(self.roi_for_trailing[str(time_key)])

                            # Если время удержания больше максимального в настройках, используем последнее значение
                            if trade_duration > max(time_keys) if time_keys else 0:
                                selected_roi = float(self.roi_for_trailing[str(max(time_keys))])

                            roi_ts_display = f"ROI_ts: {selected_roi * 100:.2f}%"

                    except Exception as e:
                        self.logger.error(f"Ошибка при расчете времени удержания/ROI_ts: {str(e)}")

                # Формируем базовое сообщение
                status_msg = [
                    f"\n[{pair}] Цена: {last_row['close']:.2f} | ",
                    f"Позиция: {'ДА' if has_position else 'НЕТ'}",
                ]

                # Если есть открытая позиция, добавляем дополнительную информацию
                if has_position:
                    if not hasattr(trade, "open_rate"):
                        self.logger.debug(f"[DEBUG] trade.open_rate не существует для {pair}")
                    else:
                        try:
                            # Рассчитываем текущий ROI
                            current_roi = (last_row["close"] / trade.open_rate - 1) * 100
                            status_msg.append(f" | ROI: {current_roi:+.2f}%")

                            # Добавляем целевой ROI для трейлинг-стопа, если он рассчитан
                            if "roi_ts_display" in locals() and roi_ts_display:
                                status_msg.append(f" | {roi_ts_display}")
                            else:
                                self.logger.debug("[DEBUG] roi_ts_display не доступен")

                            # Добавляем время удержания, если оно рассчитано
                            if "time_display" in locals() and time_display:
                                status_msg.append(f" | {time_display}")
                            else:
                                self.logger.debug("[DEBUG] time_display не доступен")

                        except Exception as e:
                            self.logger.error(f"Ошибка при расчете ROI: {str(e)}")
                            self.logger.error(
                                f"[DEBUG] last_row['close']: {last_row['close']}, trade.open_rate: {getattr(trade, 'open_rate', 'N/A')}"
                            )
                else:
                    self.logger.debug(f"[DEBUG] Нет открытой позиции для {pair}")

                # Всегда добавляем информацию о покупателях/продавцах
                status_msg.append(f" | Покупатели: {buy_volume:.1f} | Продавцы: {sell_volume:.1f}")

                # Объединяем все части сообщения

                # Объединяем все части сообщения

            except Exception as e:
                self.logger.error(f"Критическая ошибка при формировании статуса: {str(e)}")
                status_msg = [
                    f"\n[{pair}] Цена: {last_row['close']:.2f} | ",
                    f"Позиция: {'ДА' if has_position else 'НЕТ'} | ",
                    f"Покупатели: {buy_volume:.1f} | Продавцы: {sell_volume:.1f}",
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
        Определяет условия для входа в позицию на основе анализа стакана и технических индикаторов
        """
        try:
            pair = metadata.get("pair")
            # Инициализируем переменную reason
            reason = "неизвестная причина"

            # Получаем последнюю свечу
            last_row = dataframe.iloc[-1]
            current_price = float(last_row.get("close", 0))

            # Получаем параметры стратегии
            min_volume = self._get_param_value("min_cluster_size")
            min_orders = self._get_param_value("min_orders_in_cluster")

            # Получаем данные о кластере покупок
            buy_cluster_price = last_row.get("buy_cluster_price", 0)
            buy_cluster_volume = last_row.get("buy_cluster_volume", 0)
            buy_cluster_orders = last_row.get(
                "buy_cluster_orders", 0
            )  # Количество ордеров в кластере

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
            # MACD фильтр
            """
            if self._get_param_value('macd_buy_enabled'):
                macd_value = last_row.get('macd', 0)
                macdsignal_value = last_row.get('macdsignal', 0)
                prev_macd = dataframe['macd'].iloc[-2] if len(dataframe) > 1 else None
                prev_macdsignal = dataframe['macdsignal'].iloc[-2] if len(dataframe) > 1 else None
                if prev_macd is not None and prev_macdsignal is not None:
                    macd_cross_up = prev_macd <= prev_macdsignal and macd_value > macdsignal_value
                else:
                    macd_cross_up = False
                check = '✓' if macd_cross_up else '✗'
                self.logger.info(f"MACD пересечение вверх (cross up): {macd_value:.2f} crosses up {macdsignal_value:.2f}: {check}")
                if not macd_cross_up:
                    enter_long = False
            else:
                self.logger.info("MACD BUY Фильтр: Отключен")
            """
            # RSI фильтр (отключен по умолчанию)
            rsi_filter_enabled = self._get_param_value("rsi_buy_enabled")
            if rsi_filter_enabled:
                rsi_value = last_row.get("rsi")
                rsi_threshold = self._get_param_value("rsi_buy_threshold")
                if rsi_value is None or rsi_value >= rsi_threshold:
                    enter_long = False

            # EMA фильтр (отключен по умолчанию)
            ema_buy_enabled = self._get_param_value("ema_buy_enabled")
            # Удалено дублирующее логирование EMA Trend
            # if ema_buy_enabled:
            #     ema_slow_value = last_row.get('ema_slow', 0)
            #     self.logger.info(f"EMA Trend ({current_price:.2f} > {ema_slow_value:.2f}): {'✓' if current_price > ema_slow_value else '✗'}")
            # else:
            #     self.logger.info("EMA Фильтр: Отключен")

            # ATR фильтр
            atr_filter_enabled = self._get_param_value("atr_filter_enabled")
            if atr_filter_enabled:
                atr_value = last_row.get("atr")
                min_vol_pct = self._get_param_value("atr_min_volatility_pct")
                if atr_value is None or (atr_value / current_price) * 100 < min_vol_pct:
                    enter_long = False

            # Логируем условия входа всегда (даже при наличии позиции)
            self.logger.info(f"-----[{pair}] УСЛОВИЯ ДЛЯ ПОКУПКИ:")
            # Удаляем лишние логи перед блоком условий для покупки:
            # self.logger.info(f"[{pair}] Цена: {current_price:.2f} | Позиция: {'ДА' if has_position else 'НЕТ'} | Покупатели: {buy_volume:.1f} | Продавцы: {sell_volume:.1f}")
            # self.logger.info(f"EMA Trend ({current_price:.2f} > {ema_slow_value:.2f}): {'✓' if current_price > ema_slow_value else '✗'}")
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
            # Удаляем старое логирование с номером
            # self.logger.info(f"6. Нет открытой позиции: {'✓' if conditions['no_position'] else '✗'}")
            self.logger.info(f"-> Сигнал по кластерам: {'ДА' if cluster_enter_long else 'НЕТ'}")

            # --- Фильтры для входа ---
            self.logger.info("--- Фильтры для входа ---")
            # MACD фильтр
            # if self._get_param_value('macd_buy_enabled'):
            #     macd_value = last_row.get('macd', 0)
            #     macdsignal_value = last_row.get('macdsignal', 0)
            #     res = macd_value > macdsignal_value
            # Закомментировать все старые вызовы логирования MACD BUY и MACD SELL:
            # self.log_filter_result('MACD BUY', macd_value, '>', macdsignal_value, res)
            #     if not macd_cross_up:
            #         enter_long = False
            # else:
            #     self.logger.info("MACD BUY Фильтр: Отключен")

            # RSI фильтр
            rsi_filter_enabled = self._get_param_value("rsi_buy_enabled")
            if rsi_filter_enabled:
                rsi_value = last_row.get("rsi", 0)
                rsi_threshold = self._get_param_value("rsi_buy_threshold")
                res = rsi_value < rsi_threshold
                self.log_filter_result("RSI BUY", rsi_value, "<", rsi_threshold, res)
            else:
                self.logger.info("RSI BUY Фильтр: Отключен")

            # EMA фильтр
            ema_buy_enabled = self._get_param_value("ema_buy_enabled")
            if ema_buy_enabled:
                ema_slow_value = last_row.get("ema_slow", 0)
                if ema_slow_value is None or pd.isna(ema_slow_value):
                    self.logger.warning(
                        "EMA значение не найдено или равно None, пропускаем EMA фильтр"
                    )
                else:
                    ema_condition = current_price > ema_slow_value
                    self.log_filter_result(
                        "EMA BUY Trend", current_price, ">", ema_slow_value, ema_condition
                    )
                    if not ema_condition:
                        enter_long = False
                        self.logger.debug(
                            f"EMA фильтр не пройден: цена {current_price:.8f} ниже EMA {ema_slow_value:.8f}"
                        )
                    else:
                        self.logger.info(
                            f"EMA фильтр пройден: цена {current_price:.8f} выше EMA {ema_slow_value:.8f}"
                        )
            else:
                self.logger.info("EMA BUY Фильтр: Отключен")

            # MACD фильтр
            if self._get_param_value("macd_buy_enabled"):
                macd_value = last_row.get("macd", 0)
                macdsignal_value = last_row.get("macdsignal", 0)
                prev_macd = dataframe["macd"].iloc[-2] if len(dataframe) > 1 else None
                prev_macdsignal = dataframe["macdsignal"].iloc[-2] if len(dataframe) > 1 else None
                if prev_macd is not None and prev_macdsignal is not None:
                    macd_cross_up = prev_macd <= prev_macdsignal and macd_value > macdsignal_value
                else:
                    macd_cross_up = False
                check = "✓" if macd_cross_up else "✗"
                self.logger.info(
                    f"MACD пересечение вверх (cross up): {macd_value:.2f} crosses up {macdsignal_value:.2f}: {check}"
                )
                if not macd_cross_up:
                    enter_long = False
            else:
                self.logger.info("MACD BUY Фильтр: Отключен")

            # ATR фильтр
            atr_filter_enabled = self._get_param_value("atr_filter_enabled")
            if atr_filter_enabled:
                atr_value = last_row.get("atr", 0)
                atr_pct = (atr_value / current_price) * 100 if current_price > 0 else 0
                min_vol_pct = self._get_param_value("atr_min_volatility_pct")
                res = atr_pct > min_vol_pct
                self.log_filter_result("ATR Volatility", atr_pct, ">", min_vol_pct, res)
            else:
                self.logger.info("ATR Фильтр: Отключен")

            # BBTrend фильтр
            bbtrend_buy_enabled = self._get_param_value("bbtrend_buy_enabled")
            if bbtrend_buy_enabled:
                bbtrend_value = last_row.get("bbtrend", 0)
                bbtrend_min = self._get_param_value("bbtrend_buy_min")
                res = bbtrend_value >= bbtrend_min
                self.log_filter_result("BBTrend BUY", bbtrend_value, ">=", bbtrend_min, res)
            else:
                self.logger.info("BBTrend BUY Фильтр: Отключен")

            # Логируем статус позиции после фильтров
            self.logger.info(f"Нет открытой позиции: {'✓' if not has_position else '✗'}")

            # --- Фильтры на рост и откат (Anti-pump, Pullback) ---
            if self._get_param_value("anti_pump_filter_enabled"):
                max_candle_growth = self._get_param_value("max_candle_growth")
                if len(dataframe) > 1:
                    prev_close = dataframe["close"].iloc[-2]
                    curr_close = dataframe["close"].iloc[-1]
                    growth = (curr_close - prev_close) / prev_close * 100
                    res = growth > max_candle_growth
                    self.log_filter_result(
                        "Anti-pump filter: рост свечи", growth, ">", max_candle_growth, res
                    )
                    if not res:
                        enter_long = False
            if self._get_param_value("three_bar_growth_filter_enabled"):
                max_three_bar_growth = self._get_param_value("max_three_bar_growth")
                if len(dataframe) > 3:
                    growth_3 = (
                        (dataframe["close"].iloc[-1] - dataframe["close"].iloc[-4])
                        / dataframe["close"].iloc[-4]
                        * 100
                    )
                    res = growth_3 > max_three_bar_growth
                    self.log_filter_result(
                        "Anti-pump filter (3 bars): рост за 3 свечи",
                        growth_3,
                        ">",
                        max_three_bar_growth,
                        res,
                    )
                    if not res:
                        enter_long = False
            if self._get_param_value("pullback_filter_enabled"):
                min_pullback = self._get_param_value("min_pullback")
                pullback_lookback = int(self._get_param_value("pullback_lookback"))
                if len(dataframe) > pullback_lookback:
                    recent_max = dataframe["high"].iloc[-pullback_lookback:-1].max()
                    curr_close = dataframe["close"].iloc[-1]
                    pullback = (recent_max - curr_close) / recent_max * 100
                    res = pullback > min_pullback
                    comment = "откат достаточный" if res else "слишком маленький откат"
                    self.log_filter_result(
                        "Pullback filter: откат", pullback, ">", min_pullback, res, comment
                    )
                    if not res:
                        enter_long = False

            # --- Итоговый результат ---
            if not has_position:
                if enter_long:
                    self.logger.info(f"***** ИТОГОВЫЙ РЕЗУЛЬТАТ: СИГНАЛ НА ПОКУПКУ! 🚀 *****")
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
                    self.logger.info(f"***** ИТОГОВЫЙ РЕЗУЛЬТАТ: ВХОД ЗАПРЕЩЕН *****")
            else:
                self.logger.info(f"***** ИТОГОВЫЙ РЕЗУЛЬТАТ: ВХОД ЗАПРЕЩЕН *****")

            # Устанавливаем сигналы
            dataframe.loc[:, "enter_long"] = 0
            dataframe.loc[dataframe.index[-1], "enter_long"] = 1 if enter_long else 0

            return dataframe

        except Exception as e:
            self.logger.error(f"Ошибка в populate_entry_trend: {e}", exc_info=True)
            return dataframe

    def _get_atr_sl_tp(self, trade, atr_value=None):
        """
        Возвращает уровни Stop-Loss и Take-Profit (ATR) для сделки
        """
        # Если есть сохранённые значения - используем их
        if hasattr(trade, "user_data") and trade.user_data:
            sl = trade.user_data.get("sl_atr")
            tp = trade.user_data.get("tp_atr")
            if sl is not None and tp is not None:
                return sl, tp
        # Fallback: старое поведение
        open_rate = trade.open_rate
        if atr_value is None:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            if dataframe.empty:
                return None, None
            atr_value = dataframe["atr"].iloc[-1]
        sl = open_rate - (atr_value * self._get_param_value("atr_stoploss_multiplier"))
        tp = open_rate + (atr_value * self._get_param_value("atr_takeprofit_multiplier"))
        return sl, tp

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определение сигналов на выход из позиции
        """
        pair = metadata.get("pair")
        if not pair or len(dataframe) == 0:
            return dataframe
        try:
            if "exit_long" not in dataframe.columns:
                dataframe["exit_long"] = 0
            last_candle = dataframe.iloc[-1].copy()
            volume_threshold = self._get_param_value("volume_threshold")
            if not isinstance(volume_threshold, (int, float)) or volume_threshold is None:
                logger.error(f"Некорректное значение volume_threshold: {volume_threshold}")
                volume_threshold = STRATEGY_PARAMS["volume_threshold"]
            required_columns = [
                "sell_cluster_volume",
                "sell_cluster_price",
                "close",
                "buy_cluster_volume",
            ]
            if not all(col in dataframe.columns for col in required_columns):
                logger.warning(f"Отсутствуют необходимые колонки в данных для {pair}")
                return dataframe
            sell_volume = pd.to_numeric(dataframe["sell_cluster_volume"], errors="coerce")
            sell_price = pd.to_numeric(dataframe["sell_cluster_price"], errors="coerce")
            close_price = pd.to_numeric(dataframe["close"], errors="coerce")
            buy_volume = pd.to_numeric(dataframe["buy_cluster_volume"], errors="coerce")
            cluster_exit_conditions = (
                (sell_volume > volume_threshold)
                & (close_price < sell_price)
                & (sell_volume > buy_volume)
            )
            cluster_exit_long = (
                cluster_exit_conditions.iloc[-1]
                if hasattr(cluster_exit_conditions, "iloc")
                else cluster_exit_conditions
            )
            exit_long = cluster_exit_long
            """
            if self._get_param_value('macd_sell_enabled'):
                macd_value = last_candle.get('macd', 0)
                macdsignal_value = last_candle.get('macdsignal', 0)
                prev_macd = dataframe['macd'].iloc[-2] if len(dataframe) > 1 else None
                prev_macdsignal = dataframe['macdsignal'].iloc[-2] if len(dataframe) > 1 else None
                if prev_macd is not None and prev_macdsignal is not None:
                    macd_cross_down = prev_macd >= prev_macdsignal and macd_value < macdsignal_value
                else:
                    macd_cross_down = False
                check = '✓' if macd_cross_down else '✗'
                self.logger.info(f"MACD пересечение вниз (cross down): {macd_value:.2f} crosses down {macdsignal_value:.2f}: {check}")
                if not macd_cross_down:
                    exit_long = False
            else:
                self.logger.info("MACD SELL Фильтр: Отключен")
            """
            rsi_filter_enabled = self._get_param_value("rsi_sell_enabled")
            rsi_value = last_candle.get("rsi")
            rsi_threshold = self._get_param_value("rsi_sell_threshold")
            rsi_condition = rsi_value > rsi_threshold if rsi_value is not None else False
            if cluster_exit_long and rsi_filter_enabled and not rsi_condition:
                exit_long = False
            if self._has_open_position(pair):
                current_price = float(last_candle.get("close", 0))
                # Получаем открытую сделку
                trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
                trade = trades[0] if trades else None

                # Обновляем трейлинг-стоп
                if trade:
                    try:
                        # Получаем время последней свечи
                        current_time = dataframe["date"].iloc[-1].to_pydatetime()
                        # Вычисляем текущую прибыль
                        current_profit = trade.calc_profit_ratio(current_price)
                        self.adjust_trailing_stop(
                            trade, current_time, current_price, current_profit
                        )
                    except Exception as e:
                        self.logger.error(f"Ошибка при обновлении трейлинг-стопа: {e}")
                        self.logger.debug(
                            f"Трейд: {trade}, current_price: {current_price}", exc_info=True
                        )
                atr_value = last_candle.get("atr", None)
                sl, tp = (None, None)
                if trade and atr_value:
                    sl, tp = self._get_atr_sl_tp(trade, atr_value)
                self.logger.info(f"-----[{pair}] УСЛОВИЯ ДЛЯ ПРОДАЖИ:")
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
                self.logger.info(f"-> Сигнал по кластерам: {'ДА' if cluster_exit_long else 'НЕТ'}")

                # --- Фильтры для выхода ---
                self.logger.info("--- Фильтры для выхода ---")

                # RSI фильтр
                rsi_filter_enabled = self._get_param_value("rsi_sell_enabled")
                if rsi_filter_enabled:
                    rsi_value = last_candle.get("rsi", 0)
                    rsi_threshold = self._get_param_value("rsi_sell_threshold")
                    res = rsi_value > rsi_threshold
                    self.log_filter_result("RSI SELL", rsi_value, ">", rsi_threshold, res)
                else:
                    self.logger.info("RSI SELL Фильтр: Отключен")

                # EMA фильтр
                ema_sell_enabled = self._get_param_value("ema_sell_enabled")
                if ema_sell_enabled:
                    ema_slow_value = last_candle.get("ema_slow", 0)
                    current_price = float(last_candle.get("close", 0))
                    res = current_price < ema_slow_value
                    self.log_filter_result(
                        "EMA SELL Trend", current_price, "<", ema_slow_value, res
                    )
                else:
                    self.logger.info("EMA SELL Фильтр: Отключен")

                # MACD фильтр
                """
                if self._get_param_value('macd_sell_enabled'):
                    macd_value = last_candle.get('macd', 0)
                    macdsignal_value = last_candle.get('macdsignal', 0)
                    res = macd_value < macdsignal_value
                    if not macd_cross_down:
                        exit_long = False
                else:
                    self.logger.info("MACD SELL Фильтр: Отключен")
                """
                if self._get_param_value("macd_sell_enabled"):
                    macd_value = last_candle.get("macd", 0)
                    macdsignal_value = last_candle.get("macdsignal", 0)
                    prev_macd = dataframe["macd"].iloc[-2] if len(dataframe) > 1 else None
                    prev_macdsignal = (
                        dataframe["macdsignal"].iloc[-2] if len(dataframe) > 1 else None
                    )
                    if prev_macd is not None and prev_macdsignal is not None:
                        macd_cross_down = (
                            prev_macd >= prev_macdsignal and macd_value < macdsignal_value
                        )
                    else:
                        macd_cross_down = False
                    check = "✓" if macd_cross_down else "✗"
                    self.logger.info(
                        f"MACD пересечение вниз (cross down): {macd_value:.2f} crosses down {macdsignal_value:.2f}: {check}"
                    )
                    if not macd_cross_down:
                        exit_long = False
                else:
                    self.logger.info("MACD SELL Фильтр: Отключен")

                # ATR фильтр
                atr_filter_enabled = self._get_param_value("atr_filter_enabled")
                if atr_filter_enabled:
                    atr_value = last_candle.get("atr", 0)
                    atr_pct = (atr_value / current_price) * 100 if current_price > 0 else 0
                    min_vol_pct = self._get_param_value("atr_min_volatility_pct")
                    res = atr_pct > min_vol_pct
                    self.log_filter_result("ATR Volatility", atr_pct, ">", min_vol_pct, res)
                else:
                    self.logger.info("ATR Фильтр: Отключен")

                # BBTrend фильтр
                bbtrend_sell_enabled = self._get_param_value("bbtrend_sell_enabled")
                if bbtrend_sell_enabled:
                    bbtrend_value = last_candle.get("bbtrend", 0)
                    bbtrend_max = self._get_param_value("bbtrend_sell_max")
                    res = bbtrend_value <= bbtrend_max
                    self.log_filter_result("BBTrend SELL", bbtrend_value, "<=", bbtrend_max, res)
                else:
                    self.logger.info("BBTrend SELL Фильтр: Отключен")

                # Логируем статус позиции после фильтров
                self.logger.info(
                    f"Есть открытая позиция: {'✓' if self._has_open_position(pair) else '✗'}"
                )

                # --- Итоговый результат ---
                self.logger.info(f"***** ИТОГОВЫЙ РЕЗУЛЬТАТ: ВЫХОД ЗАПРЕЩЕН *****")

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
            bool: True если ROI >= нужного порога, иначе False
        """
        try:
            trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
            if not trades:
                return False

            # Берем первую открытую сделку (должна быть только одна)
            trade = trades[0]

            # Рассчитываем текущий ROI
            profit_ratio = trade.calc_profit_ratio(current_price)

            # Если включён ATR-тейк-профит - сравниваем с atr_takeprofit_min_pct
            if self._get_param_value("atr_takeprofit_enabled"):
                tp_min_pct = self._get_param_value("atr_takeprofit_min_pct")
                return profit_ratio >= tp_min_pct

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

            orderbook = exchange.fetch_l2_order_book(pair, 10)

            if side == "long":
                # Для длинной позиции ставим ордер чуть выше кластера покупок
                return proposed_rate * 1.0005  # На 0.05% выше
            else:
                # Для короткой позиции ставим ордер чуть ниже кластера продаж
                return proposed_rate * 0.9995  # На 0.05% ниже

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

        # --- Логирование параметров ATR стоп-лосса ---
        self.logger.info(
            f"ATR STOPLOSS | pair={pair} | open_rate={trade.open_rate:.4f} | ATR={atr_value:.6f} | multiplier={multiplier} | stop_price={stop_price:.4f} | stop_loss_pct={stop_loss_pct:.4%}"
        )
        return stop_loss_pct

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        # Если позиция не открыта, пропускаем выполнение
        if not trade or not trade.is_open or getattr(trade, "is_closed", False):
            self.logger.debug(
                f"[ВЫХОД] Позиция {pair} не открыта или закрыта, пропуск проверок выхода"
            )
            return None

        # Логируем отладочную информацию только в режиме DEBUG
        self.logger.debug(f"custom_exit для {pair}. Текущий профит: {current_profit * 100:.2f}%")
        self.logger.debug(
            f"Текущий roi_for_trailing: {getattr(self, 'roi_for_trailing', 'не установлен')}"
        )
        self.logger.debug(
            f"Текущий trailing_stop_positive_offset: {getattr(self, 'trailing_stop_positive_offset', 'не установлен')}"
        )

        # Обновляем трейлинг-стоп на основе времени удержания позиции
        self.adjust_trailing_stop(trade, current_time, current_rate, current_profit)

        """
        Дополнительные условия для выхода из позиции:
        - Если цена <= ATR SL - выход по стоп-лоссу (только если atr_stoploss_enabled)
        - Если цена >= ATR TP и прибыль >= atr_takeprofit_min_pct - выход по тейк-профиту (только если atr_takeprofit_enabled)
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return None
            atr_value = dataframe["atr"].iloc[-1]
            if not atr_value or pd.isna(atr_value):
                return None
            sl_multiplier = self._get_param_value("atr_stoploss_multiplier")
            tp_multiplier = self._get_param_value("atr_takeprofit_multiplier")
            tp_min_pct = self._get_param_value("atr_takeprofit_min_pct")
            sl = trade.open_rate - (atr_value * sl_multiplier)
            tp = trade.open_rate + (atr_value * tp_multiplier)
            # --- Выход по SL ---
            if self._get_param_value("atr_stoploss_enabled") and current_rate <= sl:
                self.logger.info(
                    f"ATR Stop-Loss для {pair}: цена {current_rate:.2f} <= SL {sl:.2f} (ATR: {atr_value:.8f})"
                )
                return "atr_stoploss"
            # --- Выход по TP ---
            if self._get_param_value("atr_takeprofit_enabled") and current_rate >= tp:
                profit_pct = (current_rate / trade.open_rate - 1) * 100
                if profit_pct >= tp_min_pct:
                    self.logger.info(
                        f"ATR Take-Profit для {pair}: цена {current_rate:.2f} >= TP {tp:.2f}, прибыль {profit_pct:.2f}% >= min {tp_min_pct:.2f}% (ATR: {atr_value:.8f})"
                    )
                    return "atr_takeprofit"
            return None
        except Exception as e:
            self.logger.error(f"Error in custom_exit for {pair}: {str(e)}", exc_info=True)
            return None

    def _analyze_orderbook_balance(self, orderbook):
        """Анализ баланса покупателей и продавцов"""
        total_buy_volume = sum(volume for _, volume in orderbook["bids"][:10])
        total_sell_volume = sum(volume for _, volume in orderbook["asks"][:10])

        return {
            "buy_volume": total_buy_volume,
            "sell_volume": total_sell_volume,
            "ratio": total_buy_volume / total_sell_volume if total_sell_volume > 0 else 1.0,
        }

    def _calculate_rsi_tradingview(self, prices, period=14):
        """
        RSI по формуле TradingView: первые period - SMA, далее Wilder's Smoothing
        """
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rsi = pd.Series(index=prices.index, dtype=float)
            for i in range(len(prices)):
                if i < period:
                    rsi.iloc[i] = float("nan")
                elif i == period:
                    rs = avg_gain.iloc[i] / avg_loss.iloc[i] if avg_loss.iloc[i] != 0 else 0
                    rsi.iloc[i] = 100 - (100 / (1 + rs))
                    prev_avg_gain = avg_gain.iloc[i]
                    prev_avg_loss = avg_loss.iloc[i]
                else:
                    curr_gain = gain.iloc[i]
                    curr_loss = loss.iloc[i]
                    prev_avg_gain = (prev_avg_gain * (period - 1) + curr_gain) / period
                    prev_avg_loss = (prev_avg_loss * (period - 1) + curr_loss) / period
                    rs = prev_avg_gain / prev_avg_loss if prev_avg_loss != 0 else 0
                    rsi.iloc[i] = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"Ошибка при расчёте RSI TradingView: {e}")
            return pd.Series([float("nan")] * len(prices), index=prices.index)

    def _validate_parameters(self):
        """Валидация критических параметров"""
        critical_params = [
            "min_cluster_size",
            "cluster_price_threshold",
            "min_orders_in_cluster",
            "stoploss",
        ]

        for param in critical_params:
            value = self._get_param_value(param)
            if param == "stoploss":
                # Для stoploss разрешаем отрицательные значения
                if value is None or not isinstance(value, (int, float)):
                    self.logger.error(f"Некорректное значение параметра {param}: {value}")
                    return False
            else:
                if value is None or (isinstance(value, (int, float)) and value <= 0):
                    self.logger.error(f"Некорректное значение параметра {param}: {value}")
                    return False

        self.logger.info("✅ Валидация параметров прошла успешно")
        return True

    # --- ДОБАВИТЬ: Сохранять ATR, SL, TP при открытии сделки ---
    def on_trade_entry(self, trade, last_candle):
        """
        Сохраняет ATR, SL, TP в user_data сделки при открытии
        """
        atr_value = last_candle.get("atr", None)
        open_rate = trade.open_rate
        if atr_value is not None:
            sl = open_rate - (atr_value * self._get_param_value("atr_stoploss_multiplier"))
            tp = open_rate + (atr_value * self._get_param_value("atr_takeprofit_multiplier"))
            # Сохраняем в user_data (или через setattr)
            if not hasattr(trade, "user_data") or trade.user_data is None:
                trade.user_data = {}
            trade.user_data["atr_entry"] = float(atr_value)
            trade.user_data["sl_atr"] = float(sl)
            trade.user_data["tp_atr"] = float(tp)

    # --- Добавить хук on_trade для freqtrade ---
    def on_trade(self, trade, order, pair, is_entry, **kwargs):
        """
        Хук Freqtrade: вызывается при любом изменении сделки (entry_fill, exit_fill и т.д.)
        Сохраняет ATR, SL, TP при открытии сделки.
        """
        if is_entry and order.status.name == "closed":
            # Получаем последнюю свечу для пары
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                last_candle = dataframe.iloc[-1]
                self.on_trade_entry(trade, last_candle)

    def log_filter_result(self, name, param, op, threshold, result, comment=None):
        """
        Универсальное логирование результата фильтра.
        name: название фильтра (строка)
        param: текущее значение
        op: знак сравнения (строка: '>', '<', '>=', '<=', '==', '!=')
        threshold: пороговое значение
        result: True (✓) или False (✗)
        comment: опциональный комментарий
        """
        check = "✓" if result else "✗"
        msg = f"{name}: {param:.2f} {op} {threshold}: {check}"
        if comment:
            msg += f" ({comment})"
        self.logger.info(msg)
