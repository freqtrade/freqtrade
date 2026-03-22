# Order Book Imbalance Scalping Strategy

import asyncio
import logging
import os
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
from ccxt.base.types import OrderBook
from dotenv import load_dotenv
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy


# Загружаем переменные среды из .env файла
load_dotenv()

logger = logging.getLogger(__name__)


# Используемые индикаторы для фильтра
def rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def ema(prices, period):
    """Calculate EMA (Exponential Moving Average)"""
    return prices.ewm(span=period, adjust=False).mean()


def atr(high, low, close, period=14):
    """Calculate ATR (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


class StrategyDCA(IStrategy):
    """
    Стратегия скальпинга на основе дисбаланса в книге ордеров.
    Анализирует соотношение между ордерами на покупку и продажу для
    определения краткосрочных движений цены.
    + DCA (динамическая коррекция позиции)
    """

    # Настройки стратегии
    minimal_roi = {"0": 0.007}  # 0.55% прибыли достаточно для скальпинга
    stoploss = -0.015  # Стоп-лосс 1.5%

    # Оптимальный порядок для заполнения и тестирования
    startup_candle_count = 100  # Увеличиваем количество стартовых свечей для лучшей инициализации
    process_only_new_candles = True

    # Параметры для настройки
    order_book_depth: int = 10
    buy_imbalance_threshold: float = 1.25  # 1.5  1.2
    sell_imbalance_threshold: float = 0.5  # 0.35
    volume_threshold: float = 0.4  # поукупка 0.39
    volume_sell_threshold: float = 0.8  # продажа 0.9

    # Дополнительные индикаторы для фильтрации сигналов
    use_rsi_filter: bool = True
    rsi_buy_threshold: float = 40  # 30
    rsi_sell_threshold: float = 65  # 70

    # Параметры для анализа тренда
    use_ema_filter: bool = True
    trend_ema_fast = 9  # 9
    trend_ema_slow = 21  # 21
    trend_ema_signal = 50  # 50
    trend_strength_threshold: float = 0.0005  # 0,0005

    # Параметры для анализа объема
    volume_ma_period = 20  # Период для скользящей средней объема

    # Кэш для хранения данных книги ордеров
    orderbook_cache = {}
    last_analysis_time = {}
    exchange_instance = None

    # Параметры для обновления кэша книги ордеров
    orderbook_refresh_period = 1  # Увеличиваем задержку обновления до 1 секунды

    # Настройки трейлинг-стопа
    trailing_stop = True
    trailing_stop_positive = 0.003  # Уменьшаем активацию трейлинг-стопа до 0.001(0.1%) прибыли
    trailing_stop_positive_offset = 0.0035  # Уменьшаем смещение до 0.0015(0.15%)
    trailing_only_offset_is_reached = True  # Активация только после достижения смещения

    # DCA
    # Настройки для DCA (динамической коррекции позиции)
    position_adjustment_enable = True
    max_entry_position_adjustment = 2  # Максимум 2 дополнительные закупки
    min_dca_time = 3  # Минимальный промежуток между докупками (в минутах)
    # End DCA

    # Параметры для управления рисками
    risk_per_trade = 0.02  # 2% риска на сделку
    atr_period = 14  # Период для расчета ATR
    atr_multiplier = 2.0  # Множитель для расчета стоп-лосса на основе ATR

    # Настройки для оптимизации производительности
    _last_ohlcv_update = {}
    _ohlcv_update_interval = 1  # секунды между обновлениями OHLCV
    _last_orderbook_update = {}
    _orderbook_update_interval = 1  # секунды между обновлениями стакана

    # Условия для покупки и продажи
    buy_conditions = [
        "imbalance_ratio > buy_imbalance_threshold",
        "volume > volume_threshold",
        "rsi < rsi_buy_threshold",
        "ema_fast > ema_slow",
        "trend_strength > trend_strength_threshold",  # Как вычисляется сила тренда?
    ]

    sell_conditions = [
        "imbalance_ratio < sell_imbalance_threshold",
        "rsi > rsi_sell_threshold",
        "ema_fast < ema_slow",
        "volume > volume_sell_threshold",
    ]

    def __init__(self, config: dict):
        super().__init__(config)

        # Получаем timeframe из конфига
        self.timeframe = config.get("timeframe", "5m")
        logger.info(f"Using timeframe from config: {self.timeframe}")

        # Инициализация exchange с использованием переменных среды  # noqa: RUF003
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_API_SECRET")

        if api_key is None or secret_key is None:
            raise ValueError(
                "Переменные окружения BINANCE_API_KEY и BINANCE_API_SECRET должны быть установлены."
            )

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
        try:
            logger.info("Loading markets...")
            self.exchange_instance.load_markets()
            logger.info("Markets loaded successfully")
        except Exception as e:
            logger.error(f"Error during initial markets loading: {str(e)}")
            self.exchange_instance = None

        # Инициализация асинхронных компонентов
        self._shutdown_event = None
        self._running_tasks = set()

        # Инициализация кэша книги ордеров
        self._orderbook_cache = {}
        self._orderbook_cache_time = {}
        self._orderbook_cache_ttl = 1  # Время жизни кэша в секундах

        # Инициализация условий для покупки и продажи
        self.buy_conditions = [
            "imbalance_ratio > buy_imbalance_threshold",
            "volume > volume_threshold",
            "rsi < rsi_buy_threshold",
            "ema_fast > ema_slow",
            "trend_strength > trend_strength_threshold",
        ]
        self.sell_conditions = [
            "imbalance_ratio < sell_imbalance_threshold",
            "rsi > rsi_sell_threshold",
            "ema_fast < ema_slow",
            "volume > volume_sell_threshold",
        ]

    async def shutdown(self):
        """Корректное завершение работы стратегии"""
        if self._shutdown_event:
            self._shutdown_event.set()
            for task in self._running_tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*self._running_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

    def informative_pairs(self):
        """
        Определяет информативные пары для получения дополнительных данных
        """
        return []  # не используем информативные пары

    def get_order_book(self, pair: str, depth: int = 10) -> OrderBook | None:
        """
        Получает данные книги ордеров для указанной пары с кэшированием
        """  # noqa: RUF002
        current_time = datetime.now(timezone.utc)

        # Проверяем кэш
        if pair in self._orderbook_cache:
            cache_time = self._orderbook_cache_time.get(pair)
            if (
                cache_time
                and (current_time - cache_time).total_seconds() < self._orderbook_cache_ttl
            ):
                return self._orderbook_cache[pair]

        try:
            # Проверяем инициализацию exchange
            if not self.exchange_instance:
                logger.error("Exchange instance is not initialized")
                return None

            # Получаем новые данные
            orderbook = self.exchange_instance.fetch_order_book(pair, depth)
            if (
                not isinstance(orderbook, dict)
                or "bids" not in orderbook
                or "asks" not in orderbook
            ):
                logger.error(f"Некорректная структура книги ордеров для {pair}")
                return None

            # Обновляем кэш
            self._orderbook_cache[pair] = orderbook
            self._orderbook_cache_time[pair] = current_time

            return orderbook
        except Exception as e:
            logger.error(f"Ошибка при получении книги ордеров для {pair}: {str(e)}")
            return None

    def calculate_order_book_imbalance(self, pair: str) -> float:
        """
        Рассчитывает коэффициент дисбаланса в книге ордеров с кэшированием
        """  # noqa: RUF002
        current_time = datetime.now(timezone.utc)

        # Проверяем кэш
        if pair in self._orderbook_cache:
            cache_time = self._orderbook_cache_time.get(pair)
            if (
                cache_time
                and (current_time - cache_time).total_seconds() < self._orderbook_cache_ttl
            ):
                orderbook = self._orderbook_cache[pair]
                # Рассчитываем дисбаланс из кэшированных данных
                bids_volume = sum(float(str(bid[1])) for bid in orderbook["bids"])
                asks_volume = sum(float(str(ask[1])) for ask in orderbook["asks"])
                if asks_volume > 0:
                    return bids_volume / asks_volume
                return 1.0

        try:
            # Проверяем инициализацию exchange
            if not self.exchange_instance:
                logger.error("Exchange instance is not initialized")
                return 1.0

            # Получаем новые данные
            orderbook = self.exchange_instance.fetch_order_book(pair, self.order_book_depth)
            if (
                not isinstance(orderbook, dict)
                or "bids" not in orderbook
                or "asks" not in orderbook
            ):
                logger.error(f"Некорректная структура книги ордеров для {pair}")
                return 1.0

            # Подробное логирование каждого ордера
            logger.debug(f"Detailed orderbook for {pair}:")
            logger.debug("Bids (buy orders):")
            for i, bid in enumerate(orderbook["bids"]):
                price, volume = float(str(bid[0])), float(str(bid[1]))
                logger.debug(f"  {i + 1}. Price: {price}, Volume: {volume}")

            logger.debug("Asks (sell orders):")
            for i, ask in enumerate(orderbook["asks"]):
                price, volume = float(str(ask[0])), float(str(ask[1]))
                logger.debug(f"  {i + 1}. Price: {price}, Volume: {volume}")

            # Получаем объемы на покупку и продажу
            bids_volume = sum(float(str(bid[1])) for bid in orderbook["bids"])
            asks_volume = sum(float(str(ask[1])) for ask in orderbook["asks"])

            # Логируем общие объемы
            logger.debug(f"Summary for {pair}:")
            logger.debug(f"  - Total bids volume: {bids_volume}")
            logger.debug(f"  - Total asks volume: {asks_volume}")

            # Рассчитываем коэффициент дисбаланса
            if asks_volume > 0:
                imbalance_ratio = bids_volume / asks_volume
            else:
                imbalance_ratio = 1.0
                logger.debug("  - Asks volume is 0, using default imbalance ratio: 1.0")

            # Обновляем кэш
            self._orderbook_cache[pair] = orderbook
            self._orderbook_cache_time[pair] = current_time

            return imbalance_ratio
        except Exception as e:
            logger.error(f"Ошибка при расчете дисбаланса для {pair}: {str(e)}")
            return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # Проверяем необходимость обновления данных
            current_time = datetime.now(timezone.utc)
            pair = metadata["pair"]

            # Проверяем время последнего обновления OHLCV
            if pair in self._last_ohlcv_update:
                last_update = self._last_ohlcv_update[pair]
                if (current_time - last_update).total_seconds() < self._ohlcv_update_interval:
                    # Возвращаем существующие данные, если обновление не требуется
                    return dataframe

            # Создаем копию DataFrame c сохранением индексов
            df = dataframe.copy(deep=True)

            # Инициализируем все колонки индикаторов
            indicator_columns = [
                "rsi",
                "ema_fast",
                "ema_slow",
                "ema_signal",
                "trend_strength",
                "volume_ma",
                "volume_ratio",
                "atr",
            ]
            for col in indicator_columns:
                if col not in df.columns:
                    df[col] = np.nan

            # Получаем и обрабатываем данные стакана
            try:
                # Проверяем необходимость обновления стакана
                if pair in self._last_orderbook_update:
                    last_update = self._last_orderbook_update[pair]
                    if (
                        current_time - last_update
                    ).total_seconds() < self._orderbook_update_interval:
                        # Используем кэшированные данные
                        if pair in self.orderbook_cache:
                            imbalance_ratio = self.calculate_order_book_imbalance(pair)
                            df["orderbook_imbalance"] = imbalance_ratio
                            return df

                # Обновляем данные стакана
                orderbook = self.get_order_book(pair, self.order_book_depth)
                if orderbook:
                    imbalance_ratio = self.calculate_order_book_imbalance(pair)
                    df["orderbook_imbalance"] = imbalance_ratio
                    self.orderbook_cache[pair] = orderbook
                    self._last_orderbook_update[pair] = current_time
                    logger.info(f"Current orderbook imbalance for {pair}: {imbalance_ratio}")
            except Exception as e:
                logger.error(f"Error processing orderbook: {str(e)}")

            # Рассчитываем индикаторы только если достаточно данных
            if len(df) >= 14:
                # Используем наши собственные индикаторы
                df["rsi"] = rsi(df["close"], period=14)
                df["ema_fast"] = ema(df["close"], period=self.trend_ema_fast)
                df["ema_slow"] = ema(df["close"], period=self.trend_ema_slow)
                df["ema_signal"] = ema(df["close"], period=self.trend_ema_signal)
                df["trend_strength"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
                df["volume_ma"] = df["volume"].rolling(window=10).mean()
                df["volume_ratio"] = df["volume"] / df["volume_ma"]
                df["atr"] = atr(df["high"], df["low"], df["close"], period=14)

            # Обновляем время последнего обновления OHLCV
            self._last_ohlcv_update[pair] = current_time

            return df

        except Exception as e:
            logger.error(f"Error in populate_indicators: {str(e)}")
            return dataframe

    """
    DCA
    """

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ):
        #  DCA - Динамическая корректировка размера позиции
        # Переменные для определения размера корректировки
        current_entries = trade.nr_of_successful_entries

        # ======= ДОКУПКА НА ПАДЕНИИ =======  # noqa: RUF003
        # 1-я докупка: при падении на 0.8%
        # 2-я докупка: при падении на 1.5% от средней цены

        # Рассчитываем уровни для докупки в зависимости от количества уже сделанных входов
        dca_levels = {0: -0.008, 1: -0.015}  # Первая докупка при -0.8%  # Вторая докупка при -1.5%

        # Если текущий профит ниже нашего уровня и не превышено макс. кол-во входов
        if current_entries < self.max_entry_position_adjustment:
            # Проверяем, соответствует ли текущая просадка нашему уровню для докупки
            if current_profit <= dca_levels.get(current_entries, -999):
                # Увеличиваем размер каждой последующей докупки
                # Используем мартингейл c коэффициентом 1.5
                multiplier = 1.5
                return trade.stake_amount * (multiplier**current_entries)

        # ======= ЧАСТИЧНАЯ ФИКСАЦИЯ ПРИБЫЛИ =======
        # Частичная фиксация при достижении прибыли:
        # +0.5% - продаем 30% позиции
        # +1.0% - продаем еще 30% позиции

        # Сначала проверяем, что у нас нет ожидающих ордеров (чтобы не создавать их снова)
        if not trade.has_open_orders:
            if current_profit >= 0.01:
                # При прибыли +1% фиксируем 30% позиции
                return -(trade.amount * 0.3)
            elif current_profit >= 0.005:
                # При прибыли +0.5% фиксируем 30% позиции
                return -(trade.amount * 0.3)

        # По умолчанию не корректируем позицию
        return None

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Настройка цены для входа в позицию
        Для скальпинга важно входить по наилучшей возможной цене
        """
        # Для DCA ордеров используем лимитные ордера чуть ниже рынка
        if entry_tag and "DCA" in entry_tag:
            if "DCA_entry_1" == entry_tag:
                # на 0.1% ниже текущей цены
                return proposed_rate * 0.999
            elif "DCA_entry_2" == entry_tag:
                # на 0.15% ниже текущей цены
                return proposed_rate * 0.9985
            else:
                # Для других DCA входов используем цену на 0.1% ниже
                return proposed_rate * 0.999
        # Для первоначального входа используем предложенную цену
        return proposed_rate

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: str | None,
        **kwargs,
    ) -> float:
        """
        Настройка цены для частичного или полного выхода из позиции
        """
        # для частичной фиксации используем лимитные ордера чуть выше рынка
        if exit_tag and "partial" in exit_tag:
            if exit_tag == "partial_exit_1":
                # на 0.1% выше текущей цены
                return proposed_rate * 1.001
            elif exit_tag == "partial_exit_2":
                # на 0.15% выше текущей цены
                return proposed_rate * 1.0015
            else:
                # для других частичных выходов используем цену на 0.1% выше
                return proposed_rate * 1.001

        # для финального выхода используем предложенную цену
        return proposed_rate

    """
    END DCA
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определяет условия для входа в позицию
        """
        pair = metadata["pair"]
        logger.info(f"=== BUY CONDITIONS FOR {pair}: ===")
        current_price = dataframe["close"].iloc[-1]
        logger.info(f"  - Current buy price: {current_price}")

        buy_imbalance = self.calculate_order_book_imbalance(pair)
        rsi_val = dataframe["rsi"].iloc[-1]
        volume_ratio = dataframe["volume_ratio"].iloc[-1]
        trend_strength = dataframe["trend_strength"].iloc[-1]
        ema_fast = dataframe["ema_fast"].iloc[-1]
        ema_slow = dataframe["ema_slow"].iloc[-1]
        ema_signal = dataframe["ema_signal"].iloc[-1]

        # Условия
        condition_imbalance = buy_imbalance > self.buy_imbalance_threshold
        condition_volume = volume_ratio > self.volume_threshold
        condition_rsi = rsi_val < self.rsi_buy_threshold if self.use_rsi_filter else True
        condition_ema_fast = ema_fast > ema_slow
        condition_ema_slow = ema_slow > ema_signal
        condition_trend = (
            trend_strength > self.trend_strength_threshold if self.use_ema_filter else True
        )

        logger.info(
            f"  - Imbalance ratio: {buy_imbalance} "
            f"(threshold: {self.buy_imbalance_threshold}, "
            f"condition: {condition_imbalance})"
        )
        logger.info(
            f"  - RSI: {rsi_val} (threshold: {self.rsi_buy_threshold}, condition: {condition_rsi})"
        )
        logger.info(
            f"  - Volume ratio: {volume_ratio} (threshold: {self.volume_threshold}, condition: {condition_volume})"  # noqa: E501
        )
        logger.info(
            f"  - Trend strength: {trend_strength} "
            f"(threshold: {self.trend_strength_threshold}, "
            f"condition: {condition_trend})"
        )
        logger.info(f"  - EMA Fast: {ema_fast}")
        logger.info(f"  - EMA Slow: {ema_slow}")
        logger.info(f"  - EMA condition: {ema_fast} > {ema_slow}")

        conditions = [
            condition_imbalance,
            condition_volume,
            condition_rsi,
            condition_ema_fast,
            condition_ema_slow,
            condition_trend,
        ]
        final_condition = all(conditions)
        if final_condition:
            dataframe.loc[dataframe.index[-1], "buy"] = 1
            logger.info("=== END BUY CONDITIONS CHECK ===")
        else:
            logger.info(f"No buy signal for {pair} - Some conditions failed")
            logger.info("Failed conditions:")
            if not condition_imbalance:
                logger.info(
                    f"  - Imbalance ratio too low: {buy_imbalance} <= {self.buy_imbalance_threshold}"
                )
            if not condition_volume:
                logger.info(f"  - Volume ratio too low: {volume_ratio} <= {self.volume_threshold}")
            if self.use_rsi_filter and not condition_rsi:
                logger.info(f"  - RSI too high: {rsi_val} >= {self.rsi_buy_threshold}")
            if not condition_ema_fast:
                logger.info(f"  - EMA Fast <= Slow: {ema_fast} <= {ema_slow}")
            if not condition_ema_slow:
                logger.info(f"  - EMA Slow <= Signal: {ema_slow} <= {ema_signal}")
            if not condition_trend:
                logger.info(
                    f"  - Trend strength too low: {trend_strength} <= {self.trend_strength_threshold}"
                )
            logger.info("=== END BUY CONDITIONS CHECK ===")
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Определяет условия для выхода из позиции
        """
        pair = metadata["pair"]
        logger.info(f"=== SELL CONDITIONS FOR {pair}: ===")
        current_price = dataframe["close"].iloc[-1]
        logger.info(f"  - Current sell price: {current_price}")

        sell_imbalance = self.calculate_order_book_imbalance(pair)
        rsi_val = dataframe["rsi"].iloc[-1]
        volume_ratio = dataframe["volume_ratio"].iloc[-1]
        trend_strength = dataframe["trend_strength"].iloc[-1]
        ema_fast = dataframe["ema_fast"].iloc[-1]
        ema_slow = dataframe["ema_slow"].iloc[-1]
        # ema_signal = dataframe['ema_signal'].iloc[-1]

        # Условия
        condition_imbalance = sell_imbalance < self.sell_imbalance_threshold
        condition_rsi = rsi_val > self.rsi_sell_threshold if self.use_rsi_filter else True
        condition_ema = ema_fast < ema_slow * 1.001 if self.use_ema_filter else True
        condition_volume = volume_ratio > self.volume_sell_threshold
        condition_trend = (
            trend_strength < self.trend_strength_threshold if self.use_ema_filter else True
        )

        logger.info(
            f"  - Imbalance ratio: {sell_imbalance} (threshold: {self.sell_imbalance_threshold}, condition: {condition_imbalance})"
        )
        logger.info(
            f"  - RSI: {rsi_val} (threshold: {self.rsi_sell_threshold}, condition: {condition_rsi})"
        )
        logger.info(
            f"  - Volume ratio: {volume_ratio} (threshold: {self.volume_sell_threshold}, condition: {condition_volume})"
        )
        logger.info(f"  - EMA Fast: {ema_fast}")
        logger.info(f"  - EMA Slow: {ema_slow}")
        logger.info(f"  - EMA condition: {ema_fast} < {ema_slow * 1.001}")
        logger.info(
            f"  - Trend strength: {trend_strength} "
            f"(threshold: {self.trend_strength_threshold}, "
            f"condition: {condition_trend})"
        )

        conditions = [
            condition_imbalance,
            condition_rsi,
            condition_ema,
            condition_volume,
            condition_trend,
        ]
        final_condition = all(conditions)
        # Проверяем текущую прибыль
        current_profit = (dataframe["close"].iloc[-1] - dataframe["open"].iloc[0]) / dataframe[
            "open"
        ].iloc[0]
        minimal_profit = 0.007  # Фиксированное значение минимальной ROI 0.7%

        if final_condition and current_profit >= minimal_profit:
            dataframe.loc[dataframe.index[-1], "sell"] = 1
            logger.info("=== END SELL CONDITIONS CHECK ===")
        else:
            logger.info(f"No sell signal for {pair} - Some conditions failed")
            logger.info("Failed conditions:")
            if current_profit < minimal_profit:
                logger.info(
                    f"  - Current profit {current_profit:.4%} < Minimal ROI {minimal_profit:.4%}"
                )
            if not condition_imbalance:
                logger.info(
                    f"  - Imbalance ratio too high: {sell_imbalance} >= {self.sell_imbalance_threshold}"
                )
            if self.use_rsi_filter and not condition_rsi:
                logger.info(f"  - RSI too low: {rsi_val} <= {self.rsi_sell_threshold}")
            if not condition_ema:
                logger.info(f"  - EMA Fast >= Slow: {ema_fast} >= {ema_slow * 1.001}")
            if not condition_volume:
                logger.info(
                    f"  - Volume ratio too low: {volume_ratio} <= {self.volume_sell_threshold}"
                )
            if not condition_trend:
                logger.info(
                    f"  - Trend strength too high: {trend_strength} >= {self.trend_strength_threshold}"
                )
            logger.info("=== END SELL CONDITIONS CHECK ===")
        return dataframe

    def _check_balance(self, amount: float, rate: float) -> bool:
        """Проверяет достаточность баланса для сделки"""
        try:
            if self.config.get("dry_run", False):
                free_balance = float(self.config.get("dry_run_wallet", 1000))
                logger.info(f"Dry run balance check: {free_balance} USDT")
            else:
                balance = self.exchange_instance.fetch_balance() if self.exchange_instance else None
                if not balance or not isinstance(balance, dict):
                    logger.error("Could not fetch balance")
                    return False
                usdt_balance = balance.get("USDT", {})
                if not isinstance(usdt_balance, dict):
                    logger.error("Invalid USDT balance structure")
                    return False
                free_balance = float(str(usdt_balance.get("free", "0")))
                logger.info(f"Real balance check: {free_balance} USDT")

            required_balance = float(amount) * float(rate)
            if free_balance < required_balance:
                logger.info(
                    f"Insufficient balance: Required: {required_balance} USDT, Available: {free_balance} USDT"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking balance: {str(e)}")
            return False

    def _check_open_trades(self, pair: str) -> bool:
        """Проверяет открытые позиции"""
        try:
            trades = Trade.get_trades_proxy(is_open=True)
            logger.info(f"Open trades check: {len(trades)}")
            for trade in trades:
                logger.info(f"  - {trade.pair}: Amount={trade.amount}, Open rate={trade.open_rate}")

            if len(trades) >= self.config["max_open_trades"]:
                logger.info(f"Maximum trades limit reached: {self.config['max_open_trades']}")
                return False

            if any(t.pair == pair for t in trades):
                logger.info(f"Trade already open for {pair}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking open trades: {str(e)}")
            return False

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Дополнительная проверка перед входом в позицию"""
        try:
            logger.info("=== TRADE ENTRY CONFIRMATION STARTED ===")
            logger.info(f"Time: {current_time}")
            logger.info(f"Pair: {pair}")

            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)

            imbalance_ratio = self.calculate_order_book_imbalance(pair)

            logger.info("Order details:")
            logger.info(f"  - Order book imbalance ratio: {imbalance_ratio}")
            logger.info(f"  - Required threshold: {self.buy_imbalance_threshold}")
            logger.info(f"  - Price: {rate}")
            logger.info(f"  - Amount: {amount}")
            logger.info(f"  - Order type: {order_type}")
            logger.info(f"  - Time in force: {time_in_force}")

            if not self._check_balance(amount, rate):
                logger.info("=== TRADE ENTRY REJECTED - INSUFFICIENT BALANCE ===")
                return False

            if not self._check_open_trades(pair):
                logger.info("=== TRADE ENTRY REJECTED - TRADE CHECK FAILED ===")
                return False

            if imbalance_ratio >= self.buy_imbalance_threshold:
                logger.info("=== TRADE ENTRY CONFIRMED ===")
                return True

            logger.info(
                f"Imbalance ratio too low: {imbalance_ratio} < {self.buy_imbalance_threshold}"
            )
            logger.info("=== TRADE ENTRY REJECTED - IMBALANCE RATIO TOO LOW ===")
            return False

        except Exception as e:
            logger.error(f"Error in confirm_trade_entry: {str(e)}")
            logger.info("=== TRADE ENTRY REJECTED - GENERAL ERROR ===")
            return False

    def check_balance(self, pair: str) -> None:
        """
        Проверяет доступный баланс для торговли
        """
        try:
            if not self.exchange_instance:
                self.exchange_instance = ccxt.binance({"enableRateLimit": True, "timeout": 30000})

            balance = self.exchange_instance.fetch_balance()
            logger.info(f"Available balance for {pair}:")
            logger.info(f"  - USDT: {balance['USDT']['free']}")
            logger.info(f"  - Total: {balance['total']}")
        except Exception as e:
            logger.error(f"Error checking balance: {e}")

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        sell_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Дополнительная проверка перед выходом из позиции
        """
        try:
            # Убеждаемся, что время в UTC
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            if trade.open_date.tzinfo is None:
                trade.open_date = trade.open_date.replace(tzinfo=timezone.utc)

            # Рассчитываем время в сделке
            time_in_trade = (current_time - trade.open_date).total_seconds() / 60

            logger.info(f"Trade exit confirmation for {pair}:")
            logger.info(f"  - Order type: {order_type}")
            logger.info(f"  - Amount: {amount}")
            logger.info(f"  - Rate: {rate}")
            logger.info(f"  - Time in force: {time_in_force}")
            logger.info(f"  - Sell reason: {sell_reason}")
            logger.info(f"  - Current profit: {trade.calc_profit_ratio(rate)}")
            logger.info(f"  - Current time (UTC): {current_time}")
            logger.info(f"  - Time in trade: {time_in_trade:.2f} minutes")

            # Если выход по стоп-лоссу, логируем детали
            if sell_reason == "stop_loss":
                logger.info(f"Stop loss triggered for {pair}:")
                logger.info(f"  - Entry price: {trade.open_rate}")
                logger.info(f"  - Current price: {rate}")
                logger.info(f"  - Profit: {trade.calc_profit_ratio(rate)}")
                logger.info(f"  - Time in trade: {time_in_trade:.2f} minutes")

            return True
        except Exception as e:
            logger.error(f"Error in confirm_trade_exit for {pair}: {str(e)}")
            return True  # Разрешаем выход даже при ошибке

    def custom_sell(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        Дополнительные условия для выхода из позиции
        """
        # Получаем дисбаланс
        imbalance_ratio = self.calculate_order_book_imbalance(pair)

        # Если сильный медвежий дисбаланс и мы в прибыли, закрываем позицию
        if imbalance_ratio <= self.sell_imbalance_threshold * 0.5 and current_profit > 0.002:
            return "strong_bearish_imbalance"

        # Если прибыль достаточная и дисбаланс начинает меняться, забираем прибыль
        if current_profit > 0.005 and imbalance_ratio < 0.9:  # 1.0
            return "take_profit_on_shift"

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
        Динамический расчет размера позиции на основе ATR
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < self.atr_period:
            return proposed_stake

        # Получаем ATR
        atr = dataframe["atr"].iloc[-1]

        # Рассчитываем размер позиции на основе ATR
        # Используем proposed_stake как базовый размер позиции
        risk_amount = proposed_stake * self.risk_per_trade
        position_size = risk_amount / (atr * self.atr_multiplier)

        # Ограничиваем размер позиции
        position_size = min(position_size, max_stake)
        position_size = max(position_size, min_stake)

        logger.info(f"Dynamic position sizing for {pair}:")
        logger.info(f"  - ATR: {atr}")
        logger.info(f"  - Risk amount: {risk_amount}")
        logger.info(f"  - Calculated position size: {position_size}")

        return position_size

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
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < self.atr_period:
            return self.stoploss

        # Получаем ATR
        atr = dataframe["atr"].iloc[-1]

        # Рассчитываем стоп-лосс на основе ATR
        atr_stoploss = -(atr * self.atr_multiplier / current_rate)

        # Используем более консервативный стоп-лосс
        final_stoploss = max(self.stoploss, atr_stoploss)

        logger.info(f"Dynamic stoploss for {pair}:")
        logger.info(f"  - ATR: {atr}")
        logger.info(f"  - ATR-based stoploss: {atr_stoploss}")
        logger.info(f"  - Final stoploss: {final_stoploss}")

        return final_stoploss
