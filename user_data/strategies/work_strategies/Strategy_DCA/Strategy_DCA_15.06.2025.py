# Order Book Imbalance Scalping Strategy

# Импорты из стандартной библиотеки
import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta

# Импорты сторонних библиотек
import ccxt
import numpy as np
import pandas as pd
from ccxt.base.types import OrderBook
from dotenv import load_dotenv
from pandas import DataFrame

# Локальные импорты
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy

# Запись логов на диск
ENABLE_STRATEGY_LOG = True  # Выключатель записи для strategy.log

# Настройка логгера
logger = logging.getLogger("Strategy_DCA")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Добавляем консольный handler, если его нет
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Добавляем файловый handler для strategy.log, если его нет и включена запись на диск
logfile_path = "/home/dm/freqtrade/user_data/logs/strategy.log"
if ENABLE_STRATEGY_LOG and not any(
    isinstance(h, logging.FileHandler)
    and getattr(h, "baseFilename", "") == os.path.abspath(logfile_path)
    for h in logger.handlers
):
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger.info("Strategy_DCA module loaded")

# Загружаем переменные среды из .env файла
load_dotenv()


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


class Strategy_DCA(IStrategy):
    """
    Стратегия скальпинга на основе дисбаланса в книге ордеров.
    Анализирует соотношение между ордерами на покупку и продажу для
    определения краткосрочных движений цены.
    + DCA (динамическая коррекция позиции)
    """

    # Настройки стратегии
    minimal_roi = {
        "0": 0.02,  # 2.0% для быстрого закрытия (идеальный сценарий)
        "5": 0.015,  # 1.5% если не закрылась за 5 минут
        "10": 0.01,  # 1.0% если не закрылась за 10 минут
        "15": 0.005,  # 0.55% если не закрылась за 15 минут
    }
    # Основные настройки управления капиталом
    stoploss = -0.005  # Ужесточаем стоп-лосс до 0.5% для скальпинга
    timeframe = "1m"  # Оставляем 1-минутный таймфрейм

    # Настройки для скальпинга
    startup_candle_count = 50  # Уменьшаем для более быстрого старта
    process_only_new_candles = True

    # Параметры стакана и фильтров
    order_book_depth: int = 5  # Уменьшаем глубину стакана для скорости
    use_imbalance_filter: bool = True
    use_volume_filter: bool = True
    use_rsi_filter: bool = True
    use_ema_filter: bool = True

    # Параметры для анализа объема
    volume_ma_period = 10  # Уменьшаем период для более быстрой реакции

    # Параметры для обычного режима (без всплеска)
    normal_buy_imbalance_threshold: float = 1.8  # Порог дисбаланса для входа
    normal_buy_volume_threshold: float = 0.2  # Порог объема для входа
    normal_buy_rsi_threshold: float = 40  # Порог RSI для входа

    normal_sell_imbalance_threshold: float = 0.7  # Порог дисбаланса для выхода
    normal_sell_volume_threshold: float = 0.2  # Порог объема для выхода
    normal_sell_rsi_threshold: float = 65  # Порог RSI для выхода

    # Параметры для режима всплеска
    surge_buy_imbalance_threshold: float = 2.2  # Порог дисбаланса для входа
    surge_buy_volume_threshold: float = 0.25  # Порог объема для входа
    surge_buy_rsi_threshold: float = 35  # Порог RSI для входа

    surge_sell_imbalance_threshold: float = 0.4  # Порог дисбаланса для выхода
    surge_sell_volume_threshold: float = 0.25  # Порог объема для выхода
    surge_sell_rsi_threshold: float = 70  # Порог RSI для выхода

    # Параметры для анализа тренда (EMA)
    trend_ema_fast = 3  # было 5
    trend_ema_slow = 8  # было 13
    trend_ema_signal = 4  # было 6
    trend_strength_threshold = 0.001  # Порог силы тренда (0.1%)

    # Веса для условий EMA
    ema_condition_weights = {
        "ema_fast_slow": 0.6,  # было 0.4
        "ema_slow_signal": 0.2,  # было 0.3
        "trend_strength": 0.2,  # было 0.3
    }
    condition_ema_threshold = 0.3  # Порог для условий EMA

    # Настройки трейлинг-стопа для скальпинга
    trailing_stop = True
    trailing_stop_positive = 0.003  # отставание трейлинг-стопа на 0.3% от текущей цены
    trailing_stop_positive_offset = 0.004  # активация трейлинга при достижении 0.4% прибыли
    trailing_only_offset_is_reached = True  # Активация только после достижения смещения

    # Настройки кулдауна после убыточных сделок
    cooldown_lookback = 1  # Количество последних сделок для проверки
    cooldown_period = 10  # Время кулдауна в минутах (10 минут)

    # Настройки для DCA
    position_adjustment_enable = True
    max_entry_position_adjustment = 3  # Было 2 - увеличиваем количество уровней усреднения
    min_dca_time = 1  # Было 2 - уменьшаем время между докупками

    # Параметры для управления рисками
    risk_per_trade = 0.03  # Увеличиваем риск до 3% на сделку
    atr_period = 14  # Период для расчета ATR
    atr_multiplier = 2.0  # Множитель для расчета стоп-лосса на основе ATR

    # Настройки для оптимизации производительности
    _last_ohlcv_update = {}
    _ohlcv_update_interval = 1  # секунды между обновлениями OHLCV
    _last_orderbook_update = {}
    _orderbook_update_interval = 1  # секунды между обновлениями стакана

    # Специфичные настройки для пар
    _pair_specific_thresholds = {
        "BTC/USDT": {
            "normal_buy_imbalance_threshold": 1.3,  # Более частые входы
            "normal_buy_volume_threshold": 0.12,  # Меньший порог объема
            "normal_buy_rsi_threshold": 40,  # Более высокий RSI для входов
            "surge_buy_imbalance_threshold": 1.6,  # Порог для режима всплеска
            "surge_buy_volume_threshold": 0.15,  # Объем для режима всплеска
            "surge_buy_rsi_threshold": 45,  # RSI для режима всплеска
            "cooldown_period": 20,  # Уменьшенный кулдаун
        },
        "ETH/USDT": {
            "normal_buy_imbalance_threshold": 1.25,  # Было 1.25 - снижаем для более частых входов
            "normal_buy_volume_threshold": 0.08,  # Было 0.08 - снижаем порог объема
            "normal_buy_rsi_threshold": 45,  # Было 48 - повышаем RSI для более раннего входа
            "surge_buy_imbalance_threshold": 1.5,  # Было 1.5 - снижаем для более частых входов
            "surge_buy_volume_threshold": 0.1,  # Было 0.1 - снижаем порог объема
            "surge_buy_rsi_threshold": 50,  # Оставляем 45
            "cooldown_period": 10,  # Было 10 - уменьшаем кулдаун
        },
        "BNB/USDT": {
            "normal_buy_imbalance_threshold": 1.5,
            "normal_buy_volume_threshold": 0.15,
            "normal_buy_rsi_threshold": 44,
            "surge_buy_imbalance_threshold": 1.8,
            "surge_buy_volume_threshold": 0.18,
            "surge_buy_rsi_threshold": 45,
            "cooldown_period": 15,
        },
        "DOT/USDT": {
            "normal_buy_imbalance_threshold": 1.6,  # Было 1.6 - снижаем для более частых входов
            "normal_buy_volume_threshold": 0.12,  # Было 0.12 - снижаем порог объема
            "normal_buy_rsi_threshold": 40,  # Было 47 - повышаем RSI для более раннего входа
            "surge_buy_imbalance_threshold": 1.9,  # Было 1.9 - снижаем для более частых входов
            "surge_buy_volume_threshold": 0.15,  # Было 0.15 - снижаем порог объема
            "surge_buy_rsi_threshold": 44,  # Было 44 - повышаем RSI для более раннего входа
            "cooldown_period": 8,  # Было 10 - уменьшаем кулдаун
        },
        "SOL/USDT": {
            "normal_buy_imbalance_threshold": 1.4,
            "normal_buy_volume_threshold": 0.1,
            "normal_buy_rsi_threshold": 35,
            "surge_buy_imbalance_threshold": 1.7,
            "surge_buy_volume_threshold": 0.13,
            "surge_buy_rsi_threshold": 42,
            "cooldown_period": 12,
        },
    }

    # Кэш для хранения данных книги ордеров
    _orderbook_cache = {}
    _orderbook_cache_time = {}
    _orderbook_cache_ttl = 1  # Время жизни кэша (в секундах)
    exchange_instance = None

    # Настройки для DCA (динамической коррекции позиции)
    position_adjustment_enable = True
    max_entry_position_adjustment = 2  # Максимум 2 дополнительные закупки
    min_dca_time = 2  # Минимальный промежуток между докупками (в минутах)

    # Параметры для управления рисками
    risk_per_trade = 0.03  # 3% риска на сделку (было 0.02)
    atr_period = 14  # Период для расчета ATR
    atr_multiplier = 2.0  # Множитель для расчета стоп-лосса на основе ATR

    # Настройки для оптимизации производительности
    _last_ohlcv_update = {}
    _ohlcv_update_interval = 1  # секунды между обновлениями OHLCV
    _last_orderbook_update = {}
    _orderbook_update_interval = 1  # секунды между обновлениями стакана

    def __init__(self, config: dict):
        super().__init__(config)
        logger.info(f"Using timeframe: {self.timeframe}")

        # Инициализация exchange с использованием переменных среды
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
        """
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

    def calculate_order_book_imbalance(self, pair: str) -> tuple[float, float, float]:
        """
        Рассчитывает коэффициент дисбаланса в книге ордеров и объемы
        Возвращает: (imbalance_ratio, buy_volume, sell_volume)
        """
        current_time = datetime.now(timezone.utc)

        try:
            # Проверяем инициализацию exchange
            if not self.exchange_instance:
                logger.error("Exchange instance is not initialized")
                return 1.0, 0.0, 0.0

            # Получаем новые данные
            orderbook = self.exchange_instance.fetch_order_book(pair, self.order_book_depth)
            if (
                not isinstance(orderbook, dict)
                or "bids" not in orderbook
                or "asks" not in orderbook
            ):
                logger.error(f"Некорректная структура книги ордеров для {pair}")
                return 1.0, 0.0, 0.0

            # Получаем объемы на покупку и продажу
            bids_volume = sum(float(str(bid[1])) for bid in orderbook["bids"])
            asks_volume = sum(float(str(ask[1])) for ask in orderbook["asks"])

            # Рассчитываем коэффициент дисбаланса
            imbalance_ratio = bids_volume / asks_volume if asks_volume > 0 else 1.0

            # Обновляем кэш
            self._orderbook_cache[pair] = orderbook
            self._orderbook_cache_time[pair] = current_time

            return imbalance_ratio, bids_volume, asks_volume

        except Exception as e:
            logger.error(f"Ошибка при расчете дисбаланса для {pair}: {str(e)}")
            return 1.0, 0.0, 0.0

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
                if pair in self._last_orderbook_update:
                    last_update = self._last_orderbook_update[pair]
                    if (
                        current_time - last_update
                    ).total_seconds() < self._orderbook_update_interval:
                        if pair in self._orderbook_cache:
                            imbalance_ratio, buy_volume, sell_volume = (
                                self.calculate_order_book_imbalance(pair)
                            )
                            df["orderbook_imbalance"] = imbalance_ratio
                            df["buy_volume"] = buy_volume
                            df["sell_volume"] = sell_volume
                            return df

                # Обновляем данные стакана
                orderbook = self.get_order_book(pair, self.order_book_depth)
                if orderbook:
                    imbalance_ratio, buy_volume, sell_volume = self.calculate_order_book_imbalance(
                        pair
                    )
                    df["orderbook_imbalance"] = imbalance_ratio
                    df["buy_volume"] = buy_volume
                    df["sell_volume"] = sell_volume
                    self._orderbook_cache[pair] = orderbook
                    self._last_orderbook_update[pair] = current_time
                    logger.info(
                        f"Current orderbook for {pair}: imbalance={imbalance_ratio:.2f}, buy_vol={buy_volume:.2f}, sell_vol={sell_volume:.2f}"
                    )

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
                df["volume_ma"] = df["volume"].rolling(window=self.volume_ma_period).mean()
                df["volume_ratio"] = df["volume"] / df["volume_ma"]
                df["atr"] = atr(df["high"], df["low"], df["close"], period=14)

            # Обновляем время последнего обновления OHLCV
            self._last_ohlcv_update[pair] = current_time

            return df

        except Exception as e:
            logger.error(f"Error in populate_indicators: {str(e)}")
            return dataframe

    # DCA - Динамическая корректировка размера позиции
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
        try:
            # Получаем минимальные требования биржи
            market = self.exchange_instance.market(trade.pair)
            min_amount = float(market.get("limits", {}).get("amount", {}).get("min", 0))
            min_cost = float(market.get("limits", {}).get("cost", {}).get("min", 0))
            min_stake_required = max(min_cost, min_amount * current_rate)

            current_entries = trade.nr_of_successful_entries

            # Рассчитываем уровни для докупки
            dca_levels = {0: -0.005, 1: -0.01}  # Первая докупка при -0.5%  # Вторая докупка при -1%

            # Если текущий профит ниже нашего уровня и не превышено макс. кол-во входов
            if current_entries < self.max_entry_position_adjustment:
                # Проверяем, соответствует ли текущая просадка нашему уровню для докупки
                if current_profit <= dca_levels.get(current_entries, -999):
                    # Рассчитываем размер докупки
                    multiplier = 1.5
                    dca_amount = trade.stake_amount * (multiplier**current_entries)

                    # Проверяем минимальные требования
                    if dca_amount < min_stake_required:
                        logger.warning(
                            f"DCA amount {dca_amount} too small for {trade.pair}. Min required: {min_stake_required}"
                        )
                        return None

                    return dca_amount

            # Частичная фиксация прибыли
            if not trade.has_open_orders:
                if current_profit >= 0.01:
                    # При прибыли +1% фиксируем 30% позиции
                    exit_amount = trade.amount * 0.3
                    if exit_amount * current_rate < min_cost or exit_amount < min_amount:
                        logger.warning(
                            f"Exit amount {exit_amount} too small for {trade.pair}. Min amount: {min_amount}, Min cost: {min_cost}"
                        )
                        return None
                    return -exit_amount
                elif current_profit >= 0.005:
                    # При прибыли +0.5% фиксируем 30% позиции
                    exit_amount = trade.amount * 0.3
                    if exit_amount * current_rate < min_cost or exit_amount < min_amount:
                        logger.warning(
                            f"Exit amount {exit_amount} too small for {trade.pair}. Min amount: {min_amount}, Min cost: {min_cost}"
                        )
                        return None
                    return -exit_amount

            return None

        except Exception as e:
            logger.error(f"Error in adjust_trade_position for {trade.pair}: {str(e)}")
            return None

    def get_pair_specific_cooldown(self, pair: str) -> int:
        # Получение специфичного времени кулдауна для пары
        if pair in self._pair_specific_thresholds:
            return self._pair_specific_thresholds[pair].get("cooldown_period", self.cooldown_period)
        return self.cooldown_period

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # Настройка цены для входа в позицию
        # Для скальпинга важно входить по наилучшей возможной цене

        # Проверяем кулдаун после убыточных сделок
        if self.cooldown_period:
            last_trade = (
                Trade.get_trades_proxy(is_open=False, pair=pair).order_by("-close_date").first()
            )
            if last_trade and last_trade.calc_profit_ratio() < 0:
                # Если последняя сделка была убыточной
                cooldown_period = self.get_pair_specific_cooldown(pair)
                cooldown_end = last_trade.close_date + timedelta(minutes=cooldown_period)
                if current_time < cooldown_end:
                    logger.info(
                        f"Pair {pair} is in cooldown until {cooldown_end} due to previous loss"
                    )
                    logger.info(f"  - Last trade profit: {last_trade.calc_profit_ratio():.4%}")
                    logger.info(f"  - Cooldown period: {cooldown_period} minutes")
                    return 0  # Возвращаем 0, чтобы запретить вход

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
        # Настройка цены для частичного или полного выхода из позиции
        try:
            # Получаем минимальные требования биржи
            market = self.exchange_instance.market(pair)
            min_amount = float(market.get("limits", {}).get("amount", {}).get("min", 0))
            min_cost = float(market.get("limits", {}).get("cost", {}).get("min", 0))

            # Проверяем, не меньше ли сумма выхода минимальных требований
            if exit_tag and "partial" in exit_tag:
                exit_amount = trade.amount
                if exit_tag == "partial_exit_1":
                    exit_amount = trade.amount * 0.3
                elif exit_tag == "partial_exit_2":
                    exit_amount = trade.amount * 0.3

                # Проверяем минимальные требования
                if exit_amount * proposed_rate < min_cost or exit_amount < min_amount:
                    logger.warning(
                        f"Exit amount {exit_amount} too small for {pair}. Min amount: {min_amount}, Min cost: {min_cost}"
                    )
                    # Если сумма слишком мала, выходим полностью
                    return proposed_rate

                # для частичной фиксации используем лимитные ордера чуть выше рынка
                if exit_tag == "partial_exit_1":
                    return proposed_rate * 1.001
                elif exit_tag == "partial_exit_2":
                    return proposed_rate * 1.0015
                else:
                    return proposed_rate * 1.001

            # для финального выхода используем предложенную цену
            return proposed_rate

        except Exception as e:
            logger.error(f"Error in custom_exit_price for {pair}: {str(e)}")
            return proposed_rate

    # DCA - END

    def get_pair_specific_threshold(
        self, pair: str, param_name: str, default_value: float
    ) -> float:
        # Получение специфичных порогов для пары
        logger.info(f"Getting specific threshold for {pair}, param: {param_name}")
        logger.info(f"Available specific thresholds: {self._pair_specific_thresholds}")

        if pair in self._pair_specific_thresholds:
            specific_value = self._pair_specific_thresholds[pair].get(param_name, default_value)
            logger.info(
                f"Found specific threshold for {pair}: {specific_value} (default was {default_value})"
            )
            return specific_value

        logger.info(f"No specific threshold found for {pair}, using default: {default_value}")
        return default_value

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"=== BUY CONDITIONS FOR {pair}: ===")
        current_price = dataframe["close"].iloc[-1]
        logger.info(f"  - Current buy price: {current_price}")

        # Получаем специфичные пороги для пары
        pair_imbalance_threshold = self.get_pair_specific_threshold(
            pair, "normal_buy_imbalance_threshold", self.normal_buy_imbalance_threshold
        )
        pair_volume_surge = self.get_pair_specific_threshold(
            pair, "surge_buy_volume_threshold", self.surge_buy_volume_threshold
        )
        pair_min_volume = self.get_pair_specific_threshold(
            pair, "normal_buy_volume_threshold", self.normal_buy_volume_threshold
        )
        pair_rsi_threshold = self.get_pair_specific_threshold(
            pair, "normal_buy_rsi_threshold", self.normal_buy_rsi_threshold
        )

        imbalance_ratio, buy_volume, sell_volume = self.calculate_order_book_imbalance(pair)
        rsi_val = dataframe["rsi"].iloc[-1]

        # Анализируем объемы покупок и продаж
        buy_volume_ratio = (
            buy_volume / dataframe["volume_ma"].iloc[-1]
            if dataframe["volume_ma"].iloc[-1] > 0
            else 0
        )
        sell_volume_ratio = (
            sell_volume / dataframe["volume_ma"].iloc[-1]
            if dataframe["volume_ma"].iloc[-1] > 0
            else 0
        )

        # Определяем режим работы
        is_buy_volume_surge = (
            buy_volume_ratio > pair_volume_surge if self.use_volume_filter else False
        )
        is_sell_volume_surge = (
            sell_volume_ratio > self.surge_sell_volume_threshold
            if self.use_volume_filter
            else False
        )

        logger.info(f"  - Buy volume ratio: {buy_volume_ratio:.2f}")
        logger.info(f"  - Sell volume ratio: {sell_volume_ratio:.2f}")
        logger.info(f"  - Is buy volume surge: {is_buy_volume_surge}")
        logger.info(f"  - Is sell volume surge: {is_sell_volume_surge}")

        # EMA условия
        ema_fast = dataframe["ema_fast"].iloc[-1]
        ema_slow = dataframe["ema_slow"].iloc[-1]
        ema_signal = dataframe["ema_signal"].iloc[-1]
        trend_strength = dataframe["trend_strength"].iloc[-1]

        # Базовые условия для обычного режима
        normal_condition_imbalance = imbalance_ratio > pair_imbalance_threshold
        normal_condition_volume = buy_volume_ratio > pair_min_volume
        normal_condition_rsi = rsi_val < pair_rsi_threshold
        normal_condition_ema = ema_fast > ema_slow and ema_slow > ema_signal

        # Базовые условия для режима всплеска
        surge_condition_imbalance = imbalance_ratio > self.surge_buy_imbalance_threshold
        surge_condition_volume = is_buy_volume_surge and not is_sell_volume_surge
        surge_condition_rsi = rsi_val < self.surge_buy_rsi_threshold
        surge_condition_ema = normal_condition_ema  # Используем те же условия EMA для всплеска

        # Применяем фильтры к условиям
        if not self.use_imbalance_filter:
            normal_condition_imbalance = True
            surge_condition_imbalance = True
        if not self.use_volume_filter:
            normal_condition_volume = True
            surge_condition_volume = True
        if not self.use_rsi_filter:
            normal_condition_rsi = True
            surge_condition_rsi = True
        if not self.use_ema_filter:
            normal_condition_ema = True
            surge_condition_ema = True

        # Выбираем условия в зависимости от режима
        if is_buy_volume_surge and not is_sell_volume_surge:
            logger.info("Using surge mode conditions for buy")
            conditions = [
                surge_condition_imbalance,
                surge_condition_volume,
                surge_condition_rsi,
                surge_condition_ema,
            ]
            # Логируем только если фильтр включен
            if self.use_rsi_filter:
                logger.info(
                    f"  - RSI (surge mode): {rsi_val} "
                    f"(threshold: {self.surge_buy_rsi_threshold}, "
                    f"condition: {surge_condition_rsi})"
                )
        else:
            logger.info("Using normal mode conditions for buy")
            conditions = [
                normal_condition_imbalance,
                normal_condition_volume,
                normal_condition_rsi,
                normal_condition_ema,
            ]

        # Логирование условий только для включенных фильтров
        if self.use_imbalance_filter:
            logger.info(
                f"  - Imbalance ratio: {imbalance_ratio} "
                f"(threshold: {pair_imbalance_threshold}, "
                f"condition: {normal_condition_imbalance})"
            )
        if self.use_rsi_filter:
            logger.info(
                f"  - RSI: {rsi_val} "
                f"(threshold: {pair_rsi_threshold}, "
                f"condition: {normal_condition_rsi})"
            )
        if self.use_volume_filter:
            logger.info(
                f"  - Buy volume ratio: {buy_volume_ratio:.2f} "
                f"(threshold: {pair_min_volume}, "
                f"condition: {normal_condition_volume})"
            )
        if self.use_ema_filter:
            ema_score = 0.0
            if ema_fast > ema_slow:
                ema_score += self.ema_condition_weights["ema_fast_slow"]

            logger.info(f"  - EMA Fast: {ema_fast}")
            logger.info(f"  - EMA Slow: {ema_slow}")
            logger.info(f"  - EMA Signal: {ema_signal}")
            logger.info(f"  - Trend Strength: {trend_strength}")
            logger.info(
                f"  - EMA Score: {ema_score:.2f} (threshold: {self.condition_ema_threshold})"
            )
            logger.info("  - EMA conditions check:")
            logger.info(
                f"    1. EMA Fast > EMA Slow: {ema_fast > ema_slow} ({ema_fast} > {ema_slow})"
            )
            logger.info(
                f"    2. EMA Slow > EMA Signal: {ema_slow > ema_signal} ({ema_slow} > {ema_signal})"
            )
            logger.info(f"  - EMA conditions met: {normal_condition_ema}")

        final_condition = all(conditions)
        if final_condition:
            dataframe.loc[dataframe.index[-1], "buy"] = 1
            logger.info("=== END BUY CONDITIONS CHECK ===")
        else:
            logger.info(f"No buy signal for {pair} - Some conditions failed")
            logger.info("Failed conditions:")
            if self.use_imbalance_filter and not normal_condition_imbalance:
                logger.info(
                    f"  - Imbalance ratio too low: {imbalance_ratio} <= {pair_imbalance_threshold}"
                )
            if self.use_rsi_filter and not normal_condition_rsi:
                logger.info(f"  - RSI too high: {rsi_val} >= {pair_rsi_threshold}")
            if self.use_volume_filter and not normal_condition_volume:
                logger.info(
                    f"  - Buy volume ratio too low: {buy_volume_ratio:.2f} <= {pair_min_volume}"
                )
            if self.use_ema_filter and not normal_condition_ema:
                logger.info(f"  - EMA conditions not met")
            logger.info("=== END BUY CONDITIONS CHECK ===")
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"=== SELL CONDITIONS FOR {pair}: ===")
        current_price = dataframe["close"].iloc[-1]
        logger.info(f"  - Current sell price: {current_price}")

        imbalance_ratio, buy_volume, sell_volume = self.calculate_order_book_imbalance(pair)
        rsi_val = dataframe["rsi"].iloc[-1]

        # Анализируем объемы покупок и продаж
        buy_volume_ratio = (
            buy_volume / dataframe["volume_ma"].iloc[-1]
            if dataframe["volume_ma"].iloc[-1] > 0
            else 0
        )
        sell_volume_ratio = (
            sell_volume / dataframe["volume_ma"].iloc[-1]
            if dataframe["volume_ma"].iloc[-1] > 0
            else 0
        )

        # Определяем режим работы
        is_buy_volume_surge = (
            buy_volume_ratio > self.surge_buy_volume_threshold if self.use_volume_filter else False
        )
        is_sell_volume_surge = (
            sell_volume_ratio > self.surge_sell_volume_threshold
            if self.use_volume_filter
            else False
        )

        logger.info(f"  - Buy volume ratio: {buy_volume_ratio:.2f}")
        logger.info(f"  - Sell volume ratio: {sell_volume_ratio:.2f}")
        logger.info(f"  - Is buy volume surge: {is_buy_volume_surge}")
        logger.info(f"  - Is sell volume surge: {is_sell_volume_surge}")

        # EMA условия
        ema_fast = dataframe["ema_fast"].iloc[-1]
        ema_slow = dataframe["ema_slow"].iloc[-1]
        ema_signal = dataframe["ema_signal"].iloc[-1]
        trend_strength = dataframe["trend_strength"].iloc[-1]

        # Базовые условия для обычного режима
        normal_condition_imbalance = imbalance_ratio < self.normal_sell_imbalance_threshold
        normal_condition_volume = sell_volume_ratio > self.normal_sell_volume_threshold
        normal_condition_rsi = rsi_val > self.normal_sell_rsi_threshold
        normal_condition_ema = ema_fast < ema_slow

        # Базовые условия для режима всплеска
        surge_condition_imbalance = imbalance_ratio < self.surge_sell_imbalance_threshold
        surge_condition_volume = is_sell_volume_surge
        surge_condition_rsi = rsi_val > self.surge_sell_rsi_threshold
        surge_condition_ema = normal_condition_ema  # Используем те же условия EMA для всплеска

        # Применяем фильтры к условиям
        if not self.use_imbalance_filter:
            normal_condition_imbalance = True
            surge_condition_imbalance = True
        if not self.use_volume_filter:
            normal_condition_volume = True
            surge_condition_volume = True
        if not self.use_rsi_filter:
            normal_condition_rsi = True
            surge_condition_rsi = True
        if not self.use_ema_filter:
            normal_condition_ema = True
            surge_condition_ema = True

        # Выбираем условия в зависимости от режима
        if is_sell_volume_surge:
            logger.info("Using surge mode conditions for sell")
            conditions = [
                surge_condition_imbalance,
                surge_condition_volume,
                surge_condition_rsi,
                surge_condition_ema,
            ]
        else:
            logger.info("Using normal mode conditions for sell")
            conditions = [
                normal_condition_imbalance,
                normal_condition_volume,
                normal_condition_rsi,
                normal_condition_ema,
            ]

        # Логирование условий только для включенных фильтров
        if self.use_imbalance_filter:
            logger.info(
                f"  - Imbalance ratio: {imbalance_ratio} "
                f"(threshold: {self.normal_sell_imbalance_threshold}, "
                f"condition: {normal_condition_imbalance})"
            )
        if self.use_rsi_filter:
            logger.info(
                f"  - RSI: {rsi_val} "
                f"(threshold: {self.normal_sell_rsi_threshold}, "
                f"condition: {normal_condition_rsi})"
            )
        if self.use_volume_filter:
            logger.info(
                f"  - Sell volume ratio: {sell_volume_ratio:.2f} "
                f"(threshold: {self.normal_sell_volume_threshold}, "
                f"condition: {normal_condition_volume})"
            )
        if self.use_ema_filter:
            logger.info(f"  - EMA Fast: {ema_fast}")
            logger.info(f"  - EMA Slow: {ema_slow}")
            logger.info(f"  - EMA Signal: {ema_signal}")
            logger.info(f"  - Trend Strength: {trend_strength}")
            logger.info(f"  - EMA conditions met: {normal_condition_ema}")

        final_condition = all(conditions)
        # Проверяем текущую прибыль
        current_profit = (dataframe["close"].iloc[-1] - dataframe["open"].iloc[0]) / dataframe[
            "open"
        ].iloc[0]
        minimal_profit = 0.01  # Увеличиваем минимальную ROI до 1%

        if final_condition and current_profit >= minimal_profit:
            dataframe.loc[dataframe.index[-1], "sell"] = 1
            real_profit = self._get_real_trade_profit(pair)
            if real_profit is not None:
                logger.info(
                    f"Sell signal generated for {pair} with real trade profit: {real_profit:.4%}"
                )
            else:
                logger.info(
                    f"Sell signal generated for {pair} (no open trade found) with profit {current_profit:.4%}"
                )
            logger.info("=== END SELL CONDITIONS CHECK ===")
        else:
            logger.info(f"No sell signal for {pair} - Some conditions failed")
            logger.info("Failed conditions:")
            if current_profit < minimal_profit:
                logger.info(
                    f"  - Current profit {current_profit:.4%} < Minimal ROI {minimal_profit:.4%}"
                )
            if self.use_imbalance_filter and not normal_condition_imbalance:
                logger.info(
                    f"  - Imbalance ratio too high: {imbalance_ratio} >= {self.normal_sell_imbalance_threshold}"
                )
            if self.use_rsi_filter and not normal_condition_rsi:
                logger.info(f"  - RSI too low: {rsi_val} <= {self.normal_sell_rsi_threshold}")
            if self.use_volume_filter and not normal_condition_volume:
                logger.info(
                    f"  - Sell volume ratio too low: {sell_volume_ratio:.2f} <= {self.normal_sell_volume_threshold}"
                )
            if self.use_ema_filter and not normal_condition_ema:
                logger.info(f"  - EMA conditions not met")
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

            imbalance_ratio, _, _ = self.calculate_order_book_imbalance(pair)
            threshold = self.get_pair_specific_threshold(
                pair, "normal_buy_imbalance_threshold", self.normal_buy_imbalance_threshold
            )

            logger.info("Order details:")
            logger.info(f"  - Order book imbalance ratio: {imbalance_ratio}")
            logger.info(f"  - Required threshold: {threshold}")
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

            if imbalance_ratio >= threshold:
                logger.info("=== TRADE ENTRY CONFIRMED ===")
                return True

            logger.info(f"Imbalance ratio too low: {imbalance_ratio} < {threshold}")
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
        # Убеждаемся, что время в UTC
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        if trade.open_date.tzinfo is None:
            trade.open_date = trade.open_date.replace(tzinfo=timezone.utc)

        # Получаем дисбаланс
        imbalance_ratio, _, _ = self.calculate_order_book_imbalance(pair)
        threshold = self.get_pair_specific_threshold(
            pair, "normal_sell_imbalance_threshold", self.normal_sell_imbalance_threshold
        )

        logger.info(f"Checking custom sell conditions for {pair}:")
        logger.info(f"  - Current profit: {current_profit:.4%}")
        logger.info(f"  - Imbalance ratio: {imbalance_ratio}")
        logger.info(f"  - Threshold for strong bearish: {threshold * 0.5}")
        logger.info(f"  - Threshold for take profit: 0.85")
        logger.info(
            f"  - Time in trade: {(current_time - trade.open_date).total_seconds() / 60:.2f} minutes"
        )

        # Если сильный медвежий дисбаланс и мы в прибыли
        if imbalance_ratio <= threshold * 0.5 and current_profit > 0.01:  # Уменьшен до 1%
            logger.info(f"Triggering strong_bearish_imbalance exit for {pair}")
            logger.info(
                f"  - Reason: Strong bearish imbalance ({imbalance_ratio:.4f}) with profit {current_profit:.4%}"
            )
            return "strong_bearish_imbalance"

        # Если прибыль достаточная и дисбаланс начинает меняться
        if current_profit > 0.015 and imbalance_ratio < 0.85:  # Уменьшен до 1.5%
            logger.info(f"Triggering take_profit_on_shift exit for {pair}")
            logger.info(
                f"  - Reason: Take profit on imbalance shift ({imbalance_ratio:.4f}) with profit {current_profit:.4%}"
            )
            return "take_profit_on_shift"

        logger.info(f"No custom sell conditions met for {pair}")
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
        try:
            # Получаем минимальные требования биржи
            market = self.exchange_instance.market(pair)
            min_amount = float(market.get("limits", {}).get("amount", {}).get("min", 0))
            min_cost = float(market.get("limits", {}).get("cost", {}).get("min", 0))

            # Рассчитываем минимальный размер ставки в USDT
            min_stake_required = max(min_cost, min_amount * current_rate)

            # Проверяем, достаточно ли баланса для минимальной ставки
            if self.config.get("dry_run", False):
                available_balance = float(self.config.get("dry_run_wallet", 1000))
            else:
                balance = self.exchange_instance.fetch_balance() if self.exchange_instance else None
                if not balance or not isinstance(balance, dict):
                    logger.warning(f"Insufficient balance for {pair}: Could not fetch balance")
                    return 0
                usdt_balance = balance.get("USDT", {})
                if not isinstance(usdt_balance, dict):
                    logger.warning(
                        f"Insufficient balance for {pair}: Invalid USDT balance structure"
                    )
                    return 0
                available_balance = float(str(usdt_balance.get("free", "0")))

            if available_balance < min_stake_required:
                logger.warning(
                    f"Insufficient balance for {pair}: Available {available_balance} < Min required {min_stake_required}"
                )
                return 0

            # Если баланса достаточно только для минимальной ставки, возвращаем её
            if available_balance <= min_stake_required * 1.1:  # Добавляем 10% буфер
                logger.info(f"Using minimum stake for {pair} due to limited balance")
                return min_stake_required

            # Получаем данные для расчета ATR
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < self.atr_period:
                logger.info(f"Using minimum stake for {pair} due to insufficient data")
                return min_stake_required

            # Получаем ATR
            atr = dataframe["atr"].iloc[-1]
            if atr <= 0:
                logger.warning(f"Invalid ATR for {pair}: {atr}")
                return min_stake_required

            # Рассчитываем размер позиции на основе ATR
            risk_amount = min(proposed_stake, available_balance * 0.95) * self.risk_per_trade
            position_size = risk_amount / (atr * self.atr_multiplier)

            # Ограничиваем размер позиции
            position_size = min(position_size, max_stake, available_balance * 0.95)
            position_size = max(position_size, min_stake_required)

            # Проверяем, что размер позиции соответствует минимальным требованиям биржи
            if position_size < min_stake_required:
                logger.warning(
                    f"Calculated position size {position_size} too small for {pair}. Using minimum stake {min_stake_required}"
                )
                return min_stake_required

            logger.info(f"Dynamic position sizing for {pair}:")
            logger.info(f"  - Available balance: {available_balance}")
            logger.info(f"  - Min stake required: {min_stake_required}")
            logger.info(f"  - Max stake: {max_stake}")
            logger.info(f"  - ATR: {atr}")
            logger.info(f"  - Risk amount: {risk_amount}")
            logger.info(f"  - Calculated position size: {position_size}")

            return position_size

        except Exception as e:
            logger.error(f"Error in custom_stake_amount for {pair}: {str(e)}")
            return 0

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

    # Вспомогательная функция для получения реального профита по открытой сделке
    def _get_real_trade_profit(self, pair: str):
        try:
            trades = Trade.get_trades_proxy(is_open=True)
            for trade in trades:
                if trade.pair == pair:
                    # Используем последнюю цену закрытия как текущий rate
                    current_rate = (
                        trade.close_rate
                        if hasattr(trade, "close_rate") and trade.close_rate
                        else trade.open_rate
                    )
                    return trade.calc_profit_ratio(current_rate)
            return None
        except Exception as e:
            logger.error(f"Error in _get_real_trade_profit: {e}")
            return None
