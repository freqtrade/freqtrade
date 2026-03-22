from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, RealParameter
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class BollingerRSIScalping(IStrategy):
    """
    Скальпинговая стратегия на основе полос Боллинджера и RSI
    """

    # Временной интервал для торговли
    timeframe = "1m"  # 1 минута для скальпинга

    # Размер позиции в процентах от баланса (1% на сделку)
    position_size = 0.01

    # Минимальный объем для торговли (в USDT)
    min_trade_size = 10.0  # Минимальная сумма сделки на Binance Futures

    # Максимальное количество открытых позиций
    max_open_trades = 5

    # Настройки минимального ROI (Return On Investment)
    # Формат: {"min_roi_<время_в_минутах>": минимальный_профит}
    minimal_roi = {
        "0": 0.005  # 0.5% ROI - продаем только если ROI > 0.5%
    }

    # Стоп-лосс
    stoploss = -0.01  # 1% стоп-лосс

    # Трейлинг-стоп
    trailing_stop = True
    trailing_stop_positive = 0.003  # Активировать трейлинг при 0.3% прибыли
    trailing_stop_positive_offset = 0.005  # Начать трейлинг с 0.5% прибыли

    # Параметры индикаторов
    bb_length = 20
    bb_std = 2.0
    rsi_length = 14
    rsi_buy = 60  # Увеличили порог RSI до 60
    ema_fast = 5  # Быстрая EMA
    ema_slow = 13  # Медленная EMA
    ema_trend = 34  # Для определения тренда
    volume_ma = 20  # Период для скользящей средней объема
    min_trade_volume = 100  # Минимальный объем торговой пары в USDT
    max_trades_per_day = 30  # Лимит сделок в день

    # Параметры управления капиталом
    use_sl = True
    sl_mult = 1.5  # Множитель ATR для стоп-лосса
    tp_mult = 2.0  # Множитель ATR для тейк-профита
    atr_period = 14  # Период ATR

    # Время торговли (UTC)
    trading_hours = {"start": "00:00", "end": "23:59"}

    # Дополнительные параметры
    bb_offset = 1.0  # Точное касание нижней полосы

    # Настройки для бэктеста
    startup_candle_count = 100
    max_open_trades = 5
    stake_amount = 100  # Фиксированный размер позиции
    stake_currency = "USDT"

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Логируем настройки ROI при инициализации
        logger.info(f"Настройки минимального ROI: {self.minimal_roi}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        # Сохраняем копию даты, если она есть в колонках
        date_column = None
        if "date" in dataframe.columns:
            date_column = dataframe["date"].copy()

        # Если индекс не DatetimeIndex, пытаемся его создать
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            if date_column is not None:
                # Используем колонку date для индекса
                dataframe = dataframe.set_index(pd.to_datetime(date_column))
            else:
                # Пытаемся конвертировать текущий индекс
                dataframe.index = pd.to_datetime(dataframe.index)

        # Убедимся, что индекс отсортирован по времени
        dataframe = dataframe.sort_index()

        # Добавляем колонку date, если её нет (Freqtrade ожидает её наличие)
        if "date" not in dataframe.columns:
            dataframe["date"] = dataframe.index
        else:
            # Обновляем колонку date, если она устарела
            dataframe["date"] = dataframe.index

        # Полосы Боллинджера
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=self.bb_length, stds=self.bb_std
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_middle"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_length)

        # EMA
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.ema_trend)

        # Volume MA
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=self.volume_ma)

        # ATR для стоп-лосса и тейк-профита
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)

        # Определение тренда
        dataframe["trend"] = 0
        dataframe.loc[dataframe["close"] > dataframe["ema_trend"], "trend"] = 1
        dataframe.loc[dataframe["close"] < dataframe["ema_trend"], "trend"] = -1

        # Логируем текущие значения для отладки
        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]
            logger.info(
                f"{pair} - Цена: {last_row['close']:.8f}, BB Lower: {last_row['bb_lower']:.8f}, RSI: {last_row['rsi']:.2f}, Volume: {last_row['volume']:.2f}, EMA Fast: {last_row['ema_fast']:.8f}, EMA Slow: {last_row['ema_slow']:.8f}, Trend: {last_row['trend']}"
            )

        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        side: str,
        **kwargs,
    ) -> float:
        """
        Рассчитываем размер позиции в зависимости от баланса
        """
        # Всегда используем фиксированный размер позиции для упрощения
        stake = 100.0  # Фиксированный размер позиции 100 USDT

        # Логируем информацию о размере позиции
        logger.info(f"Установлен размер позиции: {stake} USDT")

        # Проверяем минимальный размер позиции
        if stake < self.min_trade_size:
            logger.warning(
                f"Слишком маленький размер позиции: {stake} USDT. Минимум: {self.min_trade_size} USDT"
            )
            return 0

        return min(stake, max_stake)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        # Проверяем, что у нас достаточно данных
        if len(dataframe) < self.startup_candle_count:
            logger.warning(
                f"Недостаточно данных для {pair}. Нужно {self.startup_candle_count}, есть {len(dataframe)}"
            )
            return dataframe

        # Инициализируем колонки
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        # Проверяем наличие необходимых индикаторов
        required_indicators = [
            "bb_lower",
            "bb_middle",
            "rsi",
            "volume",
            "ema_fast",
            "ema_slow",
            "ema_trend",
        ]
        for indicator in required_indicators:
            if indicator not in dataframe.columns:
                logger.warning(f"Отсутствует индикатор {indicator} для {pair}")
                return dataframe

        # Рассчитываем дополнительные индикаторы
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        # Условия входа
        conditions = []

        # 1. Цена коснулась или пробила нижнюю полосу Боллинджера
        conditions.append(dataframe["close"] <= dataframe["bb_lower"] * 1.001)

        # 2. RSI в зоне перепроданности
        conditions.append(dataframe["rsi"] < 35)  # Более строгое условие

        # 3. Подтверждение восходящего тренда
        conditions.append(
            dataframe["ema_fast"] > dataframe["ema_slow"]
        )  # Быстрая EMA выше медленной
        conditions.append(dataframe["close"] > dataframe["ema_trend"])  # Цена выше трендовой EMA

        # 4. Объем выше среднего
        conditions.append(
            dataframe["volume"] > dataframe["volume_ma"] * 1.2
        )  # Более строгое условие по объему

        # 5. Волатильность в норме (полосы Боллинджера не слишком узкие)
        bb_width = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]
        conditions.append(
            bb_width > bb_width.rolling(20).mean() * 0.7
        )  # Ширина канала выше 70% от среднего

        # 6. Фильтр волатильности (ATR)
        atr = ta.ATR(dataframe, timeperiod=14)
        atr_ma = ta.SMA(atr, timeperiod=20)
        conditions.append(atr > atr_ma * 0.7)  # ATR выше 70% от своего среднего

        # Применяем условия
        all_conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[all_conditions, ["enter_long", "enter_tag"]] = (1, "bollinger_rsi_entry")

        # Логируем информацию о текущей свече
        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]

            # Логируем состояние условий входа

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")
        dataframe["exit_long"] = 0

        # Если нет данных, выходим
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        # Проверяем наличие необходимых индикаторов
        required_indicators = ["bb_upper", "bb_middle", "rsi", "ema_fast", "ema_slow", "ema_trend"]
        for indicator in required_indicators:
            if indicator not in dataframe.columns:
                logger.warning(f"Отсутствует индикатор {indicator} для {pair}")
                return dataframe

        # Рассчитываем дополнительные индикаторы
        atr = ta.ATR(dataframe, timeperiod=14)

        # Условия выхода
        exit_conditions = []

        # 1. Цена достигла тейк-профита (1.5 ATR от цены входа)
        if hasattr(self, "dp") and hasattr(self.dp, "trade"):
            open_trades = [t for t in self.dp.trade if t.pair == pair and t.is_open]
            if open_trades:
                trade = open_trades[-1]
                atr_multiplier = 1.5  # Множитель ATR для тейк-профита
                take_profit = trade.open_rate + (atr * atr_multiplier)
                exit_conditions.append(dataframe["close"] >= take_profit)

        # 2. Цена достигла стоп-лосса (0.8 ATR от цены входа)
        if hasattr(self, "dp") and hasattr(self.dp, "trade"):
            if open_trades:
                trade = open_trades[-1]
                stop_loss = trade.open_rate - (atr * 0.8)
                exit_conditions.append(dataframe["close"] <= stop_loss)

        # 3. RSI выше 70 (перекупленность)
        exit_conditions.append(dataframe["rsi"] > 70)

        # 4. Разворот тренда (быстрая EMA пересекла медленную вниз)
        exit_conditions.append(
            (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
        )

        # 5. Цена ниже трендовой EMA
        exit_conditions.append(dataframe["close"] < dataframe["ema_trend"])

        # 6. Слишком высокий объем (возможен разворот)
        volume_ma = ta.SMA(dataframe["volume"], timeperiod=20)
        # Заменяем inf на NaN и заполняем пропуски
        volume_ma = pd.Series(volume_ma).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
        return dataframe

    # Инициализируем колонки
    dataframe["enter_long"] = 0
    dataframe["enter_tag"] = ""

    # Проверяем наличие необходимых индикаторов
    required_indicators = [
        "bb_lower",
        "bb_middle",
        "rsi",
        "volume",
        "ema_fast",
        "ema_slow",
        "ema_trend",
    ]
    for indicator in required_indicators:
        if indicator not in dataframe.columns:
            logger.warning(f"Отсутствует индикатор {indicator} для {pair}")
            return dataframe

    # Рассчитываем дополнительные индикаторы
    dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)

    # Условия входа
    conditions = []

    # 1. Цена коснулась или пробила нижнюю полосу Боллинджера
    conditions.append(dataframe["close"] <= dataframe["bb_lower"] * 1.001)

    # 2. RSI в зоне перепроданности
    conditions.append(dataframe["rsi"] < 35)  # Более строгое условие

    # 3. Подтверждение восходящего тренда
    conditions.append(dataframe["ema_fast"] > dataframe["ema_slow"])  # Быстрая EMA выше медленной
    conditions.append(dataframe["close"] > dataframe["ema_trend"])  # Цена выше трендовой EMA

    # 4. Объем выше среднего
    conditions.append(
        dataframe["volume"] > dataframe["volume_ma"] * 1.2
    )  # Более строгое условие по объему

    # 5. Волатильность в норме (полосы Боллинджера не слишком узкие)
    bb_width = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]
    conditions.append(
        bb_width > bb_width.rolling(20).mean() * 0.7
    )  # Ширина канала выше 70% от среднего

    # 6. Фильтр волатильности (ATR)
    atr = ta.ATR(dataframe, timeperiod=14)
    atr_ma = ta.SMA(atr, timeperiod=20)
    conditions.append(atr > atr_ma * 0.7)  # ATR выше 70% от своего среднего

    # Применяем условия
    all_conditions = reduce(lambda x, y: x & y, conditions)
    dataframe.loc[all_conditions, ["enter_long", "enter_tag"]] = (1, "bollinger_rsi_entry")

    # Логируем информацию о текущей свече
    if len(dataframe) > 0:
        last_row = dataframe.iloc[-1]

        # Логируем состояние условий входа, если они не выполняются
        if not all_conditions.any():
            logger.info(f"\n=== АНАЛИЗ УСЛОВИЙ ВХОДА {pair} ===")
            logger.info(f"1. Цена <= BB Lower: {last_row['close'] <= last_row['bb_lower'] * 1.001}")
            logger.info(f"   (Цена: {last_row['close']:.8f}, BB Lower: {last_row['bb_lower']:.8f})")

            logger.info(f"2. RSI < 35: {last_row['rsi'] < 35} (Текущий RSI: {last_row['rsi']:.2f})")

            logger.info(f"3. EMA Fast > EMA Slow: {last_row['ema_fast'] > last_row['ema_slow']}")
            logger.info(
                f"   (EMA Fast: {last_row['ema_fast']:.8f}, EMA Slow: {last_row['ema_slow']:.8f})"
            )

            logger.info(f"4. Цена > EMA Trend: {last_row['close'] > last_row['ema_trend']}")
            logger.info(
                f"   (Цена: {last_row['close']:.8f}, EMA Trend: {last_row['ema_trend']:.8f})"
            )

            # Проверяем объем
            vol_ma = last_row.get("volume_ma", 0)
            volume_condition = last_row["volume"] > vol_ma * 1.2
            logger.info(f"5. Объем > MA(20) * 1.2: {volume_condition}")
            logger.info(f"   (Объем: {last_row['volume']:.2f}, MA(20): {vol_ma:.2f})")

            # Проверяем ширину полос Боллинджера
            bb_width = (last_row["bb_upper"] - last_row["bb_lower"]) / last_row["bb_middle"]
            bb_ma = (
                dataframe["bb_upper"].rolling(20).mean()
                - dataframe["bb_lower"].rolling(20).mean()
                / dataframe["bb_middle"].rolling(20).mean()
            )
            bb_condition = bb_width > bb_ma.iloc[-1] * 0.7 if len(bb_ma) > 0 else False
            logger.info(f"6. Ширина BB > 70% от среднего: {bb_condition}")

            atr = ta.ATR(dataframe, timeperiod=14).iloc[-1]
            atr_ma = (
                ta.SMA(ta.ATR(dataframe, timeperiod=14), timeperiod=20).iloc[-1]
                if len(dataframe) >= 20
                else atr
            )
            atr_condition = atr > atr_ma * 0.7
            logger.info(
                f"7. ATR > 70% от среднего: {atr_condition} (ATR: {atr:.8f}, ATR MA: {atr_ma:.8f})"
            )

        # Логируем сигнал на покупку, если он есть
        if last_row["enter_long"] == 1:
            logger.info(f"\n=== СИГНАЛ НА ПОКУПКУ {pair} ===")
            logger.info(f"Цена: {last_row['close']:.8f}")
            logger.info(f"BB Lower: {last_row['bb_lower']:.8f}")
            logger.info(f"RSI: {last_row['rsi']:.2f}")
            vol_ma_value = last_row.get("volume_ma", 0)
            logger.info(f"Объем: {last_row['volume']:.2f} (MA: {vol_ma_value:.2f})")
            logger.info(
                f"EMA Fast: {last_row['ema_fast']:.8f}, EMA Slow: {last_row['ema_slow']:.8f}"
            )

    return dataframe


def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    pair = metadata.get("pair", "")
    dataframe["exit_long"] = 0

    # Если нет данных, выходим
    if len(dataframe) < self.startup_candle_count:
        return dataframe

    # Проверяем наличие необходимых индикаторов
    required_indicators = ["bb_upper", "bb_middle", "rsi", "ema_fast", "ema_slow", "ema_trend"]
    for indicator in required_indicators:
        if indicator not in dataframe.columns:
            logger.warning(f"Отсутствует индикатор {indicator} для {pair}")
            return dataframe

    # Рассчитываем дополнительные индикаторы
    atr = ta.ATR(dataframe, timeperiod=14)

    # Условия выхода
    exit_conditions = []

    # 1. Цена достигла тейк-профита (1.5 ATR от цены входа)
    if hasattr(self, "dp") and hasattr(self.dp, "trade"):
        open_trades = [t for t in self.dp.trade if t.pair == pair and t.is_open]
        if open_trades:
            trade = open_trades[-1]
            atr_multiplier = 1.5  # Множитель ATR для тейк-профита
            take_profit = trade.open_rate + (atr * atr_multiplier)
            exit_conditions.append(dataframe["close"] >= take_profit)

    # 2. Цена достигла стоп-лосса (0.8 ATR от цены входа)
    if hasattr(self, "dp") and hasattr(self.dp, "trade"):
        if open_trades:
            trade = open_trades[-1]
            stop_loss = trade.open_rate - (atr * 0.8)
            exit_conditions.append(dataframe["close"] <= stop_loss)

    # 3. RSI выше 70 (перекупленность)
    exit_conditions.append(dataframe["rsi"] > 70)

    # 4. Разворот тренда (быстрая EMA пересекла медленную вниз)
    exit_conditions.append(
        (dataframe["ema_fast"] < dataframe["ema_slow"])
        & (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
    )

    # 5. Цена ниже трендовой EMA
    exit_conditions.append(dataframe["close"] < dataframe["ema_trend"])

    # 6. Слишком высокий объем (возможен разворот)
    volume_ma = ta.SMA(dataframe["volume"], timeperiod=20)
    # Заменяем inf на NaN и заполняем пропуски
    volume_ma = pd.Series(volume_ma).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
    exit_conditions.append(dataframe["volume"] > volume_ma * 3.0)

    # Сохраняем volume_ma в датафрейм для логирования
    dataframe["volume_ma"] = volume_ma

    # Применяем условия выхода
    dataframe.loc[reduce(lambda x, y: x | y, exit_conditions), "exit_long"] = 1

    # Логируем информацию о текущей свече
    if len(dataframe) > 0:
        last_row = dataframe.iloc[-1]
        if last_row["exit_long"] == 1:
            logger.info(f"\n=== СИГНАЛ НА ПРОДАЖУ {pair} ===")
            logger.info(f"Цена: {last_row['close']:.8f}")
            logger.info(f"RSI: {last_row['rsi']:.2f}")
            logger.info(
                f"EMA Fast: {last_row['ema_fast']:.8f}, EMA Slow: {last_row['ema_slow']:.8f}"
            )
            # Безопасное логирование объема
            vol_ma_value = last_row.get("volume_ma", 0)
            logger.info(f"Объем: {last_row['volume']:.2f} (MA: {vol_ma_value:.2f})")

    return dataframe
