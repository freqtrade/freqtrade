from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, RealParameter, Trade
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
    minimal_roi = {
        "0": 0.03,  # 3% ROI - продаем только если ROI > 3%
        "30": 0.02,  # Через 30 минут снижаем до 2%
        "60": 0.01,  # Через 60 минут снижаем до 1%
        "120": 0,  # Через 120 минут выходим в любом случае
    }

    # Оптимизированные параметры
    stoploss = -0.01  # -1% стоп-лосс по умолчанию
    use_custom_stoploss = True

    # Динамический стоп-лосс на основе волатильности
    stoploss_on_exchange = True
    stoploss_on_exchange_interval = 60  # Проверять стоп-лосс каждую минуту
    stoploss_on_exchange_limit_ratio = 0.99  # Лимитный ордер на 1% хуже текущей цены

    # Трейлинг стоп
    trailing_stop = True
    trailing_stop_positive = 0.01  # Начинаем трейлинг после 1% прибыли
    trailing_stop_positive_offset = 0.02  # Начинаем трейлинг после 2% прибыли
    trailing_only_offset_is_reached = True

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавляем индикаторы в датафрейм
        """
        # Полосы Боллинджера
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_middle"] = bollinger["mid"]
        dataframe["bb_lower"] = bollinger["lower"]

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_length)

        # EMA для определения тренда
        dataframe["ema12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        # Объемы
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        # ATR для стоп-лосса
        try:
            # Используем pandas_ta для более надежного расчета ATR
            import pandas_ta as pta

            atr = pta.atr(
                high=dataframe["high"],
                low=dataframe["low"],
                close=dataframe["close"],
                length=self.atr_period,
            )
            dataframe["atr"] = atr
        except Exception as e:
            logger.error(f"Ошибка при расчете ATR: {str(e)}")
            # Используем простой расчет если pandas_ta не доступен
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)

        # MACD
        try:
            # Используем встроенный расчет MACD из TA-Lib
            macd, signal, hist = ta.MACD(
                dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
            )
            dataframe["macd_line"] = macd
            dataframe["macd_signal"] = signal
            dataframe["macd_hist"] = hist
        except Exception as e:
            logger.error(f"Ошибка при расчете MACD: {str(e)}")
            dataframe["macd_line"] = 0.0
            dataframe["macd_signal"] = 0.0
            dataframe["macd_hist"] = 0.0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        # Инициализируем колонки
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        # Условия входа
        conditions = [
            # 1. Цена ниже верхней полосы Боллинджера
            (dataframe["close"] < dataframe["bb_upper"]),
            # 2. RSI в зоне перепроданности (30-50)
            (dataframe["rsi"] < 50) & (dataframe["rsi"] > 30),
            # 3. Восходящий тренд (быстрая EMA выше медленной)
            (dataframe["ema12"] > dataframe["ema26"]) & (dataframe["ema26"] > dataframe["ema50"]),
            # 4. MACD гистограмма растет
            (dataframe["macd_hist"] > dataframe["macd_hist"].shift(1))
            & (dataframe["macd_hist"] > 0),
            # 5. Объем выше среднего
            (dataframe["volume"] > dataframe["volume_ma"] * 1.2),
            # 6. Цена выше EMA50
            (dataframe["close"] > dataframe["ema50"]),
        ]

        # Применяем все условия
        if conditions:
            all_conditions = reduce(lambda x, y: x & y, conditions)
            dataframe.loc[all_conditions, "enter_long"] = 1
            dataframe.loc[all_conditions, "enter_tag"] = "bollinger_rsi_entry"

            # Устанавливаем стоп-лосс и тейк-профит
            dataframe.loc[dataframe["enter_long"] == 1, "stop_loss"] = dataframe["close"] - (
                2 * dataframe["atr"]
            )
            dataframe.loc[dataframe["enter_long"] == 1, "stoploss_pct"] = (
                dataframe["close"] - (2 * dataframe["atr"])
            ) / dataframe["close"] - 1

            # Логируем информацию о входных условиях
            if len(dataframe) > 0 and any(dataframe["enter_long"] == 1):
                last_signal = dataframe[dataframe["enter_long"] == 1].iloc[-1]
                logger.info(f"\n=== СИГНАЛ НА ПОКУПКУ {pair} ===")
                logger.info(f"Цена: {last_signal['close']:.8f}")
                logger.info(f"RSI: {last_signal['rsi']:.2f}")
                logger.info(
                    f"MACD: {last_signal['macd_line']:.8f}, Signal: {last_signal['macd_signal']:.8f}"
                )
                logger.info(
                    f"Стоп-лосс: {last_signal['stop_loss']:.8f} ({last_signal['stoploss_pct'] * 100:.2f}%)"
                )

        return dataframe

        # Логируем информацию о текущей свече
        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]

            # Упрощенные условия входа
            condition1 = True  # Всегда истина, чтобы видеть логи

            # Логируем все условия входа с подробностями
            logger.info(f"\n=== УПРОЩЕННЫЙ АНАЛИЗ УСЛОВИЙ ВХОДА {pair} ===")

            # 1. Проверяем только что цена ниже верхней полосы Боллинджера
            bb_upper = last_row["bb_upper"]
            price = last_row["close"]
            condition_bb = price < bb_upper
            logger.info(
                f"1. Цена < BB Upper: {condition_bb} (Цена: {price:.8f}, BB Upper: {bb_upper:.8f})"
            )

            # 2. Проверяем RSI в расширенном диапазоне
            rsi = last_row["rsi"]
            condition_rsi = (rsi < 60) & (rsi > 20)  # Расширяем диапазон RSI
            logger.info(f"2. RSI 20-60: {condition_rsi} (Текущий RSI: {rsi:.2f})")

            # 3. Проверяем что цена выше EMA50
            ema50 = last_row["ema50"]  # Используем уже рассчитанное значение EMA50
            condition_ema = price > ema50 * 0.98  # Снижаем порог с 0.99 до 0.98
            logger.info(
                f"3. Цена > 0.98*EMA50: {condition_ema} (Цена: {price:.8f}, EMA50: {ema50:.8f})"
            )

            # 4. Проверяем что MACD выше сигнальной линии
            condition_macd = last_row["macd_line"] > last_row["macd_signal"]
            logger.info(
                f"4. MACD > Signal: {condition_macd} (MACD: {last_row['macd_line']:.8f}, Signal: {last_row['macd_signal']:.8f})"
            )

            # Формируем итоговые условия входа
            conditions = [
                condition_bb,
                condition_rsi,
                condition_ema,
                condition_macd,
                (dataframe["volume"] > dataframe["volume_ma"] * 1.2),  # Проверка объема
            ]

            # Добавляем сигнал на покупку если все условия выполнены
            all_conditions = reduce(lambda x, y: x & y, conditions)
            dataframe.loc[all_conditions, "enter_long"] = 1
            dataframe.loc[all_conditions, "enter_tag"] = "simplified_entry"

            # Устанавливаем стоп-лосс
            dataframe.loc[all_conditions, "stop_loss"] = dataframe["close"] - (2 * dataframe["atr"])
            dataframe.loc[all_conditions, "stoploss_pct"] = (
                dataframe["close"] - (2 * dataframe["atr"])
            ) / dataframe["close"] - 1

            # Логируем итоговое решение
            logger.info(f"\n=== ИТОГОВОЕ РЕШЕНИЕ ===")
            logger.info(f"Все условия выполнены: {all_conditions.any()}")
            if not all_conditions.any():
                logger.info(
                    "Не выполнены условия: "
                    + ", ".join([str(i + 1) for i, cond in enumerate(conditions) if not cond.any()])
                )

            # Логируем сигнал на покупку, если он есть
            if all_conditions.any():
                last_signal = dataframe[all_conditions].iloc[-1]
                logger.info(f"\n=== СИГНАЛ НА ПОКУПКУ {pair} ===")
                logger.info(f"Время: {last_signal.name}")
                logger.info(f"Цена: {last_signal['close']:.8f}")
                logger.info(f"RSI: {last_signal['rsi']:.2f}")
                logger.info(
                    f"MACD: {last_signal['macd_line']:.8f}, Signal: {last_signal['macd_signal']:.8f}"
                )
                logger.info(
                    f"Стоп-лосс: {last_signal['stop_loss']:.8f} ({last_signal['stoploss_pct'] * 100:.2f}%)"
                )

            return dataframe
            logger.info(f"BB Middle: {last_row['bb_middle']:.8f}")
            logger.info(f"RSI: {last_row['rsi']:.2f}")
            logger.info(f"Объем: {last_row['volume']:.2f}")
            logger.info(
                f"EMA Fast: {last_row['ema_fast']:.8f}, EMA Slow: {last_row['ema_slow']:.8f}"
            )

            # Дополнительная отладочная информация
            logger.info(f"\n=== ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ===")
            logger.info(f"Время: {last_row.name}")
            logger.info(f"Открытие: {last_row['open']:.8f}")
            logger.info(f"Максимум: {last_row['high']:.8f}")
            logger.info(f"Минимум: {last_row['low']:.8f}")
            logger.info(f"Объем: {last_row['volume']:.2f}")
            logger.info(f"BB Upper: {last_row['bb_upper']:.8f}")
            logger.info(f"BB Lower: {last_row['bb_lower']:.8f}")

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Динамический стоп-лосс на основе волатильности
        """
        # Минимальный стоп-лосс 0.5%
        if current_profit < -0.005:
            return -1  # Закрыть сделку немедленно

        # Если прибыль больше 1%, перемещаем стоп в безубыток
        if current_profit > 0.01:
            return -0.001  # 0.1% стоп-лосс

        # По умолчанию используем 1% стоп-лосс
        return -0.01

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")
        dataframe["exit_long"] = 0

        # Если нет данных, выходим
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        # Рассчитываем индикаторы, если их нет
        if "atr" not in dataframe.columns:
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)

        if "macd_line" not in dataframe.columns or "macd_signal" not in dataframe.columns:
            try:
                import pandas_ta as pta

                macd = pta.macd(dataframe["close"], fast=12, slow=26, signal=9)
                dataframe["macd_line"] = macd.iloc[:, 0]
                dataframe["macd_signal"] = macd.iloc[:, 1]
                dataframe["macd_hist"] = macd.iloc[:, 2]
            except Exception as e:
                logger.error(f"Ошибка при расчете MACD: {str(e)}")
                return dataframe

        # Условия выхода
        exit_conditions = []

        # 1. RSI выше 70 (перекупленность)
        exit_conditions.append(dataframe["rsi"] > 70)

        # 2. Разворот тренда (быстрая EMA пересекла медленную вниз)
        ema_cross_down = (dataframe["ema12"] < dataframe["ema26"]) & (
            dataframe["ema12"].shift(1) >= dataframe["ema26"].shift(1)
        )
        exit_conditions.append(ema_cross_down)

        # 3. MACD гистограмма уменьшается
        macd_decreasing = (dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)) & (
            dataframe["macd_hist"] < 0
        )
        exit_conditions.append(macd_decreasing)

        # 4. Цена пересекла среднюю линию Боллинджера сверху вниз
        price_below_bb_middle = (dataframe["close"] < dataframe["bb_middle"]) & (
            dataframe["close"].shift(1) >= dataframe["bb_middle"].shift(1)
        )
        exit_conditions.append(price_below_bb_middle)

        # 5. Цена ниже EMA50 (смена тренда)
        price_below_ema50 = dataframe["close"] < dataframe["ema50"]
        exit_conditions.append(price_below_ema50)

        # Применяем условия выхода
        if exit_conditions:
            all_exit_conditions = reduce(lambda x, y: x | y, exit_conditions)
            dataframe.loc[all_exit_conditions, "exit_long"] = 1

            # Логируем выход из сделки
            if len(dataframe) > 0 and any(dataframe["exit_long"] == 1):
                last_exit = dataframe[dataframe["exit_long"] == 1].iloc[-1]
                logger.info(f"\n=== СИГНАЛ НА ПРОДАЖУ {pair} ===")
                logger.info(f"Цена: {last_exit['close']:.8f}")
                logger.info(f"RSI: {last_exit['rsi']:.2f}")
                logger.info(f"MACD Hist: {last_exit['macd_hist']:.8f}")
                logger.info(
                    f"Цена относительно BB Middle: {last_exit['close'] - last_exit['bb_middle']:.8f}"
                )

        return dataframe

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
        Кастомный стоп-лосс с трейлинг стопом
        """
        # Устанавливаем стоп-лосс в 2 ATR от цены входа
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1].squeeze()
            if "atr" in last_candle:
                atr = last_candle["atr"]
                stoploss = current_rate - (2 * atr)
                return -(stoploss / current_rate - 1)

        return self.stoploss

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[Union[str, bool]]:
        """
        Кастомный выход из сделки с тейк-профитом и трейлинг-стопом
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return None

        last_candle = dataframe.iloc[-1].squeeze()

        # Тейк-профит 2:1 к стоп-лоссу
        if current_profit > 0.02:  # 2% тейк-профит
            return "take_profit"

        # Трейлинг-стоп с отступом 1 ATR
        if current_profit > 0.01:  # Начинаем трейлинг после 1% прибыли
            atr = last_candle.get("atr", 0)
            if current_rate > (trade.open_rate * (1 + current_profit - 0.01)):
                return "trailing_stop"

        return None

        # 4. Стоп-лосс на основе ATR
        # Используем кумулятивный максимум цены с момента входа
        dataframe["rolling_high"] = dataframe["high"].cummax()
        # Стоп-лосс на 2 ATR ниже максимума
        atr = ta.ATR(dataframe, timeperiod=14)
        stop_loss = dataframe["rolling_high"] - (atr * 2)
        exit_conditions.append(dataframe["close"] < stop_loss)

        # Применяем условия выхода
        if exit_conditions:
            all_exit_conditions = reduce(lambda x, y: x | y, exit_conditions)
            dataframe.loc[all_exit_conditions, "exit_long"] = 1

            # Логируем срабатывание условий выхода
            if len(dataframe) > 0 and dataframe.iloc[-1]["exit_long"] == 1:
                last_row = dataframe.iloc[-1]
                logger.info(f"\n=== УСЛОВИЯ ВЫХОДА ДЛЯ {pair} ===")
                logger.info(f"1. RSI > 70: {last_row['rsi'] > 70} (RSI: {last_row['rsi']:.2f})")
                logger.info(f"2. EMA Fast < EMA Slow: {ema_cross_down.iloc[-1]}")
                logger.info(f"3. Цена > BB Upper: {last_row['close'] > last_row['bb_upper']}")
                logger.info(f"4. Цена < Stop Loss: {last_row['close'] < stop_loss.iloc[-1]}")
                logger.info(
                    f"   (Цена: {last_row['close']:.8f}, Стоп-лосс: {stop_loss.iloc[-1]:.8f})"
                )

        return dataframe
