import logging
from datetime import datetime
from typing import Dict, List, Optional
from functools import reduce

import pandas as pd
import pandas_ta as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, FloatParameter
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import informative
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class MarketStateStrategy(IStrategy):
    """
    Стратегия, адаптирующаяся под текущее состояние рынка (растущий, падающий, боковой)
    """

    # Стратегия параметры
    INTERFACE_VERSION = 3

    # Оптимизированные параметры для торговли
    minimal_roi = {
        "0": 0.05,  # 5% сразу
        "30": 0.03,  # 3% через 30 минут
        "60": 0.02,  # 2% через 1 час
        "120": 0.01,  # 1% через 2 часа
    }

    stoploss = -0.02  # Уменьшаем стоп-лосс до 2%

    # Таймфрейм
    timeframe = "15m"  # Увеличиваем таймфрейм для уменьшения количества сделок

    # Параметры для входа в позицию
    buy_rsi = IntParameter(20, 40, default=30, space="buy")  # Расширяем диапазон RSI
    sell_rsi = IntParameter(60, 80, default=70, space="sell")

    # Параметры для фильтрации сигналов
    min_adx = IntParameter(10, 25, default=15, space="buy")  # Снижаем минимальный ADX
    min_volume = DecimalParameter(0.5, 1.5, default=1.0, space="buy")  # Снижаем требования к объему
    min_distance_ema = DecimalParameter(0.001, 0.005, default=0.002, space="buy")
    min_atr_ratio = DecimalParameter(0.5, 1.5, default=1.0, space="buy")

    # Параметры для trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1%
    trailing_stop_positive_offset = 0.02  # 2%
    trailing_only_offset_is_reached = True

    # Параметры для частичного закрытия позиций
    use_partial_exit = True
    partial_exit_1 = 0.4  # 40% позиции
    partial_exit_2 = 0.3  # 30% позиции
    partial_exit_profit_1 = 0.025  # 2.5% прибыли для первого закрытия
    partial_exit_profit_2 = 0.045  # 4.5% прибыли для второго закрытия

    # Параметры для пар с низкой эффективностью
    low_perf_pairs = ["SOL/USDT", "XRP/USDT"]
    low_perf_min_distance = DecimalParameter(0.02, 0.04, default=0.03, space="buy")
    low_perf_min_volume = DecimalParameter(2.0, 4.0, default=3.0, space="buy")

    # Защитные механизмы
    protections = [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 48,
            "trade_limit": 1,
            "stop_duration_candles": 24,
            "only_per_pair": True,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 72,
            "trade_limit": 1,
            "stop_duration_candles": 36,
            "max_allowed_drawdown": 0.12,
        },
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 48,
            "trade_limit": 1,
            "stop_duration_candles": 24,
            "required_profit": 0.02,
        },
    ]

    # Добавляем новые параметры для управления рисками
    position_size_atr_multiplier = DecimalParameter(1.0, 3.0, default=2.0, space="buy")
    max_position_size = DecimalParameter(0.1, 0.3, default=0.2, space="buy")
    min_position_size = DecimalParameter(0.05, 0.1, default=0.07, space="buy")

    # Добавляем параметры для мультитаймфрейм анализа
    informative_timeframe_1h = "1h"
    informative_timeframe_4h = "4h"

    # Использование информативных таймфреймов
    @informative(informative_timeframe_1h)
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Анализ часового таймфрейма
        dataframe["ema_20_1h"] = ta.ema(dataframe["close"], length=20)
        dataframe["ema_50_1h"] = ta.ema(dataframe["close"], length=50)
        dataframe["ema_200_1h"] = ta.ema(dataframe["close"], length=200)

        # Определение тренда на часовом таймфрейме
        dataframe["trend_1h"] = 0
        dataframe.loc[
            (dataframe["ema_20_1h"] > dataframe["ema_50_1h"])
            & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"]),
            "trend_1h",
        ] = 1  # Восходящий тренд
        dataframe.loc[
            (dataframe["ema_20_1h"] < dataframe["ema_50_1h"])
            & (dataframe["ema_50_1h"] < dataframe["ema_200_1h"]),
            "trend_1h",
        ] = -1  # Нисходящий тренд

        # Определение силы тренда на часовом таймфрейме
        dataframe.ta.adx(length=14, append=True)
        dataframe["adx_1h"] = dataframe["ADX_14"]
        dataframe["di_plus_1h"] = dataframe["DMP_14"]
        dataframe["di_minus_1h"] = dataframe["DMN_14"]

        # Определение волатильности на часовом таймфрейме
        dataframe["atr_1h"] = ta.atr(
            dataframe["high"], dataframe["low"], dataframe["close"], length=14
        )
        dataframe["atr_mean_1h"] = dataframe["atr_1h"].rolling(window=20).mean()
        dataframe["atr_ratio_1h"] = dataframe["atr_1h"] / dataframe["atr_mean_1h"]

        # Определение объемного профиля на часовом таймфрейме
        dataframe["volume_ma_1h"] = ta.sma(dataframe["volume"], length=20)
        dataframe["volume_std_1h"] = dataframe["volume"].rolling(window=20).std()
        dataframe["volume_z_score_1h"] = (
            dataframe["volume"] - dataframe["volume_ma_1h"]
        ) / dataframe["volume_std_1h"]

        # Добавляем RSI и Stochastic на часовом таймфрейме
        dataframe["rsi_1h"] = ta.rsi(dataframe["close"], length=14)
        stoch = ta.stoch(dataframe["high"], dataframe["low"], dataframe["close"])
        dataframe["slowk_1h"] = stoch["STOCHk_14_3_3"]
        dataframe["slowd_1h"] = stoch["STOCHd_14_3_3"]

        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_ma_1h"]

        return dataframe

    @informative(informative_timeframe_4h)
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Анализ 4-часового таймфрейма
        dataframe["ema_20_4h"] = ta.ema(dataframe["close"], length=20)
        dataframe["ema_50_4h"] = ta.ema(dataframe["close"], length=50)
        dataframe["ema_200_4h"] = ta.ema(dataframe["close"], length=200)

        # Определение тренда на 4-часовом таймфрейме
        dataframe["trend_4h"] = 0
        dataframe.loc[
            (dataframe["ema_20_4h"] > dataframe["ema_50_4h"])
            & (dataframe["ema_50_4h"] > dataframe["ema_200_4h"]),
            "trend_4h",
        ] = 1  # Восходящий тренд
        dataframe.loc[
            (dataframe["ema_20_4h"] < dataframe["ema_50_4h"])
            & (dataframe["ema_50_4h"] < dataframe["ema_200_4h"]),
            "trend_4h",
        ] = -1  # Нисходящий тренд

        # Определение силы тренда на 4-часовом таймфрейме
        dataframe.ta.adx(length=14, append=True)
        dataframe["adx_4h"] = dataframe["ADX_14"]
        dataframe["di_plus_4h"] = dataframe["DMP_14"]
        dataframe["di_minus_4h"] = dataframe["DMN_14"]

        # Определение ключевых уровней на 4-часовом таймфрейме
        dataframe["support_4h"] = dataframe["low"].rolling(window=20).min()
        dataframe["resistance_4h"] = dataframe["high"].rolling(window=20).max()

        # Добавляем RSI на 4-часовом таймфрейме
        dataframe["rsi_4h"] = ta.rsi(dataframe["close"], length=14)

        dataframe["volume_ma"] = ta.sma(dataframe["volume"], length=20)
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_ma"]

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Базовые индикаторы
        dataframe["ema_20"] = ta.ema(dataframe["close"], length=20)
        dataframe["ema_50"] = ta.ema(dataframe["close"], length=50)
        dataframe["ema_200"] = ta.ema(dataframe["close"], length=200)

        # ADX и DI
        dataframe.ta.adx(length=14, append=True)
        dataframe["adx"] = dataframe["ADX_14"]
        dataframe["di_plus"] = dataframe["DMP_14"]
        dataframe["di_minus"] = dataframe["DMN_14"]

        # RSI
        dataframe["rsi_14"] = ta.rsi(dataframe["close"], length=14)
        dataframe["rsi_1h"] = ta.rsi(dataframe["close"], length=60)

        # Объем
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_std"] = dataframe["volume"].rolling(window=20).std()
        dataframe["volume_z_score"] = (dataframe["volume"] - dataframe["volume_mean"]) / dataframe[
            "volume_std"
        ]
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_mean"]

        # ATR
        dataframe["atr"] = ta.atr(
            dataframe["high"], dataframe["low"], dataframe["close"], length=14
        )
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["close"]

        # Моментум
        dataframe["momentum"] = ta.mom(dataframe["close"], length=10)
        dataframe["momentum_ratio"] = dataframe["momentum"] / dataframe["close"]

        # Структура рынка
        dataframe["higher_highs"] = (dataframe["high"] > dataframe["high"].shift(1)) & (
            dataframe["high"].shift(1) > dataframe["high"].shift(2)
        )
        dataframe["lower_lows"] = (dataframe["low"] < dataframe["low"].shift(1)) & (
            dataframe["low"].shift(1) < dataframe["low"].shift(2)
        )

        # Тренд на 1h
        dataframe["trend_1h"] = (dataframe["ema_20"] - dataframe["ema_50"]) / dataframe["ema_50"]

        # Улучшенное определение состояния рынка
        dataframe["market_state"] = "unknown"

        # Растущий рынок (смягченные условия)
        uptrend_conditions = (
            (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["adx"] > self.min_adx.value * 0.8)  # Снижаем требования к ADX
            & (dataframe["volume"] > dataframe["volume_mean"] * 0.8)  # Снижаем требования к объему
            & (dataframe["di_plus"] > dataframe["di_minus"])  # Убираем множитель
            & (dataframe["momentum_ratio"] > 0.2)  # Снижаем требования к моментуму
        )

        # Падающий рынок (смягченные условия)
        downtrend_conditions = (
            (dataframe["ema_20"] < dataframe["ema_50"])
            & (dataframe["adx"] > self.min_adx.value * 0.8)  # Снижаем требования к ADX
            & (dataframe["volume"] > dataframe["volume_mean"] * 0.8)  # Снижаем требования к объему
            & (dataframe["di_minus"] > dataframe["di_plus"])  # Убираем множитель
            & (dataframe["momentum_ratio"] < -0.2)  # Снижаем требования к моментуму
        )

        # Боковой рынок (смягченные условия)
        sideways_conditions = (
            (
                abs(dataframe["ema_20"] - dataframe["ema_50"]) / dataframe["ema_50"] < 0.015
            )  # Увеличиваем допустимое отклонение
            & (dataframe["volume"] > dataframe["volume_mean"] * 0.7)  # Снижаем требования к объему
            & (abs(dataframe["momentum_ratio"]) < 0.4)  # Увеличиваем допустимый моментум
            & (dataframe["adx"] < self.min_adx.value)  # Снижаем требования к ADX
        )

        # Применяем условия
        dataframe.loc[uptrend_conditions, "market_state"] = "uptrend"
        dataframe.loc[downtrend_conditions, "market_state"] = "downtrend"
        dataframe.loc[sideways_conditions, "market_state"] = "sideways"

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Смягченные условия входа
        trend_condition = (
            (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["adx"] > self.min_adx.value * 0.8)  # Снижаем требования к ADX
            & (dataframe["di_plus"] > dataframe["di_minus"])  # Убираем множитель
        )

        volume_condition = (
            (dataframe["volume"] > dataframe["volume_mean"] * 0.8)  # Снижаем требования к объему
            & (dataframe["volume_z_score"] < 2.5)  # Увеличиваем допустимый z-score
        )

        rsi_condition = (
            (dataframe["rsi_14"] < self.buy_rsi.value)
            & (dataframe["rsi_14"] > 20)  # Снижаем нижнюю границу
            & (dataframe["rsi_1h"] < 75)  # Увеличиваем верхнюю границу
        )

        market_structure_condition = (
            (dataframe["higher_highs"])
            & (dataframe["volume"] > dataframe["volume_mean"] * 0.7)  # Снижаем требования к объему
            & (dataframe["momentum_ratio"] > 0.2)  # Снижаем требования к моментуму
        )

        volatility_condition = (
            (dataframe["atr_ratio"] < 0.03)  # Увеличиваем допустимую волатильность
            & (dataframe["volume_z_score"] < 2.5)  # Увеличиваем допустимый z-score
        )

        # Комбинируем все условия
        conditions.append(
            trend_condition
            & volume_condition
            & rsi_condition
            & market_structure_condition
            & volatility_condition
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Смягченные условия выхода
        trend_condition = (
            (dataframe["ema_20"] < dataframe["ema_50"])
            & (dataframe["adx"] < self.min_adx.value * 0.7)  # Снижаем требования к ADX
        )

        rsi_condition = (
            (dataframe["rsi_14"] > self.sell_rsi.value)
            & (dataframe["rsi_1h"] > 80)  # Увеличиваем верхнюю границу
        )

        volatility_condition = (
            (dataframe["volume_z_score"] > 3.0)  # Увеличиваем допустимый z-score
            & (dataframe["atr_ratio"] > 0.04)  # Увеличиваем допустимую волатильность
        )

        market_structure_condition = (
            (dataframe["lower_lows"])
            & (dataframe["volume"] > dataframe["volume_mean"] * 1.2)  # Снижаем требования к объему
        )

        # Комбинируем условия
        conditions.append(
            trend_condition | rsi_condition | volatility_condition | market_structure_condition
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "exit_long"] = 1

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
        Кастомный стоп-лосс, динамически адаптирующийся к рыночным условиям
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Адаптивный множитель ATR
        atr_multiplier = 2.0

        # Адаптация к состоянию рынка
        if last_candle["market_state"] == "uptrend":
            atr_multiplier = 2.2
        elif last_candle["market_state"] == "downtrend":
            atr_multiplier = 1.8
        else:  # sideways
            atr_multiplier = 1.5

        # Адаптация к волатильности
        if last_candle["volume_z_score"] > 1.5:
            atr_multiplier *= 1.2
        elif last_candle["volume_z_score"] < -0.5:
            atr_multiplier *= 0.8

        # Защита прибыли
        if current_profit > 0.04:  # Если прибыль больше 4%
            return -0.015  # Устанавливаем стоп-лосс на 1.5%
        elif current_profit > 0.02:  # Если прибыль больше 2%
            return -0.02  # Устанавливаем стоп-лосс на 2%

        # Расчет стоп-лосса на основе ATR
        atr = last_candle["atr"]
        stoploss = atr * atr_multiplier / current_rate

        return -stoploss

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Частичное закрытие позиций с улучшенной логикой
        if current_profit >= self.partial_exit_profit_2:
            # Если рынок меняет состояние, закрываем больше
            if (trade.is_short and last_candle["market_state"] == "uptrend") or (
                not trade.is_short and last_candle["market_state"] == "downtrend"
            ):
                return f"partial_exit_2_{self.partial_exit_2 * 1.5}"
            return f"partial_exit_2_{self.partial_exit_2}"

        elif current_profit >= self.partial_exit_profit_1:
            # Если высокая волатильность, закрываем больше
            if last_candle["volume_z_score"] > 2.0:
                return f"partial_exit_1_{self.partial_exit_1 * 1.2}"
            return f"partial_exit_1_{self.partial_exit_1}"

        # Выход по изменению состояния рынка с подтверждением
        if trade.is_short:
            if (
                last_candle["market_state"] == "uptrend"
                and last_candle["momentum_ratio"] > 1.0
                and last_candle["volume_z_score"] > 0.5
            ):
                return "market_state_change"
        else:
            if (
                last_candle["market_state"] == "downtrend"
                and last_candle["momentum_ratio"] < -1.0
                and last_candle["volume_z_score"] > 0.5
            ):
                return "market_state_change"

        # Выход по волатильности с учетом состояния рынка
        if last_candle["volume_z_score"] > 2.5 and current_profit > 0:
            if (trade.is_short and last_candle["market_state"] == "sideways") or (
                not trade.is_short and last_candle["market_state"] == "sideways"
            ):
                return "high_volatility_sideways"

        return None

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Корректируем цену входа в зависимости от волатильности
        if last_candle["volume_z_score"] > 1.5:
            # При высокой волатильности даем больше пространства
            return proposed_rate * (1.02 if side == "long" else 0.98)
        return proposed_rate

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Базовый множитель плеча в зависимости от состояния рынка
        if last_candle["market_state"] == "uptrend":
            base_leverage = 1.2
        elif last_candle["market_state"] == "downtrend":
            base_leverage = 0.8
        else:  # sideways
            base_leverage = 0.5

        # Корректируем плечо на основе волатильности
        volatility_factor = 1.0
        if last_candle["volume_z_score"] > 1.5:
            volatility_factor = 0.7
        elif last_candle["volume_z_score"] < -0.5:
            volatility_factor = 1.2

        # Корректируем плечо на основе силы тренда
        trend_factor = 1.0
        if last_candle["trend_strength"] > 25:
            trend_factor = 1.1
        elif last_candle["trend_strength"] < 15:
            trend_factor = 0.9

        final_leverage = proposed_leverage * base_leverage * volatility_factor * trend_factor
        return min(final_leverage, max_leverage)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Проверка волатильности
        if last_candle["volume_z_score"] > 2.5:
            return False

        # Проверка силы тренда
        if last_candle["trend_strength"] < 15 and last_candle["market_state"] != "sideways":
            return False

        # Проверка структуры свечей
        if last_candle["candle_ratio"] > 0.8 and last_candle["market_state"] == "sideways":
            return False

        return True

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Вызывается в начале каждой итерации бота
        """
        # Получаем данные для всех пар
        for pair in self.dp.current_whitelist():
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty:
                last_candle = dataframe.iloc[-1].squeeze()
                logger.info(
                    f"Pair: {pair}, Market State: {last_candle['market_state']}, "
                    f"ADX: {last_candle['adx']:.2f}, "
                    f"ATR: {last_candle['atr']:.2f}"
                )


"""
Рекомендации по использованию:
1.Регулярность анализа:
-Используйте разные таймфреймы для анализа (например, 1h для общего тренда и 5m для входа)
-Обновляйте анализ при каждой новой свече
-Ведите логи состояния рынка для последующего анализа
2.Фильтры состояния рынка:
-Используйте несколько индикаторов для подтверждения состояния
-Учитывайте волатильность рынка
-Адаптируйте параметры стратегии под текущее состояние
3.Управление рисками:
-Адаптируйте размер позиции под состояние рынка
-Используйте разные стоп-лоссы для разных состояний
-Учитывайте волатильность при расчете тейк-профитов
4.Оптимизация:
-Регулярно проводите бэктестинг стратегии
-Анализируйте эффективность в разных рыночных условиях
-Корректируйте параметры на основе результатов
5.Дополнительные рекомендации:
-Используйте защитные механизмы (protections) для разных состояний рынка
-Настройте уведомления о смене состояния рынка
-Ведите статистику успешности стратегии в разных условиях
Этот подход позволит вам:
-Автоматически определять текущее состояние рынка
-Адаптировать стратегию под рыночные условия
-Улучшить управление рисками
Повысить эффективность торговли в разных рыночных условиях
"""
