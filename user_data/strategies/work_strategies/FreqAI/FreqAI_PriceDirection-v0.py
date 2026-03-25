import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


logger = logging.getLogger(__name__)


class FreqAI_PriceDirection(IStrategy):
    """
    Стратегия использует FreqAI для:
    1. Предсказания направления движения цены
    2. Оптимизации точек входа и выхода

    Требования:
    - Freqtrade с установленным FreqAI
    - В config.json должна быть секция "freqai"
    """

    # Основные параметры стратегии
    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.01,
    }

    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Таймфрейм
    timeframe = "5m"

    # Параметры для оптимизации
    buy_rsi_threshold = IntParameter(20, 40, default=30, space="buy")
    sell_rsi_threshold = IntParameter(60, 80, default=70, space="sell")

    # Параметры ML
    prediction_threshold = DecimalParameter(0.5, 0.9, default=0.65, space="buy")

    # Startup candle count
    startup_candle_count: int = 100

    # FreqAI обязательные параметры
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Для FreqAI
    plot_config = {
        "main_plot": {
            "tema": {},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Predictions": {
                "&-s_close": {"color": "blue"},
                "do_predict": {"color": "green"},
            },
        },
    }

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Создание признаков для ML модели
        Эта функция вызывается для каждого периода
        """

        # Технические индикаторы
        dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"mfi_{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Moving averages
        dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"sma_{period}"] = ta.SMA(dataframe, timeperiod=period)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe[f"macd_{period}"] = macd["macd"]
        dataframe[f"macdsignal_{period}"] = macd["macdsignal"]
        dataframe[f"macdhist_{period}"] = macd["macdhist"]

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=period, stds=2)
        dataframe[f"bb_lowerband_{period}"] = bollinger["lower"]
        dataframe[f"bb_middleband_{period}"] = bollinger["mid"]
        dataframe[f"bb_upperband_{period}"] = bollinger["upper"]
        dataframe[f"bb_width_{period}"] = (bollinger["upper"] - bollinger["lower"]) / bollinger[
            "mid"
        ]

        # Volatility
        dataframe[f"atr_{period}"] = ta.ATR(dataframe, timeperiod=period)
        dataframe[f"natr_{period}"] = ta.NATR(dataframe, timeperiod=period)

        # Volume indicators
        dataframe[f"volume_mean_{period}"] = dataframe["volume"].rolling(period).mean()
        dataframe[f"volume_std_{period}"] = dataframe["volume"].rolling(period).std()

        # Price momentum
        dataframe[f"roc_{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"mom_{period}"] = ta.MOM(dataframe, timeperiod=period)

        # Trend indicators
        dataframe[f"cci_{period}"] = ta.CCI(dataframe, timeperiod=period)
        dataframe[f"willr_{period}"] = ta.WILLR(dataframe, timeperiod=period)

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Базовые признаки (не зависящие от периода)
        """

        # Свечные паттерны
        dataframe["candle_body"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["candle_upper_shadow"] = dataframe["high"] - dataframe[["close", "open"]].max(
            axis=1
        )
        dataframe["candle_lower_shadow"] = (
            dataframe[["close", "open"]].min(axis=1) - dataframe["low"]
        )

        # Price position relative to high/low
        dataframe["price_position"] = (dataframe["close"] - dataframe["low"]) / (
            dataframe["high"] - dataframe["low"]
        )

        # Gaps
        dataframe["gap"] = dataframe["open"] - dataframe["close"].shift(1)
        dataframe["gap_pct"] = (dataframe["gap"] / dataframe["close"].shift(1)) * 100

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Стандартные признаки - вызываются с предопределенными периодами
        """

        # Периоды для расчета индикаторов
        periods = [10, 20, 50]

        for period in periods:
            dataframe = self.feature_engineering_expand_all(dataframe, period, metadata, **kwargs)

        dataframe = self.feature_engineering_expand_basic(dataframe, metadata, **kwargs)

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Определение целевых переменных для обучения ML модели

        Предсказываем:
        1. Направление движения цены через 5 свечей
        2. Процентное изменение цены
        """

        # Целевая переменная 1: направление цены через 5 свечей
        # 1 = рост, 0 = падение
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-5)  # Цена через 5 свечей
            .rolling(1)
            .mean()
        )

        # Дополнительная целевая: процент изменения
        dataframe["&-s_close_pct"] = (
            (dataframe["close"].shift(-5) - dataframe["close"]) / dataframe["close"] * 100
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавление индикаторов в dataframe
        FreqAI автоматически добавит свои предсказания
        """

        # Базовые индикаторы для визуализации и дополнительной логики
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # Volume
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()

        # FreqAI добавит колонку 'do_predict' и предсказания
        # Предсказания будут в колонке с префиксом '&-s_close' без префикса

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Сигналы на вход в позицию

        Условия:
        1. ML модель предсказывает рост цены с высокой вероятностью
        2. RSI не перекуплен
        3. Объем выше среднего
        4. EMA 20 выше EMA 50 (восходящий тренд)
        """

        conditions = []

        # ML предсказание положительное
        conditions.append(dataframe["&-s_close"] > dataframe["close"])

        # Уверенность модели выше порога
        conditions.append(
            (dataframe["&-s_close"] - dataframe["close"]) / dataframe["close"]
            > self.prediction_threshold.value / 100
        )

        # RSI условия
        conditions.append(dataframe["rsi"] < self.buy_rsi_threshold.value)
        conditions.append(dataframe["rsi"] > 20)  # Не перепродано

        # Trend conditions
        conditions.append(dataframe["ema_20"] > dataframe["ema_50"])

        # Volume
        conditions.append(dataframe["volume"] > dataframe["volume_mean"])

        # FreqAI готов делать предсказания
        conditions.append(dataframe["do_predict"] == 1)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Сигналы на выход из позиции

        Условия:
        1. ML модель предсказывает падение
        2. RSI перекуплен
        3. Цена ниже предсказания модели
        """

        conditions = []

        # ML предсказание отрицательное
        conditions.append(dataframe["&-s_close"] < dataframe["close"])

        # RSI перекуплен
        conditions.append(dataframe["rsi"] > self.sell_rsi_threshold.value)

        # Или EMA crossover вниз
        conditions.append(qtpylib.crossed_below(dataframe["ema_20"], dataframe["ema_50"]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),  # OR condition
                "exit_long",
            ] = 1

        return dataframe
