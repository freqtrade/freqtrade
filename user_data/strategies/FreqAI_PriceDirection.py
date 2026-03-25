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

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Принудительно задаем freqaimodel для обхода бага веб-сервера
        self.config["freqaimodel"] = "LightGBMRegressor"

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
    timeframe = "15m"

    # Параметры для оптимизации
    buy_rsi_threshold = IntParameter(20, 40, default=30, space="buy")
    sell_rsi_threshold = IntParameter(60, 80, default=70, space="sell")

    # Параметры ML
    prediction_threshold = DecimalParameter(0.5, 0.9, default=0.51, space="buy")

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
            "ema_20": {},
            "ema_50": {},
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
        dataframe[f"%-rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi_{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

        # Moving averages
        dataframe[f"%-ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-sma_{period}"] = ta.SMA(dataframe, timeperiod=period)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe[f"%-macd_{period}"] = macd["macd"]
        dataframe[f"%-macdsignal_{period}"] = macd["macdsignal"]
        dataframe[f"%-macdhist_{period}"] = macd["macdhist"]

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=period, stds=2)
        dataframe[f"%-bb_lowerband_{period}"] = bollinger["lower"]
        dataframe[f"%-bb_middleband_{period}"] = bollinger["mid"]
        dataframe[f"%-bb_upperband_{period}"] = bollinger["upper"]
        bb_range = bollinger["upper"] - bollinger["lower"]
        dataframe[f"%-bb_width_{period}"] = bb_range / bollinger["mid"]

        # Volatility
        dataframe[f"%-atr_{period}"] = ta.ATR(dataframe, timeperiod=period)
        dataframe[f"%-natr_{period}"] = ta.NATR(dataframe, timeperiod=period)

        # Volume indicators
        dataframe[f"%-volume_mean_{period}"] = dataframe["volume"].rolling(period).mean()
        dataframe[f"%-volume_std_{period}"] = dataframe["volume"].rolling(period).std()

        # Price momentum
        dataframe[f"%-roc_{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"%-mom_{period}"] = ta.MOM(dataframe, timeperiod=period)

        # Trend indicators
        dataframe[f"%-cci_{period}"] = ta.CCI(dataframe, timeperiod=period)
        dataframe[f"%-willr_{period}"] = ta.WILLR(dataframe, timeperiod=period)

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Базовые признаки (не зависящие от периода)
        """

        # Свечные паттерны
        dataframe["%-candle_body"] = abs(dataframe["close"] - dataframe["open"])
        high_low = dataframe[["close", "open"]]
        dataframe["%-candle_upper_shadow"] = dataframe["high"] - high_low.max(axis=1)
        dataframe["%-candle_lower_shadow"] = high_low.min(axis=1) - dataframe["low"]

        # Price position relative to high/low
        hl_range = dataframe["high"] - dataframe["low"]
        dataframe["%-price_position"] = (dataframe["close"] - dataframe["low"]) / hl_range

        # Gaps
        dataframe["%-gap"] = dataframe["open"] - dataframe["close"].shift(1)
        dataframe["%-gap_pct"] = (dataframe["%-gap"] / dataframe["close"].shift(1)) * 100

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
        # (УБРАНО: LightGBM поддерживает только 1 цель за раз)
        # dataframe["&-s_close_pct"] = (
        #     (dataframe["close"].shift(-5) - dataframe["close"]) / dataframe["close"] * 100
        # )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавление индикаторов в dataframe
        """
        # 1. Ваши базовые индикаторы (оставляем как есть)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()

        # 2. ГЛАВНАЯ СТРОКА: ЗАПУСК FREQAI
        # Без этой строки методы feature_engineering никогда не будут вызваны!
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Проверяем, что модель уже выдала предсказания
        if "&-s_close" not in dataframe.columns or "do_predict" not in dataframe.columns:
            return dataframe

        conditions = []

        # FreqAI готов делать предсказания
        conditions.append(dataframe["do_predict"] == 1)

        # 1. Порог уверенности модели (на основе вашего prediction_threshold)
        # В регрессии проверяем, что предсказанная цена выше текущей хотя бы на threshold (%)
        conditions.append(
            dataframe["&-s_close"]
            > dataframe["close"] * (1 + self.prediction_threshold.value / 100)
        )

        # 2. RSI как дополнительный фильтр
        conditions.append(dataframe["rsi"] < self.buy_rsi_threshold.value)
        conditions.append(
            dataframe["rsi"] > 20
        )  # Избегаем ложных входов на экстремальной перепроданности

        # 3. Дополнительные фильтры из v0 для надежности входа
        conditions.append(dataframe["ema_20"] > dataframe["ema_50"])  # Восходящий микро-тренд
        conditions.append(dataframe["volume"] > dataframe["volume_mean"])  # Объем выше среднего

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Сигналы на выход из позиции
        """
        if "&-s_close" not in dataframe.columns or "do_predict" not in dataframe.columns:
            return dataframe

        # Условия для выхода объединяются через логическое ИЛИ (любой из сигналов = выход),
        # но мы должны проверять это только когда модель активна (do_predict == 1)

        exit_conditions = []

        # ML предсказание отрицательное
        exit_conditions.append(dataframe["&-s_close"] < dataframe["close"])

        # RSI перекуплен
        exit_conditions.append(dataframe["rsi"] > self.sell_rsi_threshold.value)

        # Пересечение скользящих средних вниз
        exit_conditions.append(qtpylib.crossed_below(dataframe["ema_20"], dataframe["ema_50"]))

        if exit_conditions:
            dataframe.loc[
                (dataframe["do_predict"] == 1)
                & reduce(
                    lambda x, y: x | y, exit_conditions
                ),  # Сработал хотя бы один триггер на выход
                "exit_long",
            ] = 1

        return dataframe
