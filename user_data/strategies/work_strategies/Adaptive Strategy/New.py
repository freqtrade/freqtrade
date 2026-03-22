# 1. Использование информативных таймфреймов
@informative("1h", "BTC/USDT")
def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Анализ более высокого таймфрейма для определения общего тренда
    dataframe["trend"] = ta.TREND(dataframe, 20)
    return dataframe


# 2. Создание индикаторов состояния рынка в стратегии
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Определение тренда
    dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
    dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
    dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

    # Определение волатильности
    dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
    dataframe["bbands_upper"], dataframe["bbands_middle"], dataframe["bbands_lower"] = ta.BBANDS(
        dataframe, timeperiod=20
    )

    # Определение силы тренда
    dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

    # Определение состояния рынка
    dataframe["market_state"] = "unknown"

    # Растущий рынок
    dataframe.loc[
        (dataframe["ema_20"] > dataframe["ema_50"])
        & (dataframe["ema_50"] > dataframe["ema_200"])
        & (dataframe["adx"] > 25),
        "market_state",
    ] = "uptrend"

    # Падающий рынок
    dataframe.loc[
        (dataframe["ema_20"] < dataframe["ema_50"])
        & (dataframe["ema_50"] < dataframe["ema_200"])
        & (dataframe["adx"] > 25),
        "market_state",
    ] = "downtrend"

    # Боковой рынок
    dataframe.loc[
        (abs(dataframe["ema_20"] - dataframe["ema_50"]) / dataframe["ema_50"] < 0.02)
        & (dataframe["adx"] < 25),
        "market_state",
    ] = "sideways"

    return dataframe


# 3. Адаптация стратегии под текущее состояние рынка
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe["enter_long"] = 0
    dataframe["enter_short"] = 0

    # Растущий рынок - используем трендследящую стратегию
    uptrend_mask = dataframe["market_state"] == "uptrend"
    dataframe.loc[uptrend_mask & (dataframe["close"] > dataframe["ema_20"]), "enter_long"] = 1

    # Падающий рынок - используем стратегию на понижение
    downtrend_mask = dataframe["market_state"] == "downtrend"
    dataframe.loc[downtrend_mask & (dataframe["close"] < dataframe["ema_20"]), "enter_short"] = 1

    # Боковой рынок - используем осцилляторную стратегию
    sideways_mask = dataframe["market_state"] == "sideways"
    dataframe.loc[
        sideways_mask & (dataframe["close"] < dataframe["bbands_lower"]), "enter_long"
    ] = 1
    dataframe.loc[
        sideways_mask & (dataframe["close"] > dataframe["bbands_upper"]), "enter_short"
    ] = 1

    return dataframe


# 4. Настройка параметров в зависимости от состояния рынка
def custom_stoploss(
    self,
    pair: str,
    trade: Trade,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    **kwargs,
) -> float:
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last_candle = dataframe.iloc[-1].squeeze()

    # Адаптируем стоп-лосс в зависимости от состояния рынка
    if last_candle["market_state"] == "uptrend":
        return -0.05  # Более широкий стоп-лосс для растущего рынка
    elif last_candle["market_state"] == "downtrend":
        return -0.03  # Более узкий стоп-лосс для падающего рынка
    else:  # sideways
        return -0.02  # Самый узкий стоп-лосс для бокового рынка


# 5. Мониторинг и логирование состояния рынка
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
