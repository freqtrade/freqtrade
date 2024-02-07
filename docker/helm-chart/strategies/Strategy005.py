# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, IntParameter
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa

class Strategy005(IStrategy):
    """
    Strategy 005
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy005
    """
    INTERFACE_VERSION = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {'1440': 0.01, '80': 0.02, '40': 0.03, '20': 0.04, '0': 0.05}
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.1
    # Optimal timeframe for the strategy
    timeframe = '5m'
    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    # run "populate_indicators" only for new candle
    process_only_new_candles = True
    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    # Optional order type mapping
    order_types = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
    entry_volumeAVG = IntParameter(low=50, high=300, default=70, space='entry', optimize=True)
    entry_rsi = IntParameter(low=1, high=100, default=30, space='entry', optimize=True)
    entry_fastd = IntParameter(low=1, high=100, default=30, space='entry', optimize=True)
    entry_fishRsiNorma = IntParameter(low=1, high=100, default=30, space='entry', optimize=True)
    exit_rsi = IntParameter(low=1, high=100, default=70, space='exit', optimize=True)
    exit_minusDI = IntParameter(low=1, high=100, default=50, space='exit', optimize=True)
    exit_fishRsiNorma = IntParameter(low=1, high=100, default=50, space='exit', optimize=True)
    exit_trigger = CategoricalParameter(['rsi-macd-minusdi', 'sar-fisherRsi'], default=30, space='exit', optimize=True)
    # Buy hyperspace params:
    entry_params = {'entry_fastd': 1, 'entry_fishRsiNorma': 5, 'entry_rsi': 26, 'entry_volumeAVG': 150}
    # Sell hyperspace params:
    exit_params = {'exit_fishRsiNorma': 30, 'exit_minusDI': 4, 'exit_rsi': 74, 'exit_trigger': 'rsi-macd-minusdi'}

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        # Minus Directional Indicator / Movement
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        # Overlap Studies
        # ------------------------------------
        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)
        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with entry column
        """
        # Prod
        dataframe.loc[(dataframe['close'] > 2e-06) & (dataframe['volume'] > dataframe['volume'].rolling(self.entry_volumeAVG.value).mean() * 4) & (dataframe['close'] < dataframe['sma']) & (dataframe['fastd'] > dataframe['fastk']) & (dataframe['rsi'] > self.entry_rsi.value) & (dataframe['fastd'] > self.entry_fastd.value) & (dataframe['fisher_rsi_norma'] < self.entry_fishRsiNorma.value), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with entry column
        """
        conditions = []
        if self.exit_trigger.value == 'rsi-macd-minusdi':
            conditions.append(qtpylib.crossed_above(dataframe['rsi'], self.exit_rsi.value))
            conditions.append(dataframe['macd'] < 0)
            conditions.append(dataframe['minus_di'] > self.exit_minusDI.value)
        if self.exit_trigger.value == 'sar-fisherRsi':
            conditions.append(dataframe['sar'] > dataframe['close'])
            conditions.append(dataframe['fisher_rsi'] > self.exit_fishRsiNorma.value)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1
        return dataframe