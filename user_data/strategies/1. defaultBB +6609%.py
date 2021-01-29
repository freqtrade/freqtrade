# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class defaultB(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "4320": 0.0018,
        #"300": 0.01,
        #"60": 0.04, # 0.01
        #"30": 0.06, # 0.02
        "0": 0.30 # 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.0521 # -0.05

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True

    # trailing_stop_positive = 0.005
    trailing_stop_positive = 0.0005
    trailing_stop_positive_offset = 0.034

    # Optimal timeframe for the strategy.
    timeframe = '30m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values canconfig222 be overridden in the "ask_strategy" section in the config.
    use_sell_signal = False
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
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
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['adx50'] = ta.ADX(dataframe, timeperiod=50)


        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['cmo'] = ta.CMO(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi25'] = ta.RSI(dataframe, timeperiod=25)
        dataframe['rsi50'] = ta.RSI(dataframe, timeperiod=50)
        dataframe['rsi100'] = ta.RSI(dataframe, timeperiod=100)
        dataframe['rsi200'] = ta.RSI(dataframe, timeperiod=200)
        dataframe['rsi400'] = ta.RSI(dataframe, timeperiod=400)
        dataframe['rsi1000'] = ta.RSI(dataframe, timeperiod=1000)


        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']


        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi20'] = ta.MFI(dataframe, timeperiod=20)
        dataframe['mfi50'] = ta.MFI(dataframe, timeperiod=50)
        dataframe['mfi100'] = ta.MFI(dataframe, timeperiod=100)
        dataframe['mfi200'] = ta.MFI(dataframe, timeperiod=200)
        dataframe['mfi400'] = ta.MFI(dataframe, timeperiod=400)
        dataframe['mfi1000'] = ta.MFI(dataframe, timeperiod=1000)

        # # ROC
        dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema400'] = ta.EMA(dataframe, timeperiod=100)


        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['tema3'] = ta.TEMA(dataframe, timeperiod=3)
        dataframe['tema5'] = ta.TEMA(dataframe, timeperiod=5)
        dataframe['tema14'] = ta.TEMA(dataframe, timeperiod=14)
        dataframe['tema21'] = ta.TEMA(dataframe, timeperiod=21)
        dataframe['tema50'] = ta.TEMA(dataframe, timeperiod=50)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # # Chart type
        # # ------------------------------------
        # helps with pattern recognition
        dataframe['average'] = (dataframe['close'] + dataframe['open'] + dataframe['high'] + dataframe['low']) / 4
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=200, price='close')
        dataframe['sma_medium'] = ta.SMA(dataframe, timeperiod=100, price='close')
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=50, price='close')

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[ # Breakout: TEMA crosses over longer TEMA
            (
                (qtpylib.crossed_above(dataframe['tema5'], dataframe['tema14'])) &
                (dataframe['adx'] > 25)
            ),
            'buy'] = 1
        dataframe.loc[ # Breakout: RSI plus momentum
            (
                (qtpylib.crossed_above(dataframe['rsi'], 60)) &
                (dataframe['adx'] > 30)
            ),
            'buy'] = 1
        dataframe.loc[ # Breakout: crossing middle of Bollinger Bands
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['bb_middleband'])) &
                (dataframe['adx'] > 30) &
                (dataframe['adx'] > dataframe['adx'].shift(1)) &
                (dataframe['adx'].shift(1) > dataframe['adx'].shift(2)) &
                (dataframe['adx'].shift(2) > dataframe['adx'].shift(3)) &
                (dataframe['close'] > dataframe['ema21']) &
                (dataframe['close'] > dataframe['ema50']) &
                (dataframe['roc'] > 0) &
                (dataframe['ema50'] > 0) &
                (dataframe['ema21'] > dataframe['ema50'])
            ),
            'buy'] = 1
        dataframe.loc[ # Breakout: Increasing momentum and positive price action
            (
                (qtpylib.crossed_above(dataframe['adx'], 25)) &
                (dataframe['ema21'] > dataframe['ema50'])
            ),
            'buy'] = 1
        dataframe.loc[ # BTFD: CCI bounce
                # Doesn't like BTC Dominance tracking sideways
                # Enable by uncommenting the indicators
            (
                #(qtpylib.crossed_above(dataframe['cci'], -100))
            ),
            'buy'] = 1
        dataframe.loc[ # BTFD: Price Crosses into the Bollinger Bands from below
                # Doesn't like BTC Dominance tracking sideways
                # Enable by uncommenting the indicators
            (
                #(qtpylib.crossed_above(dataframe['close'], dataframe['bb_lowerband']))
            ),
            'buy'] = 1
        dataframe.loc[ # BTFD: V-shaped bottom.
                # Doesn't like BTC Dominance tracking sideways
                # Enable by uncommenting the indicators
            (
                #(dataframe['average'].shift(5) > dataframe['average'].shift(4)) &
                #(dataframe['average'].shift(4) > dataframe['average'].shift(3)) &
                #(dataframe['average'].shift(3) > dataframe['average'].shift(2)) &
                #(dataframe['average'].shift(2) > dataframe['average'].shift(1)) &
                #(dataframe['average'].shift(1) < dataframe['average'].shift(0)) &
                #(dataframe['low'].shift(1) < dataframe['bb_middleband']) &
                #(dataframe['cci'].shift(1) < -100)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
		(qtpylib.crossed_above(dataframe['mfi'], 95))  # Signal: RSI crosses above 70
            ),
            'sell'] = 1
        dataframe.loc[
            (
		(dataframe['mfi'] > 90) &  # Signal: RSI crosses above 70
		(dataframe['tema'] < dataframe['tema'].shift(3)) & # Guard: negative trend
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband'])) &
                (dataframe['mfi'] > 60)  # High MFI
            ),
            'sell'] = 1
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['adx'], 28)) &
                (dataframe['rsi'] > 60)  #  High RSI
            ),
            'sell'] = 1
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['adx'], 28)) &
                (dataframe['mfi'] > 60)  # High MFI
            ),
            'sell'] = 1
        return dataframe

