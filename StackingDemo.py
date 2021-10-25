# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.persistence import Trade
from datetime import datetime,timezone,timedelta

"""
 Warning:
This is still work in progress, so there is no warranty that everything works as intended, 
it is possible that this strategy results in huge losses or doesn't even work at all.
Make sure to only run this in dry_mode so you don't lose any money.

"""

class StackingDemo(IStrategy):
    """
    This is the default strategy template with added functions for trade stacking / buying the same positions multiple times.
    It should function like this:
        Find good buys using indicators. 
        When a new buy occurs the strategy will enable rebuys of the pair like this:
            self.custom_info[metadata["pair"]]["rebuy"] = 1
        Then, if the price should drop after the last buy within the timerange of rebuy_time_limit_hours,
        the same pair will be purchased again. This is intended to help with reducing possible losses.
        If the price only goes up after the first buy, the strategy won't buy this pair again, and after the time limit is over,
        look for other pairs to buy.
        For selling there is this flag:
            self.custom_info[metadata["pair"]]["resell"] = 1
        which should simply sell all trades of this pair until none are left.

    You can set how many pairs you want to trade and how many trades you want to allow for a pair, 
    but you must make sure to set max_open_trades to the produce of max_open_pairs and max_open_trades in your configuration file.
    Also allow_position_stacking has to be set to true in the configuration file.

    For backtesting make sure to provide --enable-position-stacking as an argument in the command line.
    Backtesting will be slow.
    Hyperopt was not tested.
    
    # run the bot:
    freqtrade trade -c StackingConfig.json -s StackingDemo --db-url sqlite:///tradesv3_StackingDemo_dry-run.sqlite --dry-run
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # how many pairs to trade / trades per pair if allow_position_stacking is enabled
    max_open_pairs, max_trades_per_pair = 4, 3
    # make sure to have this value in your config file
    max_open_trades = max_open_pairs * max_trades_per_pair

    # debugging
    print_trades = True

    # specify for how long to want to allow rebuys of this pair
    rebuy_time_limit_hours = 2

    # store additional information needed for this strategy:
    custom_info = {}
    custom_num_open_pairs = {}

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
#        "60": 0.01,
#        "30": 0.02,
        "0": 0.001
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
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

        # STACKING STUFF

        # confirm config
        self.max_trades_per_pair = self.config['max_open_trades'] / self.max_open_pairs
        if not self.config["allow_position_stacking"]:
            self.max_trades_per_pair = 1

        # store number of open pairs
        self.custom_num_open_pairs = {"num_open_pairs": 0}

        # Store custom information for this pair:
        if not metadata["pair"] in self.custom_info:
            self.custom_info[metadata["pair"]] = {}

        if not "rebuy" in self.custom_info[metadata["pair"]]:
            # number of trades for this pair
            self.custom_info[metadata["pair"]]["num_trades"] = 0
            # use rebuy/resell as buy-/sell- indicators
            self.custom_info[metadata["pair"]]["rebuy"] = 0
            self.custom_info[metadata["pair"]]["resell"] = 0
            # store latest open_date for this pair
            self.custom_info[metadata["pair"]]["last_open_date"] = datetime.now(timezone.utc) - timedelta(days=100)
            # stare the value of the latest open price for this pair
            self.custom_info[metadata["pair"]]["latest_open_rate"] = 0

        # INDICATORS

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upperband"] = keltner["upper"]
        # dataframe["kc_lowerband"] = keltner["lower"]
        # dataframe["kc_middleband"] = keltner["mid"]
        # dataframe["kc_percent"] = (
        #     (dataframe["close"] - dataframe["kc_lowerband"]) /
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        # )
        # dataframe["kc_width"] = (
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        # )

        # # Ultimate Oscillator
        # dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch['slowd']
        # dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        # stoch_rsi = ta.STOCHRSI(dataframe)
        # dataframe['fastd_rsi'] = stoch_rsi['fastd']
        # dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

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

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) / dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        # dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
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
        dataframe.loc[
            (
                (
                    (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                    (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
                    (dataframe['tema'] > dataframe['tema'].shift(1)) |  # Guard: tema is raising
                    # use either buy signal or rebuy flag to trigger a buy
                    (self.custom_info[metadata["pair"]]["rebuy"] == 1)
                ) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
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
                (
                    (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
                    (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                    (dataframe['tema'] < dataframe['tema'].shift(1)) |  # Guard: tema is falling
                    # use either sell signal or resell flag to trigger a sell
                    (self.custom_info[metadata["pair"]]["resell"] == 1)
                ) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

    # use_custom_sell = True

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> 'Optional[Union[str, bool]]':
        """
        Custom sell signal logic indicating that specified position should be sold. Returning a
        string or True from this method is equal to setting sell signal on a candle at specified
        time. This method is not called when sell signal is set.

        This method should be overridden to create sell signals that depend on trade parameters. For
        example you could implement a sell relative to the candle when the trade was opened,
        or a custom 1:2 risk-reward ROI.

        Custom sell reason max length is 64. Exceeding characters will be removed.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: To execute sell, return a string with custom sell reason or True. Otherwise return
        None or False.
        """
        # if self.custom_info[pair]["resell"] == 1:
        #    return 'resell'
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: 'datetime', **kwargs) -> bool:
        return_statement = True

        if self.config['allow_position_stacking']:
            return_statement = self.check_open_trades(pair, rate, current_time)

        # debugging
        if return_statement and self.print_trades:
            # use str.join() for speed
            out = (current_time.strftime("%c"), " Bought: ", pair, ", rate: ", str(rate), ", rebuy: ", str(self.custom_info[pair]["rebuy"]), ", trades: ", str(self.custom_info[pair]["num_trades"]))
            print("".join(out))

        return return_statement

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: 'datetime', **kwargs) -> bool:

        if self.config["allow_position_stacking"]:

            # unlock open pairs limit after every sell
            self.unlock_reason('Open pairs limit')

            # unlock open pairs limit after last item is sold
            if self.custom_info[pair]["num_trades"] == 1:
                # decrement open_pairs_count by 1 if last item is sold
                self.custom_num_open_pairs["num_open_pairs"]-=1
                self.custom_info[pair]["resell"] = 0
                # reset rate
                self.custom_info[pair]["latest_open_rate"] = 0.0
                self.unlock_reason('Trades per pair limit')

            # change dataframe to produce sell signal after a sell
            if self.custom_info[pair]["num_trades"] >= self.max_trades_per_pair:
                self.custom_info[pair]["resell"] = 1

            # decrement number of trades by 1:
            self.custom_info[pair]["num_trades"]-=1

        # debugging stuff
        if self.print_trades:
            # use str.join() for speed
            out = (current_time.strftime("%c"), " Sold: ", pair, ", rate: ", str(rate),", profit: ", str(trade.calc_profit_ratio(rate)), ", resell: ", str(self.custom_info[pair]["resell"]), ", trades: ", str(self.custom_info[pair]["num_trades"]))
            print("".join(out))

        return True

    def check_open_trades(self, pair: str, rate: float, current_time: datetime):

        # retrieve information about current open pairs
        tr_info = self.get_trade_information(pair)

        # update number of open trades for the pair
        self.custom_info[pair]["num_trades"] = tr_info[1]
        self.custom_num_open_pairs["num_open_pairs"] = len(tr_info[0])
        # update value of the last open price
        self.custom_info[pair]["latest_open_rate"] = tr_info[2]

        # don't buy if we have enough trades for this pair
        if self.custom_info[pair]["num_trades"] >= self.max_trades_per_pair:
            # lock if we already have enough pairs open, will be unlocked after last item of a pair is sold
            self.lock_pair(pair, until=datetime.now(timezone.utc) + timedelta(days=100), reason='Trades per pair limit')
            self.custom_info[pair]["rebuy"] = 0
            return False

        # don't buy if we have enough pairs
        if self.custom_num_open_pairs["num_open_pairs"] >= self.max_open_pairs:
            if not pair in tr_info[0]:
                # lock if this pair is not in our list, will be unlocked after the next sell
                self.lock_pair(pair, until=datetime.now(timezone.utc) + timedelta(days=100), reason='Open pairs limit')
                self.custom_info[pair]["rebuy"] = 0
                return False

        # don't buy at a higher price, try until time limit is exceeded; skips if it's the first trade'
        if rate > self.custom_info[pair]["latest_open_rate"] and self.custom_info[pair]["latest_open_rate"] != 0.0:
            # how long do we want to try buying cheaper before we look for other pairs?
            if (current_time - self.custom_info[pair]['last_open_date']).seconds/3600 > self.rebuy_time_limit_hours:
                self.custom_info[pair]["rebuy"] = 0
                self.unlock_reason('Open pairs limit')
            return False

        # set rebuy flag if num_trades < limit-1
        if self.custom_info[pair]["num_trades"] < self.max_trades_per_pair-1:
            self.custom_info[pair]["rebuy"] = 1
        else:
            self.custom_info[pair]["rebuy"] = 0

        # update rate
        self.custom_info[pair]["latest_open_rate"] = rate

        #update date open
        self.custom_info[pair]["last_open_date"] = current_time

        # increment trade count by 1
        self.custom_info[pair]["num_trades"]+=1

        return True

    # custom function to help with the strategy
    def get_trade_information(self, pair:str):

        latest_open_rate, trade_count = 0, 0.0
        # store all open pairs
        open_pairs = []

        ### start nested function
        def compare_trade(trade: Trade):
            nonlocal trade_count, latest_open_rate, pair
            if trade.pair == pair:
                # update latest_rate
                latest_open_rate = trade.open_rate
                trade_count+=1
            return trade.pair
        ### end nested function

        # replaced for loop with map for speed
        open_pairs = map(compare_trade, Trade.get_open_trades())
        # remove duplicates
        open_pairs = (list(dict.fromkeys(open_pairs)))

        #print(*open_pairs, sep="\n")

        # put this all together to reduce the amount of loops
        return open_pairs, trade_count, latest_open_rate
