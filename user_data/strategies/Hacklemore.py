import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI
from statistics import mean
from wao.wao_strategy import WAOStrategy

"""
TODO: 
    - Better buy signal.
    - Potentially leverage an external data source?
"""


class Hacklemore(WAOStrategy):
    brain = "Freq_Hacklemore"

    def __init__(self, config: dict):
        super().__init__(config, self.brain, 8, 0.15)
    timeframe = '5m'
    inf_timeframe = '1h'

    # ROI table:
    minimal_roi = {
        "0": 0.15,
        "5": 0.025,
        "10": 0.015,
        "30": 0.005
    }

    # Stoploss:
    stoploss = -0.085

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    startup_candle_count: int = 24

    custom_trade_info = {}

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Misc. calculations regarding existing open positions (reset on every loop iteration)
        self.custom_trade_info[metadata['pair']] = trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = False

        if self.config['runmode'].value in ('live', 'dry_run'):

            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
            other_trades = Trade.get_trades([Trade.pair != metadata['pair'], Trade.is_open.is_(True), ]).all()

            if active_trade:
                current_rate = self.get_current_price(metadata['pair'])
                current_price_low = self.get_current_price_low(metadata['pair'])
                active_trade[0].adjust_min_max_rates(current_rate, current_price_low)
                trade_data['active_trade'] = True
                trade_data['current_profit'] = active_trade[0].calc_profit_ratio(current_rate)
                trade_data['peak_profit'] = active_trade[0].calc_profit_ratio(active_trade[0].max_rate)

            if other_trades:
                trade_data['other_trades'] = True
                total_other_profit = tuple(
                    trade.calc_profit_ratio(self.get_current_price(trade.pair)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(total_other_profit)

        self.custom_trade_info[metadata['pair']] = trade_data

        # Set up other indicators
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['rmi-slow'] = RMI(dataframe, length=20, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=9, mom=3)
        dataframe['sar'] = ta.SAR(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Trend calculations
        dataframe['max'] = dataframe['high'].rolling(12).max()
        dataframe['min'] = dataframe['low'].rolling(12).min()
        dataframe['upper'] = np.where(dataframe['max'] > dataframe['max'].shift(), 1, 0)
        dataframe['lower'] = np.where(dataframe['min'] < dataframe['min'].shift(), 1, 0)
        dataframe['up_trend'] = np.where(dataframe['upper'].rolling(3, min_periods=1).sum() != 0, 1, 0)
        dataframe['dn_trend'] = np.where(dataframe['lower'].rolling(3, min_periods=1).sum() != 0, 1, 0)

        # Informative Pair Indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        informative['ema3'] = ta.EMA(informative, timeperiod=3)
        informative['ema12'] = ta.EMA(informative, timeperiod=12)

        inf_macd = ta.MACD(informative)
        informative['macd'] = inf_macd['macd']
        informative['macdsignal'] = inf_macd['macdsignal']
        informative['macdhist'] = inf_macd['macdhist']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # Persist a buy signal for existing trades to make use of ignore_roi_if_buy_signal = True
        # when this buy signal is not present a sell can happen according to ROI table
        if trade_data['active_trade']:
            if (trade_data['peak_profit'] > 0):
                conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * 0.8))
            conditions.append(dataframe['rmi-slow'] >= 60)

        # Normal buy triggers that apply to new trades we want to enter
        else:
            conditions.append(
                # informative timeframe conditions
                (dataframe[f"ema3_{self.inf_timeframe}"] > dataframe[f"ema12_{self.inf_timeframe}"]) &
                (dataframe[f"macd_{self.inf_timeframe}"] > dataframe[f"macdsignal_{self.inf_timeframe}"]) &

                # default timeframe conditions
                (dataframe['up_trend'] == 1) &
                (dataframe['rmi-slow'] >= 55) &
                (dataframe['macdhist'] > 0) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 30)) &

                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['rmi-slow'] >= dataframe['rmi-slow'].rolling(3).mean()) &
                (dataframe['close'] > dataframe['close'].shift()) &
                (dataframe['close'].shift() > dataframe['close'].shift(2)) &
                (dataframe['sar'] < dataframe['close']) &
                (dataframe['sar'].shift() < dataframe['close'].shift())
            )

        # applies to both new buys and persisting buy signal
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # if we are in an active trade for this pair
        if trade_data['active_trade']:
            # if we are at a loss, consider what the trend looks and preempt the stoploss
            conditions.append(
                (trade_data['current_profit'] < 0) &
                (trade_data['current_profit'] > self.stoploss) &
                (dataframe['dn_trend'] == 1) &
                (dataframe['rmi-fast'] < 50) &
                (dataframe['volume'].gt(0))
            )

            # if there are other open trades in addition to this one, consider the average profit 
            # across them all (not including this one), don't sell if entire market is down big and wait for recovery
            if trade_data['other_trades']:
                conditions.append(trade_data['avg_other_profit'] >= -0.005)

        # the bot comes through this loop even when there isn't an open trade to sell
        # so we pass an impossible condiiton here because we don't want a sell signal 
        # clogging up the charts and not having one leads the bot to crash
        else:
            conditions.append(dataframe['volume'].lt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

    """
    Custom methods
    """

    def get_current_price(self, pair: str) -> float:
        """
        # Using ticker seems significantly faster than orderbook.
        side = "asks"
        if (self.config['ask_strategy']['price_side'] == "bid"):
            side = "bids"
        
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[side][0][0]
        """

        ticker = self.dp.ticker(pair)
        current_price = ticker['last']

        return current_price

    def get_current_price_low(self, pair: str) -> float:
        """
        # Using ticker seems significantly faster than orderbook.
        side = "asks"
        if (self.config['ask_strategy']['price_side'] == "bid"):
            side = "bids"

        ob = self.dp.orderbook(pair, 1)
        current_price = ob[side][0][0]
        """

        ticker = self.dp.ticker(pair)
        current_price = ticker['last']

        return current_price

    """
    Price protection on trade entry and timeouts, built-in Freqtrade functionality
    https://www.freqtrade.io/en/latest/strategy-advanced/
    """

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > order['price'] * 1.01:
            return True
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel sell order if price is more than 1% below the order.
        if current_price < order['price'] * 0.99:
            return True
        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
                            **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > rate * 1.01:
            return False
        return True


class Hacklemore_Slow(Hacklemore):
    timeframe = '1h'
    inf_timeframe = '4h'

    # ROI table:
    minimal_roi = {
        "0": 0.15,
        "10": 0.10,
        "20": 0.05,
        "30": 0.025,
        "60": 0.015,
        "120": 0.005
    }

    # Stoploss:
    stoploss = -0.085
