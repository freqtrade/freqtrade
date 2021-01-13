import talib.abstract as ta
from pandas import DataFrame
import scipy.signal
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy


class TDSequentialStrategy(IStrategy):
    """
    Strategy based on TD Sequential indicator.
    source:
    https://hackernoon.com/how-to-buy-sell-cryptocurrency-with-number-indicator-td-sequential-5af46f0ebce1

    Buy trigger:
        When you see 9 consecutive closes "lower" than the close 4 bars prior.
        An ideal buy is when the low of bars 6 and 7 in the count are exceeded by the low of bars 8 or 9.

    Sell trigger:
        When you see 9 consecutive closes "higher" than the close 4 candles prior.
        An ideal sell is when the the high of bars 6 and 7 in the count are exceeded by the high of bars 8 or 9.

    Created by @bmoulkaf
    """
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy
    minimal_roi = {'0': 5}

    # Optimal stoploss designed for the strategy
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional time in force for orders
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
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
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        dataframe['exceed_high'] = False
        dataframe['exceed_low'] = False

        # count consecutive closes “lower” than the close 4 bars prior.
        dataframe['seq_buy'] = dataframe['close'] < dataframe['close'].shift(4)
        dataframe['seq_buy'] = dataframe['seq_buy'] * (dataframe['seq_buy'].groupby(
            (dataframe['seq_buy'] != dataframe['seq_buy'].shift()).cumsum()).cumcount() + 1)

        # count consecutive closes “higher” than the close 4 bars prior.
        dataframe['seq_sell'] = dataframe['close'] > dataframe['close'].shift(4)
        dataframe['seq_sell'] = dataframe['seq_sell'] * (dataframe['seq_sell'].groupby(
            (dataframe['seq_sell'] != dataframe['seq_sell'].shift()).cumsum()).cumcount() + 1)

        for index, row in dataframe.iterrows():
            # check if the low of bars 6 and 7 in the count are exceeded by the low of bars 8 or 9.
            seq_b = row['seq_buy']
            if seq_b == 8:
                dataframe.loc[index, 'exceed_low'] = (row['low'] < dataframe.loc[index - 2, 'low']) | \
                                    (row['low'] < dataframe.loc[index - 1, 'low'])
            if seq_b > 8:
                dataframe.loc[index, 'exceed_low'] = (row['low'] < dataframe.loc[index - 3 - (seq_b - 9), 'low']) | \
                                    (row['low'] < dataframe.loc[index - 2 - (seq_b - 9), 'low'])
                if seq_b == 9:
                    dataframe.loc[index, 'exceed_low'] = row['exceed_low'] | dataframe.loc[index-1, 'exceed_low']

            # check if the high of bars 6 and 7 in the count are exceeded by the high of bars 8 or 9.
            seq_s = row['seq_sell']
            if seq_s == 8:
                dataframe.loc[index, 'exceed_high'] = (row['high'] > dataframe.loc[index - 2, 'high']) | \
                                    (row['high'] > dataframe.loc[index - 1, 'high'])
            if seq_s > 8:
                dataframe.loc[index, 'exceed_high'] = (row['high'] > dataframe.loc[index - 3 - (seq_s - 9), 'high']) | \
                                    (row['high'] > dataframe.loc[index - 2 - (seq_s - 9), 'high'])
                if seq_s == 9:
                    dataframe.loc[index, 'exceed_high'] = row['exceed_high'] | dataframe.loc[index-1, 'exceed_high']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe["buy"] = 0
        dataframe.loc[((dataframe['exceed_low']) &
                      (dataframe['seq_buy'] > 8))
                      , 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy columnNA / NaN values
        """
        dataframe["sell"] = 0
        dataframe.loc[((dataframe['exceed_high']) |
                       (dataframe['seq_sell'] > 8))
                      , 'sell'] = 1
        return dataframe
