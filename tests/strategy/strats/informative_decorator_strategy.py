# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from pandas import DataFrame

from freqtrade.strategy import informative, merge_informative_pair
from freqtrade.strategy.interface import IStrategy


class InformativeDecoratorTest(IStrategy):
    """
    Strategy used by tests freqtrade bot.
    Please do not modify this strategy, it's  intended for internal use only.
    Please look at the SampleStrategy in the user_data/strategy directory
    or strategy repository https://github.com/freqtrade/freqtrade-strategies
    for samples and inspiration.
    """
    INTERFACE_VERSION = 2
    stoploss = -0.10
    timeframe = '5m'
    startup_candle_count: int = 20

    def informative_pairs(self):
        return [('BTC/USDT', '5m')]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sell'] = 0
        return dataframe

    # Decorator stacking test.
    @informative('30m')
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = 14
        return dataframe

    # Simple informative test.
    @informative('1h', 'BTC/{stake}')
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = 14
        return dataframe

    # Quote currency different from stake currency test.
    @informative('1h', 'ETH/BTC')
    def populate_indicators_eth_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = 14
        return dataframe

    # Formatting test.
    @informative('30m', 'BTC/{stake}', '{column}_{BASE}_{QUOTE}_{base}_{quote}_{asset}_{timeframe}')
    def populate_indicators_btc_1h_2(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = 14
        return dataframe

    # Custom formatter test
    @informative('30m', 'ETH/{stake}', fmt=lambda column, **kwargs: column + '_from_callable')
    def populate_indicators_eth_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = 14
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Strategy timeframe indicators for current pair.
        dataframe['rsi'] = 14
        # Informative pairs are available in this method.
        dataframe['rsi_less'] = dataframe['rsi'] < dataframe['rsi_1h']

        # Mixing manual informative pairs with decorators.
        informative = self.dp.get_pair_dataframe('BTC/USDT', '5m')
        informative['rsi'] = 14
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '5m', ffill=True)

        return dataframe
