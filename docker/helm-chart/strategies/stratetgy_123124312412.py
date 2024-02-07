import logging
from datetime import datetime
from functools import reduce
from typing import Optional

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical.indicators import ichimoku

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


class HPStrategyTFJPAConfirmV1(IStrategy):
    INTERFACE_VERSION = 2
    support_dict = {}
    resistance_dict = {}
    lowest_prices = {}
    highest_prices = {}
    price_drop_percentage = {}
    pairs_close_to_high = []
    locked = []
    stoploss = -0.99

    is_optimize_cofi = True
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False
    position_adjustment_enable = True
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    timeframe = '1m'
    inf_1h = '1h'
    process_only_new_candles = True
    startup_candle_count = 400
    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    buy_params = {
        "buy_adx": 24,
        "buy_ema_cofi": 0.979,
        "buy_ewo_high": 9.793,
        "buy_fastd": 29,
        "buy_fastk": 30,
        "distance_to_support_treshold": 0.041,
        "rsi_buy": 36,
        "base_nb_candles_buy": 12,  # value loaded from strategy
        "candles_before": 37,  # value loaded from strategy
        "candles_dca_multiplier": 56,  # value loaded from strategy
        "dca_order_divider": 3,  # value loaded from strategy
        "dca_wallet_divider": 5,  # value loaded from strategy
        "ewo_high": 3.001,  # value loaded from strategy
        "ewo_low": -10.289,  # value loaded from strategy
        "lambo2_ema_14_factor": 0.981,  # value loaded from strategy
        "lambo2_rsi_14_limit": 39,  # value loaded from strategy
        "lambo2_rsi_4_limit": 44,  # value loaded from strategy
        "low_offset": 0.987,  # value loaded from strategy
        "max_safety_orders": 9,  # value loaded from strategy
        "open_trade_limit": 5,  # value loaded from strategy
        "pct_drop_treshold": 0.011,  # value loaded from strategy
        "stoch_treshold": 25,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 22,  # value loaded from strategy
        "high_offset": 1.014,  # value loaded from strategy
        "high_offset_2": 1.01,  # value loaded from strategy
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.12,
        "30": 0.04,
        "54": 0.017,
        "135": 0
    }

    is_optimize_dca = False
    is_optimize_sr = True

    stoch_treshold = IntParameter(20, 40, default=buy_params['stoch_treshold'], space='buy', optimize=False)

    distance_to_support_treshold = DecimalParameter(0.01, 0.05, default=buy_params['distance_to_support_treshold'],
                                                    space='buy', optimize=is_optimize_sr)
    pct_drop_treshold = DecimalParameter(0.01, 0.05, default=buy_params['pct_drop_treshold'], space='buy',
                                         optimize=is_optimize_dca)
    candles_before = IntParameter(30, 200, default=buy_params['candles_before'], space='buy',
                                  optimize=is_optimize_dca)
    candles_dca_multiplier = IntParameter(30, 60, default=buy_params['candles_dca_multiplier'], space='buy',
                                          optimize=is_optimize_dca)
    open_trade_limit = IntParameter(1, 10, default=buy_params['open_trade_limit'], space='buy', optimize=False)

    dca_wallet_divider = IntParameter(open_trade_limit.value - 1, 10, default=buy_params['dca_wallet_divider'],
                                      space='buy', optimize=is_optimize_dca)

    dca_order_divider = IntParameter(2, 10, default=buy_params['dca_order_divider'], space='buy',
                                     optimize=is_optimize_dca)

    max_safety_orders = IntParameter(1, 10, default=buy_params['max_safety_orders'], space='buy',
                                     optimize=is_optimize_dca)

    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize=is_optimize_cofi)
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    low_offset = DecimalParameter(0.975, 0.995, default=buy_params['low_offset'], space='buy', optimize=False)

    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=buy_params['lambo2_ema_14_factor'],
                                            space='buy', optimize=False)
    lambo2_rsi_4_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=False)
    lambo2_rsi_14_limit = IntParameter(10, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=False)

    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -7.0, default=buy_params['ewo_low'], space='buy', optimize=False)
    ewo_high = DecimalParameter(3.0, 5, default=buy_params['ewo_high'], space='buy', optimize=False)

    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    max_open_trades = 200
    amend_last_stake_amount = True

    start = 0.02
    increment = 0.02
    maximum = 0.2

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell',
                                        optimize=False)
    high_offset = DecimalParameter(1.000, 1.010, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.000, 1.010, default=sell_params['high_offset_2'], space='sell', optimize=True)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    def version(self) -> str:
        return f"{super().version()} TFJPAConfirm "

    def pivot_points(self, high, low, period=10):
        pivot_high = high.rolling(window=2 * period + 1, center=True).max()
        pivot_low = low.rolling(window=2 * period + 1, center=True).min()
        return high == pivot_high, low == pivot_low

    def calculate_support_resistance(self, df, period=10):
        high_pivot, low_pivot = self.pivot_points(df['high'], df['low'], period)
        df['resistance'] = df['high'][high_pivot]
        df['support'] = df['low'][low_pivot]
        return df

    def calculate_support_resistance_dicts(self, pair: str, df: DataFrame):
        try:
            df = self.calculate_support_resistance(df)
            self.support_dict[pair] = self.calculate_dynamic_clusters(df['support'].dropna().tolist(), 4)
            self.resistance_dict[pair] = self.calculate_dynamic_clusters(df['resistance'].dropna().tolist(), 4)
        except Exception as ex:
            logging.error(str(ex))

    def calculate_dynamic_clusters(self, values, max_clusters):

        def cluster_values(threshold):
            sorted_values = sorted(values)
            clusters = []
            current_cluster = [sorted_values[0]]

            for value in sorted_values[1:]:
                if value - current_cluster[-1] <= threshold:
                    current_cluster.append(value)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [value]

            clusters.append(current_cluster)
            return clusters

        threshold = 0.3  # Počáteční prahová hodnota
        while True:
            clusters = cluster_values(threshold)
            if len(clusters) <= max_clusters:
                break
            threshold += 0.3

        # Výpočet průměrů pro každý shluk
        cluster_averages = [round(sum(cluster) / len(cluster), 2) for cluster in clusters]
        return cluster_averages

    def percentage_drop_indicator(self, dataframe, period, threshold=0.3):
        # Výpočet nejvyšší ceny za poslední období
        highest_high = dataframe['high'].rolling(period).max()
        # Vypočet procentuálního poklesu pod nejvyšší cenou
        percentage_drop = (highest_high - dataframe['close']) / highest_high * 100
        dataframe.loc[percentage_drop < threshold, 'percentage_drop_buy'] = 1
        dataframe.loc[percentage_drop > threshold, 'percentage_drop_buy'] = 0
        return dataframe

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD', 'PAX', 'USD', 'EUR', 'GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.extend(
            ((btc_info_pair, self.timeframe), (btc_info_pair, self.inf_1h))
        )
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Přidání nových sloupců kvůli funkcionalitě ostatních metod
        if 'sell' not in dataframe.columns:
            dataframe.loc[:, 'sell'] = 0
        if 'sell_tag' not in dataframe.columns:
            dataframe.loc[:, 'sell_tag'] = ''
        if 'buy' not in dataframe.columns:
            dataframe.loc[:, 'buy'] = 0
        if 'buy_tag' not in dataframe.columns:
            dataframe.loc[:, 'buy_tag'] = ''

        dataframe['price_history'] = dataframe['close'].shift(1)
        low_min = dataframe['low'].rolling(window=14).min()
        high_max = dataframe['high'].rolling(window=14).max()
        dataframe['stoch_k'] = 100 * (dataframe['close'] - low_min) / (high_max - low_min)
        dataframe['stoch_d'] = dataframe['stoch_k'].rolling(window=3).mean()

        # Výpočet Fibonacciho retracement úrovní
        dataframe['high_max'] = dataframe['high'].rolling(window=30).max()  # posledních 30 svíček
        dataframe['low_min'] = dataframe['low'].rolling(window=30).min()

        # Výpočet Fibonacciho úrovní
        diff = dataframe['high_max'] - dataframe['low_min']
        dataframe['fib_236'] = dataframe['high_max'] - 0.236 * diff
        dataframe['fib_382'] = dataframe['high_max'] - 0.382 * diff
        dataframe['fib_500'] = dataframe['high_max'] - 0.500 * diff
        dataframe['fib_618'] = dataframe['high_max'] - 0.618 * diff
        dataframe['fib_786'] = dataframe['high_max'] - 0.786 * diff

        if self.config['stake_currency'] in ['USDT', 'BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        condition = dataframe['ema_8'] > dataframe['ema_14']
        percentage_difference = 100 * (dataframe['ema_8'] - dataframe['ema_14']).abs() / dataframe['ema_14']
        dataframe['ema_pct_diff'] = percentage_difference.where(condition, -percentage_difference)
        dataframe['prev_ema_pct_diff'] = dataframe['ema_pct_diff'].shift(1)

        crossover_up = (dataframe['ema_8'].shift(1) < dataframe['ema_14'].shift(1)) & (
                dataframe['ema_8'] > dataframe['ema_14'])

        close_to_crossover_up = (dataframe['ema_8'] < dataframe['ema_14']) & (
                dataframe['ema_8'].shift(1) < dataframe['ema_14'].shift(1)) & (
                                        dataframe['ema_8'] > dataframe['ema_8'].shift(1))

        ema_buy_signal = ((dataframe['ema_pct_diff'] < 0) & (dataframe['prev_ema_pct_diff'] < 0) & (
                dataframe['ema_pct_diff'].abs() < dataframe['prev_ema_pct_diff'].abs()))

        dataframe['ema_diff_buy_signal'] = ((ema_buy_signal | crossover_up | close_to_crossover_up)
                                            & (dataframe['rsi'] <= 55) & (dataframe['volume'] > 0))

        dataframe['ema_diff_sell_signal'] = ((dataframe['ema_pct_diff'] > 0) &
                                             (dataframe['prev_ema_pct_diff'] > 0) &
                                             (dataframe['ema_pct_diff'].abs() < dataframe['prev_ema_pct_diff'].abs()))

        dataframe = self.pump_dump_protection(dataframe)

        # Bullish Divergence
        # Ujistěte se, že výpočet používá pouze historická data
        low_min = dataframe['low'].rolling(window=14).min()
        rsi_min = dataframe['rsi'].rolling(window=14).min()
        bullish_div = (low_min.shift(1) > low_min) & (rsi_min.shift(1) < rsi_min)
        dataframe['bullish_divergence'] = bullish_div.astype(int)

        # Fractals
        # Upravte tak, aby se nezahrnovala budoucí data
        dataframe['fractal_top'] = (dataframe['high'] > dataframe['high'].shift(2)) & \
                                   (dataframe['high'] > dataframe['high'].shift(1)) & \
                                   (dataframe['high'] > dataframe['high']) & \
                                   (dataframe['high'] > dataframe['high'].shift(-1))
        dataframe['fractal_bottom'] = (dataframe['low'] < dataframe['low'].shift(2)) & \
                                      (dataframe['low'] < dataframe['low'].shift(1)) & \
                                      (dataframe['low'] < dataframe['low']) & \
                                      (dataframe['low'] < dataframe['low'].shift(-1))

        dataframe['turnaround_signal'] = bullish_div & (dataframe['fractal_bottom'])
        dataframe['rolling_max'] = dataframe['high'].cummax()
        dataframe['drawdown'] = (dataframe['rolling_max'] - dataframe['low']) / dataframe['rolling_max']
        dataframe['below_90_percent_drawdown'] = dataframe['drawdown'] >= 0.90

        # MACD a Volatility Factor
        # MACD výpočet zůstává nezměněn
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Výpočet volatility (použití rolling standard deviation)
        dataframe['volatility'] = dataframe['close'].rolling(window=14).std()

        # Výpočet volatility pomocí ATR nebo standardní odchylky
        dataframe['volatility'] = dataframe['close'].rolling(window=14).std()

        # Normalizace volatility
        min_volatility = dataframe['volatility'].rolling(window=14).min()
        max_volatility = dataframe['volatility'].rolling(window=14).max()
        dataframe['volatility_factor'] = (dataframe['volatility'] - min_volatility) / \
                                         (max_volatility - min_volatility)

        # Přizpůsobení MACD na základě volatility
        dataframe['macd_adjusted'] = dataframe['macd'] * (1 - dataframe['volatility_factor'])
        dataframe['macdsignal_adjusted'] = dataframe['macdsignal'] * (1 + dataframe['volatility_factor'])

        dataframe = self.percentage_drop_indicator(dataframe, 9, threshold=0.21)

        ichi = ichimoku(dataframe)
        dataframe['senkou_span_a'] = ichi['senkou_span_a']
        dataframe['senkou_span_b'] = ichi['senkou_span_b']

        # Vytvoření vah pro vážený průměr
        weights = np.linspace(1, 0, 300)  # Váhy od 1 (nejnovější) do 0 (nejstarší)
        weights /= weights.sum()  # Normalizace vah tak, aby jejich součet byl 1

        # Výpočet váženého průměru RSI pro posledních 300 svící
        dataframe['weighted_rsi'] = dataframe['rsi'].rolling(window=300).apply(
            lambda x: np.sum(weights * x[-300:]), raw=False
        )

        dataframe['sar'] = ta.SAR(dataframe, start=self.start, increment=self.increment, maximum=self.maximum)
        dataframe['sar_buy'] = (dataframe['sar'] < dataframe['low']).astype(int)
        dataframe['sar_sell'] = (dataframe['sar'] > dataframe['high']).astype(int)

        resampled_frame = dataframe.resample('5T', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        resampled_frame['higher_tf_trend'] = (resampled_frame['close'] > resampled_frame['open']).astype(int)
        resampled_frame['higher_tf_trend'] = resampled_frame['higher_tf_trend'].replace({1: 1, 0: -1})
        dataframe['higher_tf_trend'] = dataframe['date'].map(resampled_frame['higher_tf_trend'])

        # Zjištění hladin supportů a resistancí
        self.calculate_support_resistance_dicts(metadata['pair'], dataframe)

        # Přidání RSI a EMA indikátorů
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Detekce bullish svíčkových vzorů
        dataframe['bullish_engulfing'] = ta.CDLENGULFING(dataframe['open'], dataframe['high'], dataframe['low'],
                                                         dataframe['close']) > 0
        dataframe['hammer'] = ta.CDLHAMMER(dataframe['open'], dataframe['high'], dataframe['low'],
                                           dataframe['close']) > 0
        # Přidání SAR
        dataframe['sar'] = ta.SAR(dataframe, start=self.start, increment=self.increment, maximum=self.maximum)
        dataframe['sar_buy'] = (dataframe['sar'] < dataframe['low']).astype(int)
        dataframe['sar_sell'] = (dataframe['sar'] > dataframe['high']).astype(int)

        # Identifikace úrovní podpory a odporu
        dataframe['support'] = dataframe['close'].rolling(window=20).min()
        dataframe['resistance'] = dataframe['close'].rolling(window=20).max()

        # stochastic_cond = (
        #         (dataframe['stoch_k'] <= self.stoch_treshold.value) &
        #         (dataframe['stoch_d'] <= self.stoch_treshold.value) &
        #         (dataframe['stoch_k'] > dataframe['stoch_d'])
        # )
        # dataframe.loc[stochastic_cond, 'buy_tag'] = 'buy_stoch_kd'
        # dataframe.loc[stochastic_cond, 'buy'] = 1

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        fib_cond = (
                (dataframe['close'] <= dataframe['fib_618']) &
                (dataframe['close'] >= dataframe['fib_618'] * 0.99)
        )
        dataframe.loc[fib_cond, 'buy_tag'] += 'fib_0618_'
        conditions.append(fib_cond)

        # stochastic_cond = (
        #         (dataframe['stoch_k'] <= self.stoch_treshold.value) &
        #         (dataframe['stoch_d'] <= self.stoch_treshold.value) &
        #         (dataframe['stoch_k'] > dataframe['stoch_d'])
        # )
        # dataframe.loc[stochastic_cond, 'buy_tag'] += 'stoch_kd_'
        # conditions.append(stochastic_cond)
        #
        # lambo2 = (
        #     # bool(self.lambo2_enabled.value) &
        #     # (dataframe['pump_warning'] == 0) &
        #         (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
        #         (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
        #         (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        # )
        # dataframe.loc[lambo2, 'buy_tag'] += 'lambo2_'
        # conditions.append(lambo2)
        #
        # buy1ewo = (
        #         (dataframe['rsi_fast'] < 35) &
        #         (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
        #         (dataframe['EWO'] > self.ewo_high.value) &
        #         (dataframe['rsi'] < self.rsi_buy.value) &
        #         (dataframe['volume'] > 0) &
        #         (dataframe['close'] < (
        #                 dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        # )
        # dataframe.loc[buy1ewo, 'buy_tag'] += 'buy1eworsi_'
        # conditions.append(buy1ewo)
        #
        # buy2ewo = (
        #         (dataframe['rsi_fast'] < 35) &
        #         (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
        #         (dataframe['EWO'] < self.ewo_low.value) &
        #         (dataframe['volume'] > 0) &
        #         (dataframe['close'] < (
        #                 dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        # )
        # dataframe.loc[buy2ewo, 'buy_tag'] += 'buy2ewo_'
        # conditions.append(buy2ewo)
        #
        # is_cofi = (
        #         (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
        #         (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
        #         (dataframe['fastk'] < self.buy_fastk.value) &
        #         (dataframe['fastd'] < self.buy_fastd.value) &
        #         (dataframe['adx'] > self.buy_adx.value) &
        #         (dataframe['EWO'] > self.buy_ewo_high.value)
        # )
        # dataframe.loc[is_cofi, 'buy_tag'] += 'cofi_'
        # conditions.append(is_cofi)

        # """ Přidání potvrzení nákupního signálu """
        cond_sar = self.confirm_by_sar(dataframe)
        # cond_candles = self.confirm_by_candles(dataframe)
        dataframe.loc[cond_sar, 'buy_tag'] += 'sar_'
        # dataframe.loc[cond_candles, 'buy_tag'] += 'candles_'
        conditions.append(cond_sar)
        # conditions.append(cond_candles)

        if conditions:
            final_condition = reduce(lambda x, y: x & y, conditions)
            dataframe.loc[final_condition, 'buy'] = 1

        dont_buy_conditions = [
            dataframe['pnd_volume_warn'] < 0.0,
            dataframe['btc_rsi_8_1h'] < 35.0
        ]

        for condition in dont_buy_conditions:
            dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['fib_618'])
            ),
            'sell'] = 1
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        # """ Získání dat pro potvrzení nákupního signálu """
        # try:
        #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        #     df = dataframe.copy()
        # except Exception as e:
        #     logging.error(f"Error getting analyzed dataframe: {e}")
        #     return False
        #
        # # Získání aktuální svíčky
        # last_candle = df.iloc[-1].squeeze()
        # # Podmínky pro potvrzení nákupního signálu
        # cond_candles = self.confirm_by_candles(last_candle)
        # cond_sar = self.confirm_by_sar(last_candle)

        # Příprava výsledku
        result = (Trade.get_open_trade_count() < self.open_trade_limit.value)
        # and (cond_candles or cond_sar)
        return result

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        if current_profit < -0.05 and (current_time - trade.open_date_utc).days >= 30:
            return 'unclog'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        sell_reason = f"{sell_reason}_" + trade.buy_tag
        current_profit = trade.calc_profit_ratio(rate)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        # Aktuální hodnoty EMA
        ema_8_current = dataframe['ema_8'].iat[-1]
        ema_14_current = dataframe['ema_14'].iat[-1]

        # Hodnoty EMA předchozí svíčky
        ema_8_previous = dataframe['ema_8'].iat[-2]
        ema_14_previous = dataframe['ema_14'].iat[-2]

        # Výpočet rozdílu EMA mezi aktuální a předchozí svíčkou
        diff_current = abs(ema_8_current - ema_14_current)
        diff_previous = abs(ema_8_previous - ema_14_previous)

        # Výpočet procentní změny mezi diff_current a diff_previous
        diff_change_pct = (diff_previous - diff_current) / diff_previous

        if 'unclog' in sell_reason or 'force' in sell_reason:
            logging.info(f"CTE - FORCE or UNCLOG, EXIT")
            return True
        elif current_profit >= 0.0025:
            if ema_8_current <= ema_14_current and diff_change_pct >= 0.025:
                logging.info(
                    f"CTE - EMA 8 {ema_8_current} <= EMA 14 {ema_14_current} with decrease in difference >= 3%, EXIT")
                return True
            elif ema_8_current > ema_14_current and diff_current > diff_previous:
                logging.info(f"CTE - EMA 8 {ema_8_current} > EMA 14 {ema_14_current} with increasing difference, HOLD")
                return False
            else:
                logging.info(f"CTE - Conditions not met, EXIT")
                return True
        else:
            return False

    def confirm_by_sar(self, data_dict):
        """ Based on TA indicators, populates the buy signal for the given dataframe """
        cond = (data_dict['sar_buy'] > 0)
        return cond

    def confirm_by_candles(self, data_dict):
        """ Based on TA indicators, populates the buy signal for the given dataframe """
        cond = ((data_dict['rsi'] < 30) &
                (data_dict['close'] > data_dict['ema']) &
                (data_dict['bullish_engulfing'] | data_dict['hammer']) &
                (data_dict['low'] < data_dict['support']) | (data_dict['high'] > data_dict['resistance']))
        return cond

    def base_tf_btc_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['price_trend_long'] = (
                dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        return dataframe

    def pump_dump_protection(self, dataframe: DataFrame) -> DataFrame:
        df36h = dataframe.copy().shift(432)
        df24h = dataframe.copy().shift(288)
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()
        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])
        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0),
                                                -1, 0)
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        current_time = datetime.utcnow()  # Datový typ: datetime

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            df = dataframe.copy()  # Datový typ: pandas DataFrame
        except Exception as e:
            logging.error(f"Error getting analyzed dataframe: {e}")
            return None

        # Kontrola, zda obchodní pár má definovanou podporu
        if trade.pair in self.support_dict:
            # Získání seznamu podpor pro daný obchodní pár
            s = self.support_dict[trade.pair]  # Datový typ: list
            # Výpočet nejbližší podpory pro každou cenu uzavření v dataframe
            df['nearest_support'] = df['close'].apply(
                lambda x: min([support for support in s if support <= x], default=x,
                              key=lambda support: abs(x - support))
            )

            if 'nearest_support' in df.columns:
                # Získání poslední svíčky (candle) z dataframe
                last_candle = df.iloc[-1]  # Datový typ: pandas Series
                if 'nearest_support' in last_candle:
                    nearest_support = last_candle['nearest_support']  # Datový typ: float
                    # Výpočet procentní vzdálenosti k nejbližší podpoře
                    distance_to_support_pct = abs(
                        (nearest_support - current_rate) / current_rate)  # Datový typ: float, jednotka: %
                    # Kontrola, zda je aktuální kurz blízko nebo pod nejbližší podporou
                    if (0 <= distance_to_support_pct <= self.distance_to_support_treshold.value) or (
                            current_rate < nearest_support):
                        # Počítání uzavřených nákupních příkazů
                        count_of_buys = sum(order.ft_order_side == 'buy' and order.status == 'closed' for order in
                                            trade.orders)  # Datový typ: int
                        # Zjištění času posledního nákupu
                        last_buy_time = max(
                            [order.order_date for order in trade.orders if order.ft_order_side == 'buy'],
                            default=trade.open_date_utc)
                        last_buy_time = last_buy_time.replace(
                            tzinfo=None)  # Odstranění časové zóny, Datový typ: datetime
                        # Výpočet intervalu svíčky (candle) v minutách
                        candle_interval = self.timeframe_to_minutes(self.timeframe)  # Datový typ: int, jednotka: minuty
                        # Výpočet času od posledního nákupu v minutách
                        time_since_last_buy = (
                                                      current_time - last_buy_time).total_seconds() / 60  # Datový typ: float, jednotka: minuty
                        # Výpočet počtu svíček, které musí uplynout před dalším nákupem
                        candles = self.candles_before.value + (
                                self.candles_dca_multiplier.value * (count_of_buys - 1))  # Datový typ: int
                        # Kontrola, zda uplynul dostatečný čas od posledního nákupu
                        if time_since_last_buy < candles * candle_interval:
                            return None
                        # Kontrola, zda počet bezpečnostních příkazů (safety orders) není překročen
                        if self.max_safety_orders.value >= count_of_buys:
                            # Hledání posledního uzavřeného nákupního příkazu
                            last_buy_order = None
                            for order in reversed(trade.orders):
                                if order.ft_order_side == 'buy' and order.status == 'closed':
                                    last_buy_order = order
                                    break

                            # Výpočet procentní rozdílu mezi posledním nákupním příkazem a aktuálním kurzem
                            pct_diff = self.calculate_percentage_difference(original_price=last_buy_order.price,
                                                                            current_price=current_rate)  # Datový typ: float, jednotka: %
                            # Kontrola, zda je procentní rozdíl menší než prahová hodnota
                            if pct_diff <= -self.pct_drop_treshold.value:
                                if last_buy_order and current_rate < last_buy_order.price:
                                    # Kontrola RSI podmínky pro DCA
                                    rsi_value = last_candle['rsi']  # Předpokládá se, že RSI je součástí dataframe
                                    w_rsi = last_candle[
                                        'weighted_rsi']  # Předpokládá se, že Weighted RSI je součástí dataframe

                                    if rsi_value <= w_rsi:
                                        # Logování informací o obchodu
                                        logging.info(
                                            f'AP1 {trade.pair}, Profit: {current_profit}, Stake {trade.stake_amount}')

                                        # Získání celkové částky sázky v peněžence
                                        total_stake_amount = self.wallets.get_total_stake_amount() / self.dca_wallet_divider.value  # Datový typ: float

                                        # Výpočet částky pro další sázku pomocí DCA (Dollar Cost Averaging)
                                        calculated_dca_stake = self.calculate_dca_price(base_value=trade.stake_amount,
                                                                                        decline=current_profit * 100,
                                                                                        target_percent=1)  # Datový typ: float
                                        # Upravení velikosti sázky, pokud je vyšší než dostupný zůstatek
                                        while calculated_dca_stake >= total_stake_amount:
                                            calculated_dca_stake = calculated_dca_stake / self.dca_order_divider.value  # Datový typ: float
                                        # Logování informací o upravené sázce
                                        logging.info(f'AP2 {trade.pair}, DCA: {calculated_dca_stake}')
                                        # Vrácení upravené velikosti sázky
                                        return calculated_dca_stake
            # Vrácení None, pokud nejsou splněny podmínky pro upravení obchodní pozice
            return None

    def calculate_percentage_difference(self, original_price, current_price):
        percentage_diff = ((current_price - original_price) / original_price)
        return percentage_diff

    def calculate_dca_price(self, base_value, decline, target_percent):
        return (((base_value / 100) * abs(decline)) / target_percent) * 100

    def timeframe_to_minutes(self, timeframe):
        """Převede timeframe na minuty."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError("Neznámý timeframe: {}".format(timeframe))


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    return (ema1 - ema2) / df['close'] * 100