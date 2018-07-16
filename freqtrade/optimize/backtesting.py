# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
import operator
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import arrow
from pandas import DataFrame, to_datetime
from tabulate import tabulate

import freqtrade.optimize as optimize
from freqtrade import DependencyException, constants
from freqtrade.analyze import Analyze
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Exchange
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade
from profilehooks import profile
from collections import OrderedDict
import timeit
from time import sleep

logger = logging.getLogger(__name__)


class BacktestResult(NamedTuple):
    """
    NamedTuple Defining BacktestResults inputs.
    """
    pair: str
    profit_percent: float
    profit_abs: float
    open_time: datetime
    close_time: datetime
    open_index: int
    close_index: int
    trade_duration: float
    open_at_end: bool
    open_rate: float
    close_rate: float


class Backtesting(object):
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.analyze = Analyze(self.config)
        self.ticker_interval = self.analyze.strategy.ticker_interval
        self.tickerdata_to_dataframe = self.analyze.tickerdata_to_dataframe
        self.populate_buy_trend = self.analyze.populate_buy_trend
        self.populate_sell_trend = self.analyze.populate_sell_trend

        # Reset keys for backtesting
        self.config['exchange']['key'] = ''
        self.config['exchange']['secret'] = ''
        self.config['exchange']['password'] = ''
        self.config['exchange']['uid'] = ''
        self.config['dry_run'] = True
        self.exchange = Exchange(self.config)
        self.fee = self.exchange.get_fee()

        self.stop_loss_value = self.analyze.strategy.stoploss

        #### backslap config
        '''
        Numpy arrays are used for 100x speed up
        We requires setting Int values for
        buy stop triggers and stop calculated on
        # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5 - stop 6
        '''
        self.np_buy: int = 0
        self.np_open: int = 1
        self.np_close: int = 2
        self.np_sell: int = 3
        self.np_high: int = 4
        self.np_low: int = 5
        self.np_stop: int = 6
        self.np_bto: int = self.np_close  # buys_triggered_on - should be close
        self.np_bco: int = self.np_open  # buys calculated on - open of the next candle.
        self.np_sto: int = self.np_low  # stops_triggered_on - Should be low, FT uses close
        self.np_sco: int = self.np_stop  # stops_calculated_on - Should be stop, FT uses close
        #self.np_sto: int = self.np_close  # stops_triggered_on - Should be low, FT uses close
        #self.np_sco: int = self.np_close  # stops_calculated_on - Should be stop, FT uses close

        self.use_backslap = True             # Enable backslap - if false Orginal code is executed.
        self.debug = False                   # Main debug enable, very print heavy, enable 2 loops recommended
        self.debug_timing = True            # Stages within Backslap
        self.debug_2loops = False            # Limit each pair to two loops, useful when debugging
        self.debug_vector = False            # Debug vector calcs
        self.debug_timing_main_loop = False  # print overall timing per pair - works in Backtest and Backslap

        self.backslap_show_trades = False     # prints trades in addition to summary report
        self.backslap_save_trades = True      # saves trades as a pretty table to backslap.txt


    @staticmethod
    def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
        """
        Get the maximum timeframe for the given backtest data
        :param data: dictionary with preprocessed backtesting data
        :return: tuple containing min_date, max_date
        """
        timeframe = [
            (arrow.get(min(frame.date)), arrow.get(max(frame.date)))
            for frame in data.values()
        ]
        return min(timeframe, key=operator.itemgetter(0))[0], \
            max(timeframe, key=operator.itemgetter(1))[1]

    def _generate_text_table(self, data: Dict[str, Dict], results: DataFrame) -> str:
        """
        Generates and returns a text table for the given backtest data and the results dataframe
        :return: pretty printed table with tabulate as str
        """
        stake_currency = str(self.config.get('stake_currency'))

        floatfmt = ('s', 'd', '.2f', '.2f', '.8f', '.1f')
        tabular_data = []
        headers = ['pair', 'buy count', 'avg profit %', 'cum profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for pair in data:
            result = results[results.pair == pair]
            tabular_data.append([
                pair,
                len(result.index),
                result.profit_percent.mean() * 100.0,
                result.profit_percent.sum() * 100.0,
                result.profit_abs.sum(),
                result.trade_duration.mean(),
                len(result[result.profit_abs > 0]),
                len(result[result.profit_abs < 0])
            ])

        # Append Total
        tabular_data.append([
            'TOTAL',
            len(results.index),
            results.profit_percent.mean() * 100.0,
            results.profit_percent.sum() * 100.0,
            results.profit_abs.sum(),
            results.trade_duration.mean(),
            len(results[results.profit_abs > 0]),
            len(results[results.profit_abs < 0])
        ])
        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt, tablefmt="pipe")

    def _store_backtest_result(self, recordfilename: Optional[str], results: DataFrame) -> None:

        records = [(t.pair, t.profit_percent, t.open_time.timestamp(),
                    t.close_time.timestamp(), t.open_index - 1, t.trade_duration,
                    t.open_rate, t.close_rate, t.open_at_end)
                   for index, t in results.iterrows()]

        if records:
            logger.info('Dumping backtest results to %s', recordfilename)
            file_dump_json(recordfilename, records)

    def _get_sell_trade_entry(
            self, pair: str, buy_row: DataFrame,
            partial_ticker: List, trade_count_lock: Dict, args: Dict) -> Optional[BacktestResult]:

        stake_amount = args['stake_amount']
        max_open_trades = args.get('max_open_trades', 0)
        trade = Trade(
            open_rate=buy_row.open,
            open_date=buy_row.date,
            stake_amount=stake_amount,
            amount=stake_amount / buy_row.open,
            fee_open=self.fee,
            fee_close=self.fee
        )

        # calculate win/lose forwards from buy point
        for sell_row in partial_ticker:
            if max_open_trades > 0:
                # Increase trade_count_lock for every iteration
                trade_count_lock[sell_row.date] = trade_count_lock.get(sell_row.date, 0) + 1

            buy_signal = sell_row.buy
            if self.analyze.should_sell(trade, sell_row.open, sell_row.date, buy_signal,
                                        sell_row.sell):

                return BacktestResult(pair=pair,
                                      profit_percent=trade.calc_profit_percent(rate=sell_row.open),
                                      profit_abs=trade.calc_profit(rate=sell_row.open),
                                      open_time=buy_row.date,
                                      close_time=sell_row.date,
                                      trade_duration=(sell_row.date - buy_row.date).seconds // 60,
                                      open_index=buy_row.Index,
                                      close_index=sell_row.Index,
                                      open_at_end=False,
                                      open_rate=buy_row.open,
                                      close_rate=sell_row.open
                                      )
        if partial_ticker:
            # no sell condition found - trade stil open at end of backtest period
            sell_row = partial_ticker[-1]
            btr = BacktestResult(pair=pair,
                                 profit_percent=trade.calc_profit_percent(rate=sell_row.open),
                                 profit_abs=trade.calc_profit(rate=sell_row.open),
                                 open_time=buy_row.date,
                                 close_time=sell_row.date,
                                 trade_duration=(sell_row.date - buy_row.date).seconds // 60,
                                 open_index=buy_row.Index,
                                 close_index=sell_row.Index,
                                 open_at_end=True,
                                 open_rate=buy_row.open,
                                 close_rate=sell_row.open
                                 )
            logger.debug('Force_selling still open trade %s with %s perc - %s', btr.pair,
                         btr.profit_percent, btr.profit_abs)
            return btr
        return None

    def s(self):
        st = timeit.default_timer()
        return st

    def f(self, st):
        return (timeit.default_timer() - st)


    def backtest(self, args: Dict) -> DataFrame:
        """
        Implements backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid, logging on this method

        :param args: a dict containing:
            stake_amount: btc amount to use for each trade
            processed: a processed dictionary with format {pair, data}
            max_open_trades: maximum number of concurrent trades (default: 0, disabled)
            realistic: do we try to simulate realistic trades? (default: True)
        :return: DataFrame
        """

        use_backslap = self.use_backslap
        debug_timing = self.debug_timing_main_loop

        if use_backslap: # Use Back Slap code

            headers = ['date', 'buy', 'open', 'close', 'sell', 'high', 'low']
            processed = args['processed']
            max_open_trades = args.get('max_open_trades', 0)
            realistic = args.get('realistic', False)
            trades = []
            trade_count_lock: Dict = {}

            ########################### Call out BSlap Loop instead of Original BT code
            bslap_results: list = []
            for pair, pair_data in processed.items():
                if debug_timing: # Start timer
                    fl = self.s()

                ticker_data = self.populate_sell_trend(
                        self.populate_buy_trend(pair_data))[headers].copy()

                if debug_timing: # print time taken
                    flt = self.f(fl)
                    #print("populate_buy_trend:", pair, round(flt, 10))
                    st = self.s()

                # #dump same DFs to disk for offline testing in scratch
                # f_pair:str = pair
                # csv = f_pair.replace("/", "_")
                # csv="/Users/creslin/PycharmProjects/freqtrade_new/frames/" + csv
                # ticker_data.to_csv(csv, sep='\t', encoding='utf-8')

                #call bslap - results are a list of dicts
                bslap_pair_results = self.backslap_pair(ticker_data, pair)
                last_bslap_results = bslap_results
                bslap_results = last_bslap_results + bslap_pair_results

                if debug_timing:  # print time taken
                    tt = self.f(st)
                    print("Time to  BackSlap :", pair, round(tt,10))
                    print("-----------------------")


            # Switch List of Trade Dicts (bslap_results) to Dataframe
            # Fill missing, calculable columns, profit, duration , abs etc.
            bslap_results_df = DataFrame(bslap_results)
            bslap_results_df['open_time'] = to_datetime(bslap_results_df['open_time'])
            bslap_results_df['close_time'] = to_datetime(bslap_results_df['close_time'])

            ### don't use this, itll drop exit type field
            # bslap_results_df = DataFrame(bslap_results, columns=BacktestResult._fields)

            bslap_results_df = self.vector_fill_results_table(bslap_results_df)

            return bslap_results_df

        else: # use Original Back test code
            ########################## Original BT loop

            headers = ['date', 'buy', 'open', 'close', 'sell']
            processed = args['processed']
            max_open_trades = args.get('max_open_trades', 0)
            realistic = args.get('realistic', False)
            trades = []
            trade_count_lock: Dict = {}

            for pair, pair_data in processed.items():
                if debug_timing: # Start timer
                    fl = self.s()

                pair_data['buy'], pair_data['sell'] = 0, 0  # cleanup from previous run

                ticker_data = self.populate_sell_trend(
                    self.populate_buy_trend(pair_data))[headers].copy()

                # to avoid using data from future, we buy/sell with signal from previous candle
                ticker_data.loc[:, 'buy'] = ticker_data['buy'].shift(1)
                ticker_data.loc[:, 'sell'] = ticker_data['sell'].shift(1)

                ticker_data.drop(ticker_data.head(1).index, inplace=True)

                if debug_timing: # print time taken
                    flt = self.f(fl)
                    #print("populate_buy_trend:", pair, round(flt, 10))
                    st = self.s()

                # Convert from Pandas to list for performance reasons
                # (Looping Pandas is slow.)
                ticker = [x for x in ticker_data.itertuples()]

                lock_pair_until = None
                for index, row in enumerate(ticker):
                    if row.buy == 0 or row.sell == 1:
                        continue  # skip rows where no buy signal or that would immediately sell off

                    if realistic:
                        if lock_pair_until is not None and row.date <= lock_pair_until:
                            continue
                    if max_open_trades > 0:
                        # Check if max_open_trades has already been reached for the given date
                        if not trade_count_lock.get(row.date, 0) < max_open_trades:
                            continue

                        trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                    trade_entry = self._get_sell_trade_entry(pair, row, ticker[index + 1:],
                                                             trade_count_lock, args)


                    if trade_entry:
                        lock_pair_until = trade_entry.close_time
                        trades.append(trade_entry)
                    else:
                        # Set lock_pair_until to end of testing period if trade could not be closed
                        # This happens only if the buy-signal was with the last candle
                        lock_pair_until = ticker_data.iloc[-1].date

                if debug_timing:  # print time taken
                    tt = self.f(st)
                    print("Time to BackTest :", pair, round(tt, 10))
                    print("-----------------------")

            return DataFrame.from_records(trades, columns=BacktestResult._fields)
            ####################### Original BT loop end

    def vector_fill_results_table(self, bslap_results_df: DataFrame):
        """
        The Results frame contains a number of columns that are calculable
        from othe columns. These are left blank till all rows are added,
        to be populated in single vector calls.

        Columns to be populated are:
        - Profit
        - trade duration
        - profit abs

        :param bslap_results Dataframe
        :return: bslap_results Dataframe
        """
        import pandas as pd
        debug = self.debug_vector

        # stake and fees
        # stake = 0.015
        # 0.05% is 0.0005
        #fee = 0.001

        stake = self.config.get('stake_amount')
        fee = self.fee
        open_fee = fee / 2
        close_fee = fee / 2

        if debug:
            print("Stake is,", stake, "the sum of currency to spend per trade")
            print("The open fee is", open_fee, "The close fee is", close_fee)
        if debug:
            from pandas import set_option
            set_option('display.max_rows', 5000)
            set_option('display.max_columns', 10)
            pd.set_option('display.width', 1000)
            pd.set_option('max_colwidth', 40)
            pd.set_option('precision', 12)

        # Populate duration
        bslap_results_df['trade_duration'] = bslap_results_df['close_time'] - bslap_results_df['open_time']
        # if debug:
        #     print(bslap_results_df[['open_time', 'close_time', 'trade_duration']])

        ## Spends, Takes, Profit, Absolute Profit
        # Buy Price
        bslap_results_df['buy_sum'] = stake * bslap_results_df['open_rate']
        bslap_results_df['buy_fee'] = bslap_results_df['buy_sum'] * open_fee
        bslap_results_df['buy_spend'] = bslap_results_df['buy_sum'] + bslap_results_df['buy_fee']
        # Sell price
        bslap_results_df['sell_sum'] = stake * bslap_results_df['close_rate']
        bslap_results_df['sell_fee'] = bslap_results_df['sell_sum'] * close_fee
        bslap_results_df['sell_take'] = bslap_results_df['sell_sum'] - bslap_results_df['sell_fee']
        # profit_percent
        bslap_results_df['profit_percent'] = bslap_results_df['sell_take'] / bslap_results_df['buy_spend'] - 1
        # Absolute profit
        bslap_results_df['profit_abs'] = bslap_results_df['sell_take'] - bslap_results_df['buy_spend']

        if debug:
            print("\n")
            print(bslap_results_df[
                      ['buy_sum', 'buy_fee', 'buy_spend', 'sell_sum','sell_fee', 'sell_take', 'profit_percent', 'profit_abs', 'exit_type']])

        return bslap_results_df

    def np_get_t_open_ind(self, np_buy_arr, t_exit_ind: int):
        import utils_find_1st as utf1st
        """
         The purpose of this def is to return the next "buy" = 1
         after t_exit_ind.

         t_exit_ind is the index the last trade exited on
         or 0 if first time around this loop.
         """
        # Timers, to be called if in debug
        def s():
            st = timeit.default_timer()
            return st
        def f(st):
            return (timeit.default_timer() - st)

        st = s()
        t_open_ind: int

        """
        Create a view on our buy index starting after last trade exit
        Search for next buy
        """
        np_buy_arr_v = np_buy_arr[t_exit_ind:]
        t_open_ind = utf1st.find_1st(np_buy_arr_v, 1, utf1st.cmp_equal)

        '''
        If -1 is returned no buy has been found, preserve the value
        '''
        if t_open_ind != -1:  # send back the -1 if no buys found. otherwise update index
            t_open_ind = t_open_ind + t_exit_ind  # Align numpy index

        return t_open_ind

    def backslap_pair(self, ticker_data, pair):
        import pandas as pd
        import numpy as np
        import timeit
        import utils_find_1st as utf1st
        from datetime import datetime

        ### backslap debug wrap
        # debug_2loops = False  # only loop twice, for faster debug
        # debug_timing = False  # print timing for each step
        # debug = False  # print values, to check accuracy
        debug_2loops = self.debug_2loops  # only loop twice, for faster debug
        debug_timing = self.debug_timing  # print timing for each step
        debug = self.debug  # print values, to check accuracy

        # Read Stop Loss Values and Stake
        stop = self.stop_loss_value
        p_stop = (stop + 1)  # What stop really means, e.g 0.01 is 0.99 of price

        if debug:
            print("Stop is ", stop, "value from stragey file")
            print("p_stop is", p_stop, "value used to multiply to entry price")

        if debug:
            from pandas import set_option
            set_option('display.max_rows', 5000)
            set_option('display.max_columns', 8)
            pd.set_option('display.width', 1000)
            pd.set_option('max_colwidth', 40)
            pd.set_option('precision', 12)
        def s():
            st = timeit.default_timer()
            return st
        def f(st):
            return (timeit.default_timer() - st)
        #### backslap config
        '''
        Numpy arrays are used for 100x speed up
        We requires setting Int values for
        buy stop triggers and stop calculated on
        # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5 - stop 6
        '''
        # np_buy: int = 0
        # np_open: int = 1
        # np_close: int = 2
        # np_sell: int = 3
        # np_high: int = 4
        # np_low: int = 5
        # np_stop: int = 6
        # np_bto: int = np_close  # buys_triggered_on - should be close
        # np_bco: int = np_open  # buys calculated on - open of the next candle.
        # #np_sto: int = np_low  # stops_triggered_on - Should be low, FT uses close
        # #np_sco: int = np_stop  # stops_calculated_on - Should be stop, FT uses close
        # np_sto: int = np_close  # stops_triggered_on - Should be low, FT uses close
        # np_sco: int = np_close  # stops_calculated_on - Should be stop, FT uses close

        #######
        #  Use vars set at top of backtest
        np_buy: int = self.np_buy
        np_open: int = self.np_open
        np_close: int = self.np_close
        np_sell: int = self.np_sell
        np_high: int = self.np_high
        np_low: int = self.np_low
        np_stop: int = self.np_stop
        np_bto: int = self.np_bto  # buys_triggered_on - should be close
        np_bco: int = self.np_bco  # buys calculated on - open of the next candle.
        np_sto: int = self.np_sto  # stops_triggered_on - Should be low, FT uses close
        np_sco: int = self.np_sco  # stops_calculated_on - Should be stop, FT uses close

        ### End Config

        pair: str = pair

        #ticker_data: DataFrame = ticker_dfs[t_file]
        bslap: DataFrame = ticker_data

        # Build a single dimension numpy array from "buy" index for faster search
        # (500x faster than pandas)
        np_buy_arr = bslap['buy'].values
        np_buy_arr_len: int = len(np_buy_arr)

        # use numpy array for faster searches in loop, 20x faster than pandas
        # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
        np_bslap = np.array(bslap[['buy', 'open', 'close', 'sell', 'high', 'low']])

        # Build a numpy list of date-times.
        # We use these when building the trade
        # The rationale is to address a value from a pandas cell is thousands of
        # times more expensive. Processing time went X25 when trying to use any data from pandas
        np_bslap_dates = bslap['date'].values

        loop: int = 0  # how many time around the loop
        t_exit_ind = 0  # Start loop from first index
        t_exit_last = 0  # To test for exit

        st = s()  # Start timer for processing dataframe
        if debug:
            print('Processing:', pair)

        # Results will be stored in a list of dicts
        bslap_pair_results: list = []
        bslap_result: dict = {}

        while t_exit_ind < np_buy_arr_len:
            loop = loop + 1
            if debug or debug_timing:
                print("-- T_exit_Ind - Numpy Index is", t_exit_ind, " ----------------------- Loop", loop, pair)
            if debug_2loops:
                if loop == 3:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Loop debug max met - breaking")
                    break
            '''
             Dev phases
             Phase 1
              1) Manage buy, sell, stop enter/exit
               a) Find first buy index
               b) Discover first stop and sell hit after buy index
               c) Chose first instance as trade exit
    
             Phase 2
              2) Manage dynamic Stop and ROI Exit
               a) Create trade slice from 1
               b) search within trade slice for dynamice stop hit
               c) search within trade slice for ROI hit
             '''

            if debug_timing:
                st = s()
            '''
            0 - Find next buy entry
            Finds index for first (buy = 1) flag
    
            Requires: np_buy_arr - a 1D array of the 'buy' column. To find next "1"
            Required: t_exit_ind - Either 0, first loop. Or The index we last exited on
            Provides: The next "buy" index after t_exit_ind
    
            If -1 is returned no buy has been found in remainder of array, skip to exit loop
            '''
            t_open_ind = self.np_get_t_open_ind(np_buy_arr, t_exit_ind)

            if debug:
                print("\n(0) numpy debug \nnp_get_t_open, has returned the next valid buy index as", t_open_ind)
                print("If -1 there are no valid buys in the remainder of ticker data. Skipping to end of loop")
            if debug_timing:
                t_t = f(st)
                print("0-numpy", str.format('{0:.17f}', t_t))
                st = s()

            if t_open_ind != -1:

                """
                1 - Create view to search within for our open trade
    
                The view is our search space for the next Stop or Sell
                Numpy view is employed as:
                1,000 faster than pandas searches
                Pandas cannot assure it will always return a view, it may make a slow copy.
    
                The view contains columns:
                buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
    
                Requires: np_bslap is our numpy array of the ticker DataFrame
                Requires: t_open_ind is the index row with the  buy.
                Provided: np_t_open_v View of array after trade.
                """
                np_t_open_v = np_bslap[t_open_ind:]

                if debug:
                    print("\n(1) numpy debug \nNumpy view row 0 is now Ticker_Data Index", t_open_ind)
                    print("Numpy View: Buy - Open - Close - Sell - High - Low")
                    print("Row 0", np_t_open_v[0])
                    print("Row 1", np_t_open_v[1], )
                if debug_timing:
                    t_t = f(st)
                    print("2-numpy", str.format('{0:.17f}', t_t))
                    st = s()

                '''
                2 - Calculate our stop-loss price
    
                As stop is based on buy price of our trade
                - (BTO)Buys are Triggered On np_bto, typically the CLOSE of candle
                - (BCO)Buys are Calculated On np_bco, default is OPEN of the next candle.
                This is as we only see the CLOSE after it has happened.
                The  back test assumption is we have bought at first available price, the OPEN
    
                Requires: np_bslap   - is our numpy array of the ticker DataFrame
                Requires: t_open_ind - is the index row with the first buy.
                Requires: p_stop     - is the stop rate, ie. 0.99 is -1%
                Provides: np_t_stop_pri - The value stop-loss will be triggered on
                '''
                np_t_stop_pri = (np_bslap[t_open_ind + 1, np_bco] * p_stop)

                if debug:
                    print("\n(2) numpy debug\nStop-Loss has been calculated at:", np_t_stop_pri)
                if debug_timing:
                    t_t = f(st)
                    print("2-numpy", str.format('{0:.17f}', t_t))
                    st = s()

                '''
                3 -  Find candle STO is under Stop-Loss After Trade opened.
    
                where [np_sto] (stop tiggered on variable: "close", "low" etc) < np_t_stop_pri
    
                Requires: np_t_open_v   Numpy view of ticker_data after trade open
                Requires: np_sto        User Var(STO)StopTriggeredOn. Typically set to "low" or "close"
                Requires: np_t_stop_pri The stop-loss price STO must fall under to trigger stop
                Provides: np_t_stop_ind The first candle after trade open where STO is under stop-loss
                '''
                np_t_stop_ind = utf1st.find_1st(np_t_open_v[:, np_sto],
                                                np_t_stop_pri,
                                                utf1st.cmp_smaller)

                if debug:
                    print("\n(3) numpy debug\nNext view index with STO (stop trigger on) under Stop-Loss is", np_t_stop_ind,
                          ". STO is using field", np_sto,
                          "\nFrom key: buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5\n")

                    print("If -1 returned there is no stop found to end of view, then next two array lines are garbage")
                    print("Row", np_t_stop_ind, np_t_open_v[np_t_stop_ind])
                    print("Row", np_t_stop_ind + 1, np_t_open_v[np_t_stop_ind + 1])
                if debug_timing:
                    t_t = f(st)
                    print("3-numpy", str.format('{0:.17f}', t_t))
                    st = s()

                '''
                4 - Find first sell index after trade open
    
                First index in the view np_t_open_v where ['sell'] = 1
    
                Requires: np_t_open_v - view of ticker_data from buy onwards
                Requires: no_sell - integer '3', the buy column in the array
                Provides: np_t_sell_ind index of view where first sell=1 after buy
                '''
                # Use numpy array for faster search for sell
                # Sell uses column 3.
                # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
                # Numpy searches 25-35x quicker than pandas on this data

                np_t_sell_ind = utf1st.find_1st(np_t_open_v[:, np_sell],
                                                1, utf1st.cmp_equal)
                if debug:
                    print("\n(4) numpy debug\nNext view index with sell = 1 is ", np_t_sell_ind)
                    print("If 0 or less is returned there is no sell found to end of view, then next lines garbage")
                    print("Row", np_t_sell_ind, np_t_open_v[np_t_sell_ind])
                    print("Row", np_t_sell_ind + 1, np_t_open_v[np_t_sell_ind + 1])
                if debug_timing:
                    t_t = f(st)
                    print("4-numpy", str.format('{0:.17f}', t_t))
                    st = s()

                '''
                5 - Determine which was hit first a stop or sell
                To then use as exit index price-field (sell on buy, stop on stop)
    
                STOP takes priority over SELL as would be 'in candle' from tick data
                Sell would use Open from Next candle.
                So in a draw Stop would be hit first on ticker data in live
    
                Validity of when types of trades may be executed can be summarised as:
    
                            Tick	View
                            index	index	Buy Sell open	low	     close	 high   Stop price
                open 2am	94	        -1	0	 0	-----	------	 ------  -----   -----
                open 3am 	95	        0	1	 0	-----	------	 trg buy -----   -----
                open 4am 	96	        1	0	 1	Enter	trgstop	 trg sel ROI out Stop out
                open 5am 	97	        2	0	 0	Exit	------	 ------- -----   -----
                open 6am	98	        3	0	 0	----- 	------	 ------- -----   -----
    
                -1 means not found till end of view i.e no valid Stop found. Exclude from match.
                Stop tiggering in 1, candle we bought at OPEN is valid.
    
                Buys and sells are triggered at candle close
                Both with action their postions at the open of the next candle Index + 1
    
                Stop and buy Indexes are on the view. To map to the ticker dataframe
                the t_open_ind index should be summed.
    
                np_t_stop_ind: Stop Found index in view
                t_exit_ind   : Sell found in view
                t_open_ind   : Where view was started on ticker_data
    
                TODO: fix this frig for logig test,, case/switch/dictionary would be better...
                      more so when later testing many options, dynamic stop / roi etc
                cludge - Im setting np_t_sell_ind as 9999999999 when -1 (not found)
                cludge - Im setting np_t_stop_ind as 9999999999 when -1 (not found)
    
                '''
                if debug:
                    print("\n(5) numpy debug\nStop or Sell Logic Processing")

                # cludge for logic test (-1) means it was not found, set crazy high to lose < test
                np_t_sell_ind = 99999999 if np_t_sell_ind <= 0 else np_t_sell_ind
                np_t_stop_ind = 99999999 if np_t_stop_ind == -1 else np_t_stop_ind

                # Stoploss trigger found before a sell =1
                if np_t_stop_ind < 99999999 and np_t_stop_ind <= np_t_sell_ind:
                    t_exit_ind = t_open_ind + np_t_stop_ind  # Set Exit row index
                    t_exit_type = 'stop'  # Set Exit type (stop)
                    np_t_exit_pri = np_sco  # The price field our STOP exit will use
                    if debug:
                        print("Type STOP is first exit condition. "
                              "At view index:", np_t_stop_ind, ". Ticker data exit index is", t_exit_ind)

                # Buy = 1 found before a stoploss triggered
                elif np_t_sell_ind < 99999999 and np_t_sell_ind < np_t_stop_ind:
                    # move sell onto next candle, we only look back on sell
                    # will use the open price later.
                    t_exit_ind = t_open_ind + np_t_sell_ind + 1  # Set Exit row index
                    t_exit_type = 'sell'  # Set Exit type (sell)
                    np_t_exit_pri = np_open  # The price field our SELL exit will use
                    if debug:
                        print("Type SELL is first exit condition. "
                              "At view index", np_t_sell_ind, ". Ticker data exit index is", t_exit_ind)

                # No stop or buy left in view - set t_exit_last -1 to handle gracefully
                else:
                    t_exit_last: int = -1  # Signal loop to exit, no buys or sells found.
                    t_exit_type = "No Exit"
                    np_t_exit_pri = 999  # field price should be calculated on. 999 a non-existent column
                    if debug:
                        print("No valid STOP or SELL found. Signalling t_exit_last to gracefully exit")

                # TODO: fix having to cludge/uncludge this ..
                # Undo cludge
                np_t_sell_ind = -1 if np_t_sell_ind == 99999999 else np_t_sell_ind
                np_t_stop_ind = -1 if np_t_stop_ind == 99999999 else np_t_stop_ind

                if debug_timing:
                    t_t = f(st)
                    print("5-logic", str.format('{0:.17f}', t_t))
                    st = s()

                if debug:
                    '''
                    Print out the buys, stops, sells
                    Include Line before and after to for easy
                    Human verification
                    '''
                    # Combine the np_t_stop_pri value to bslap dataframe to make debug
                    # life easy. This is the current stop price based on buy price_
                    # This is slow but don't care about performance in debug
                    #
                    # When referencing equiv np_column, as example np_sto, its 5 in numpy and 6 in df, so +1
                    # as there is no data column in the numpy array.
                    bslap['np_stop_pri'] = np_t_stop_pri

                    # Buy
                    print("\n\nDATAFRAME DEBUG =================== BUY ", pair)
                    print("Numpy Array BUY Index is:", 0)
                    print("DataFrame BUY Index is:", t_open_ind, "displaying DF \n")
                    print("HINT, BUY trade should use OPEN price from next candle, i.e ", t_open_ind + 1)
                    op_is = t_open_ind - 1  # Print open index start, line before
                    op_if = t_open_ind + 3  # Print open index finish, line after
                    print(bslap.iloc[op_is:op_if], "\n")

                    # Stop - Stops trigger price np_sto (+1 for pandas column), and price received np_sco +1. (Stop Trigger|Calculated On)
                    if np_t_stop_ind < 0:
                        print("DATAFRAME DEBUG =================== STOP  ", pair)
                        print("No STOPS were found until the end of ticker data file\n")
                    else:
                        print("DATAFRAME DEBUG =================== STOP  ", pair)
                        print("Numpy Array STOP Index is:", np_t_stop_ind, "View starts at index", t_open_ind)
                        df_stop_index = (t_open_ind + np_t_stop_ind)

                        print("DataFrame STOP Index is:", df_stop_index, "displaying DF \n")
                        print("First Stoploss trigger after Trade entered at OPEN in candle", t_open_ind + 1, "is ",
                              df_stop_index, ": \n",
                              str.format('{0:.17f}', bslap.iloc[df_stop_index][np_sto + 1]),
                              "is less than", str.format('{0:.17f}', np_t_stop_pri))

                        print("A stoploss exit will be calculated at rate:",
                              str.format('{0:.17f}', bslap.iloc[df_stop_index][np_sco + 1]))

                        print("\nHINT, STOPs should exit in-candle, i.e", df_stop_index,
                              ": As live STOPs are not linked to O-C times")

                        st_is = df_stop_index - 1  # Print stop index start, line before
                        st_if = df_stop_index + 2  # Print stop index finish, line after
                        print(bslap.iloc[st_is:st_if], "\n")

                    # Sell
                    if np_t_sell_ind < 0:
                        print("DATAFRAME DEBUG =================== SELL ", pair)
                        print("No SELLS were found till the end of ticker data file\n")
                    else:
                        print("DATAFRAME DEBUG =================== SELL ", pair)
                        print("Numpy View SELL Index is:", np_t_sell_ind, "View starts at index", t_open_ind)
                        df_sell_index = (t_open_ind + np_t_sell_ind)

                        print("DataFrame SELL Index is:", df_sell_index, "displaying DF \n")
                        print("First Sell Index after Trade open is in candle", df_sell_index)
                        print("HINT, if exit is SELL (not stop) trade should use OPEN price from next candle",
                              df_sell_index + 1)
                        sl_is = df_sell_index - 1  # Print sell index start, line before
                        sl_if = df_sell_index + 3  # Print sell index finish, line after
                        print(bslap.iloc[sl_is:sl_if], "\n")

                    # Chosen Exit (stop or sell)

                    print("DATAFRAME DEBUG =================== EXIT ", pair)
                    print("Exit type is :", t_exit_type)
                    print("trade exit price field is", np_t_exit_pri, "\n")

                if debug_timing:
                    t_t = f(st)
                    print("6-depra", str.format('{0:.17f}', t_t))
                    st = s()

                ## use numpy view "np_t_open_v" for speed. Columns are
                # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
                # exception is 6 which is use the stop value.

                # TODO no! this is hard coded bleh fix this open
                np_trade_enter_price = np_bslap[t_open_ind + 1, np_open]
                if t_exit_type == 'stop':
                    if np_t_exit_pri == 6:
                        np_trade_exit_price = np_t_stop_pri
                    else:
                        np_trade_exit_price = np_bslap[t_exit_ind, np_t_exit_pri]
                if t_exit_type == 'sell':
                    np_trade_exit_price = np_bslap[t_exit_ind, np_t_exit_pri]

                # Catch no exit found
                if t_exit_type == "No Exit":
                    np_trade_exit_price = 0

                if debug_timing:
                    t_t = f(st)
                    print("7-numpy", str.format('{0:.17f}', t_t))
                    st = s()

                if debug:
                    print("//////////////////////////////////////////////")
                    print("+++++++++++++++++++++++++++++++++ Trade Enter ")
                    print("np_trade Enter Price is ", str.format('{0:.17f}', np_trade_enter_price))
                    print("--------------------------------- Trade Exit ")
                    print("Trade Exit Type is ", t_exit_type)
                    print("np_trade Exit Price is", str.format('{0:.17f}', np_trade_exit_price))
                    print("//////////////////////////////////////////////")

            else:  # no buys were found, step 0 returned -1
                # Gracefully exit the loop
                t_exit_last == -1
                if debug:
                    print("\n(E) No buys were found in remaining ticker file. Exiting", pair)

            # Loop control - catch no closed trades.
            if debug:
                print("---------------------------------------- end of loop", loop,
                      " Dataframe Exit Index is: ", t_exit_ind)
                print("Exit Index Last, Exit Index Now Are: ", t_exit_last, t_exit_ind)

            if t_exit_last >= t_exit_ind or t_exit_last == -1:
                """
                Break loop and go on to next pair.
    
                When last trade exit equals index of last exit, there is no
                opportunity to close any more trades.
                """
                # TODO :add handing here to record none closed open trades
                if debug:
                    print(bslap_pair_results)
                break
            else:
                """
                Add trade to backtest looking results list of dicts
                Loop back to look for more trades.
                """
                # Build trade dictionary
                ## In general if a field can be calculated later from other fields leave blank here
                ## Its X(number of trades faster) to calc all in a single vector than 1 trade at a time

                # create a new dict
                close_index: int = t_exit_ind
                bslap_result = {}  # Must have at start or we end up with a list of multiple same last result
                bslap_result["pair"] = pair
                bslap_result["profit_percent"] = ""  # To be 1 vector calc across trades when loop complete
                bslap_result["profit_abs"] = ""  # To be 1 vector calc across trades when loop complete
                bslap_result["open_time"] = np_bslap_dates[t_open_ind + 1]  # use numpy array, pandas 20x slower
                bslap_result["close_time"] = np_bslap_dates[close_index]  # use numpy array, pandas 20x slower
                bslap_result["open_index"] = t_open_ind + 1  # +1 as we buy on next.
                bslap_result["close_index"] = close_index
                bslap_result["trade_duration"] = ""  # To be 1 vector calc across trades when loop complete
                bslap_result["open_at_end"] = False
                bslap_result["open_rate"] = round(np_trade_enter_price, 15)
                bslap_result["close_rate"] = round(np_trade_exit_price, 15)
                bslap_result["exit_type"] = t_exit_type
                # append the dict to the list and print list
                bslap_pair_results.append(bslap_result)

                if debug:
                    print("The trade dict is: \n", bslap_result)
                    print("Trades dicts in list after append are: \n ", bslap_pair_results)

                """
                Loop back to start. t_exit_last becomes where loop
                will seek to open new trades from.
                Push index on 1 to not open on close
                """
                t_exit_last = t_exit_ind + 1

            if debug_timing:
                t_t = f(st)
                print("8+trade", str.format('{0:.17f}', t_t))

        # Send back List of trade dicts
        return bslap_pair_results

    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data = {}
        pairs = self.config['exchange']['pair_whitelist']
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        if self.config.get('live'):
            logger.info('Downloading data for all pairs in whitelist ...')
            for pair in pairs:
                data[pair] = self.exchange.get_ticker_history(pair, self.ticker_interval)
        else:
            logger.info('Using local backtesting data (using whitelist in given config) ...')

            timerange = Arguments.parse_timerange(None if self.config.get(
                'timerange') is None else str(self.config.get('timerange')))
            data = optimize.load_data(
                self.config['datadir'],
                pairs=pairs,
                ticker_interval=self.ticker_interval,
                refresh_pairs=self.config.get('refresh_pairs', False),
                exchange=self.exchange,
                timerange=timerange
            )

        if not data:
            logger.critical("No data found. Terminating.")
            return
        # Ignore max_open_trades in backtesting, except realistic flag was passed
        if self.config.get('realistic_simulation', False):
            max_open_trades = self.config['max_open_trades']
        else:
            logger.info('Ignoring max_open_trades (realistic_simulation not set) ...')
            max_open_trades = 0

        preprocessed = self.tickerdata_to_dataframe(data)

        # Print timeframe
        min_date, max_date = self.get_timeframe(preprocessed)
        logger.info(
            'Measuring data from %s up to %s (%s days)..',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )

        # Execute backtest and print results
        results = self.backtest(
            {
                'stake_amount': self.config.get('stake_amount'),
                'processed': preprocessed,
                'max_open_trades': max_open_trades,
                'realistic': self.config.get('realistic_simulation', False),
            }
        )

        if self.config.get('export', False):
            self._store_backtest_result(self.config.get('exportfilename'), results)

        if self.use_backslap:
            logger.info(
                '\n====================================================== '
                'BackSLAP REPORT'
                ' =======================================================\n'
                '%s',
                self._generate_text_table(
                    data,
                    results
                )
            )
            #optional print trades
            if self.backslap_show_trades:
                TradesFrame = results.filter(['open_time', 'pair', 'exit_type', 'profit_percent', 'profit_abs',
                                              'buy_spend', 'sell_take', 'trade_duration', 'close_time'], axis=1)
                def to_fwf(df, fname):
                    content = tabulate(df.values.tolist(), list(df.columns), floatfmt=".8f", tablefmt='psql')
                    print(content)

                DataFrame.to_fwf = to_fwf(TradesFrame, "backslap.txt")

            #optional save trades
            if self.backslap_save_trades:
                TradesFrame = results.filter(['open_time', 'pair', 'exit_type', 'profit_percent', 'profit_abs',
                                              'buy_spend', 'sell_take', 'trade_duration', 'close_time'], axis=1)
                def to_fwf(df, fname):
                    content = tabulate(df.values.tolist(), list(df.columns), floatfmt=".8f", tablefmt='psql')
                    open(fname, "w").write(content)
                DataFrame.to_fwf = to_fwf(TradesFrame, "backslap.txt")

        else:
            logger.info(
                '\n================================================= '
                'BACKTEST REPORT'
                ' ==================================================\n'
                '%s',
                self._generate_text_table(
                    data,
                    results
                )
            )

        ## TODO. Catch open trades for this report.
        # logger.info(
        #     '\n=============================================== '
        #     'LEFT OPEN TRADES REPORT'
        #     ' ===============================================\n'
        #     '%s',
        #     self._generate_text_table(
        #         data,
        #         results.loc[results.open_at_end]
        #     )
        # )


def setup_configuration(args: Namespace) -> Dict[str, Any]:
    """
    Prepare the configuration for the backtesting
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args)
    config = configuration.get_config()

    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    if config['stake_amount'] == constants.UNLIMITED_STAKE_AMOUNT:
        raise DependencyException('stake amount could not be "%s" for backtesting' %
                                  constants.UNLIMITED_STAKE_AMOUNT)

    return config


def start(args: Namespace) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Initialize configuration
    config = setup_configuration(args)
    logger.info('Starting freqtrade in Backtesting mode')

    # Initialize backtesting object
    backtesting = Backtesting(config)
    backtesting.start()
