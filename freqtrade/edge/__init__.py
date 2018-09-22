import logging
import operator
import sys
from argparse import Namespace
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import arrow
from pandas import DataFrame, to_datetime
from tabulate import tabulate
import numpy as np

import freqtrade.optimize as optimize
from freqtrade import DependencyException, constants
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Exchange
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.resolver import IStrategy, StrategyResolver
from freqtrade.optimize.backtesting import Backtesting
from collections import OrderedDict
import timeit
from time import sleep

import pdb

logger = logging.getLogger(__name__)

class Edge():

    config: Dict = {}

    def __init__(self, config: Dict[str, Any], exchange = None) -> None:
        """
        constructor
        """
        self.config = config
        self.strategy: IStrategy = StrategyResolver(self.config).strategy
        self.ticker_interval = self.strategy.ticker_interval
        self.tickerdata_to_dataframe = self.strategy.tickerdata_to_dataframe
        self.get_timeframe = Backtesting.get_timeframe
        self.populate_buy_trend = self.strategy.populate_buy_trend
        self.populate_sell_trend = self.strategy.populate_sell_trend
        
        self.edge_config = self.config.get('edge', {})

        self._last_updated = None
        self._cached_pairs = []
        self._total_capital = self.edge_config['total_capital_in_stake_currency']
        self._allowed_risk = self.edge_config['allowed_risk']

        ###
        #
        ###
        if exchange is None:
            self.config['exchange']['secret'] = ''
            self.config['exchange']['password'] = ''
            self.config['exchange']['uid'] = ''
            self.config['dry_run'] = True
            self.exchange = Exchange(self.config)
        else:
            self.exchange = exchange

        self.fee = self.exchange.get_fee()

        self.stop_loss_value = self.strategy.stoploss

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
        # self.np_sto: int = self.np_close  # stops_triggered_on - Should be low, FT uses close
        # self.np_sco: int = self.np_close  # stops_calculated_on - Should be stop, FT uses close

        self.debug = False  # Main debug enable, very print heavy, enable 2 loops recommended
        self.debug_timing = False  # Stages within Backslap
        self.debug_2loops = False  # Limit each pair to two loops, useful when debugging
        self.debug_vector = False  # Debug vector calcs
        self.debug_timing_main_loop = False  # print overall timing per pair - works in Backtest and Backslap

        self.backslap_show_trades = False  # prints trades in addition to summary report
        self.backslap_save_trades = True  # saves trades as a pretty table to backslap.txt

        self.stop_stops: int = 9999  # stop back testing any pair with this many stops, set to 999999 to not hit


    def calculate(self) -> bool:
        pairs = self.config['exchange']['pair_whitelist']
        heartbeat = self.config['edge']['process_throttle_secs']

        if ((self._last_updated is not None) and (self._last_updated + heartbeat > arrow.utcnow().timestamp)):
            return False

        data = {}

        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])
 
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
            logger.critical("No data found. Edge is stopped ...")
            return
        
        preprocessed = self.tickerdata_to_dataframe(data)

        # Print timeframe
        min_date, max_date = self.get_timeframe(preprocessed)
        logger.info(
            'Measuring data from %s up to %s (%s days) ...',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )


        headers = ['date', 'buy', 'open', 'close', 'sell', 'high', 'low']

        # Max open trades need not be considered in Edge positioning
        max_open_trades = 0

        realistic = False
        stoploss_range_min = float(self.edge_config.get('stoploss_range_min', -0.01))
        stoploss_range_max = float(self.edge_config.get('stoploss_range_max', -0.05))
        stoploss_range_step = float(self.edge_config.get('stoploss_range_step', -0.001))

        stoploss_range = np.arange(stoploss_range_min, stoploss_range_max, stoploss_range_step)
        trades = []
        trade_count_lock: Dict = {}

        ########################### Call out BSlap Loop instead of Original BT code
        bslap_results: list = []
        for pair, pair_data in preprocessed.items():
            
            # Sorting dataframe by date and reset index 
            pair_data = pair_data.sort_values(by=['date'])
            pair_data = pair_data.reset_index(drop=True)

            ticker_data = self.populate_sell_trend(
                self.populate_buy_trend(pair_data))[headers].copy()

            # call backslap - results are a list of dicts
            for stoploss in stoploss_range:
                bslap_results += self.backslap_pair(ticker_data, pair, round(stoploss, 6))

        # Switch List of Trade Dicts (bslap_results) to Dataframe
        # Fill missing, calculable columns, profit, duration , abs etc.
        bslap_results_df = DataFrame(bslap_results)

        if len(bslap_results_df) > 0:  # Only post process a frame if it has a record
            bslap_results_df = self.vector_fill_results_table(bslap_results_df)
        else:
            from freqtrade.optimize.backtesting import BacktestResult

            bslap_results_df = []
            bslap_results_df = DataFrame.from_records(bslap_results_df, columns=BacktestResult._fields)
 
        self._cached_pairs = self._process_result(data, bslap_results_df, stoploss_range)
        self._last_updated = arrow.utcnow().timestamp
        return True
        
    def sort_pairs(self, pairs) -> bool:
        if len(self._cached_pairs) == 0:
            self.calculate()
        edge_sorted_pairs = [x[0] for x in self._cached_pairs]
        return [x for _, x in sorted(zip(edge_sorted_pairs,pairs), key=lambda pair: pair[0])]
        
    
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
        import numpy as np
        debug = self.debug_vector

        # stake and fees
        # stake = 0.015
        # 0.05% is 0.0005
        # fee = 0.001

        stake = self.config.get('stake_amount')
        fee = self.fee
        open_fee = fee / 2
        close_fee = fee / 2


        bslap_results_df['trade_duration'] = bslap_results_df['close_time'] - bslap_results_df['open_time']
        bslap_results_df['trade_duration'] = bslap_results_df['trade_duration'].map(lambda x: int(x.total_seconds() / 60))

        ## Spends, Takes, Profit, Absolute Profit
        # print(bslap_results_df)
        # Buy Price
        bslap_results_df['buy_vol'] = stake / bslap_results_df['open_rate']  # How many target are we buying
        bslap_results_df['buy_fee'] = stake * open_fee
        bslap_results_df['buy_spend'] = stake + bslap_results_df['buy_fee']  # How much we're spending

        # Sell price
        bslap_results_df['sell_sum'] = bslap_results_df['buy_vol'] * bslap_results_df['close_rate']
        bslap_results_df['sell_fee'] = bslap_results_df['sell_sum'] * close_fee
        bslap_results_df['sell_take'] = bslap_results_df['sell_sum'] - bslap_results_df['sell_fee']
        # profit_percent
        bslap_results_df['profit_percent'] = (bslap_results_df['sell_take'] - bslap_results_df['buy_spend']) \
                                             / bslap_results_df['buy_spend']
        # Absolute profit
        bslap_results_df['profit_abs'] = bslap_results_df['sell_take'] - bslap_results_df['buy_spend']


        return bslap_results_df

    def np_get_t_open_ind(self, np_buy_arr, t_exit_ind: int, np_buy_arr_len: int, stop_stops: int,
                          stop_stops_count: int):
        import utils_find_1st as utf1st
        """
         The purpose of this def is to return the next "buy" = 1
         after t_exit_ind.
         
         This function will also check is the stop limit for the pair has been reached. 
         if stop_stops is the limit and stop_stops_count it the number of times the stop has been hit.

         t_exit_ind is the index the last trade exited on
         or 0 if first time around this loop.
         
         stop_stops i
         """
        debug = self.debug

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

        if t_open_ind == np_buy_arr_len - 1:  # If buy found on last candle ignore, there is no OPEN in next to use
            t_open_ind = -1  # -1 ends the loop

        if stop_stops_count >= stop_stops:  # if maximum number of stops allowed in a pair is hit, exit loop
            t_open_ind = -1  # -1 ends the loop
            if debug:
                print("Max stop limit ", stop_stops, "reached. Moving to next pair")

        return t_open_ind

    def _process_result(self, data: Dict[str, Dict], results: DataFrame, stoploss_range) -> str:
        """
        This is a temporary version of edge positioning calculation. 
        The function will be eventually moved to a plugin called Edge in order to calculate necessary WR, RRR and 
        other indictaors related to money management periodically (each X minutes) and keep it in a storage. 
        The calulation will be done per pair and per strategy. 
        """

        # Removing open trades from dataset
        results = results[results.open_at_end == False]
        ###################################

        # Removing pairs having less than min_trades_number
        min_trades_number = self.edge_config.get('min_trade_number', 15)
        results = results.groupby('pair').filter(lambda x: len(x) > min_trades_number)
        ###################################


        # Removing outliers (Only Pumps) from the dataset 
        # The method to detect outliers is to calculate standard deviation
        # Then every value more than (standard deviation + 2*average) is out (pump)
        #
        # Calculating standard deviation of profits
        std = results[["profit_abs"]].std()
        #
        # Calculating average of profits
        avg = results[["profit_abs"]].mean()
        #
        # Removing Pumps
        if self.edge_config.get('remove_pumps', True):
            results = results[results.profit_abs < float(avg + 2*std)]
        ##########################################################################

        # Removing trades having a duration more than X minutes (set in config)
        max_trade_duration = self.edge_config.get('max_trade_duration_minute', 1440)
        results = results[results.trade_duration < max_trade_duration]
        #######################################################################


        # Win Rate is the number of profitable trades
        # Divided by number of trades
        def winrate(x):
            x = x[x > 0].count() / x.count()
            return x
        #############################

        # Risk Reward Ratio
        # 1 / ((loss money / losing trades) / (gained money / winning trades))
        def risk_reward_ratio(x):
            x = abs(1/ ((x[x<0].sum() / x[x < 0].count()) / (x[x > 0].sum() / x[x > 0].count())))
            return x
        ##############################

        # Required Risk Reward
        # (1/(winrate - 1)
        def required_risk_reward(x):
            x = (1/(x[x > 0].count()/x.count()) -1)
            return x
        ##############################
        
        # The difference between risk reward ratio and required risk reward
        # We use it as an indicator to find the most interesting pair to trade
        def delta(x):
            x = (abs(1/ ((x[x < 0].sum() / x[x < 0].count()) / (x[x > 0].sum() / x[x > 0].count())))) - (1/(x[x > 0].count()/x.count()) -1)
            return x
        ##############################

        
        final = results.groupby(['pair', 'stoploss'])['profit_abs'].\
            agg([winrate, risk_reward_ratio, required_risk_reward, delta]).\
            reset_index().sort_values(by=['delta', 'stoploss'], ascending=False)\
            .groupby('pair').first().sort_values(by=['delta'], ascending=False)
        
        # Returning an array of pairs in order of "delta"
        return final.reset_index().values

    def backslap_pair(self, ticker_data, pair, stoploss):
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
        # pdb.set_trace()
        #stop = self.stop_loss_value
        stop = stoploss
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

        # ticker_data: DataFrame = ticker_dfs[t_file]
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

        stop_stops = self.stop_stops  # Int of stops within a pair to stop trading a pair at
        stop_stops_count = 0  # stop counter per pair

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
                    print(
                        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Loop debug max met - breaking")
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
            Requires: np_buy_arr_len - length of pair array. 
            Requires: stops_stops - number of stops allowed before stop trading a pair 
            Requires: stop_stop_counts - count of stops hit in the pair
            Provides: The next "buy" index after t_exit_ind
    
            If -1 is returned no buy has been found in remainder of array, skip to exit loop
            '''
            t_open_ind = self.np_get_t_open_ind(np_buy_arr, t_exit_ind, np_buy_arr_len, stop_stops, stop_stops_count)

            if debug:
                print("\n(0) numpy debug \nnp_get_t_open, has returned the next valid buy index as", t_open_ind)
                print("If -1 there are no valid buys in the remainder of ticker data. Skipping to end of loop")
            if debug_timing:
                t_t = f(st)
                print("0-numpy", str.format('{0:.17f}', t_t))
                st = s()

            if t_open_ind != -1:

                """
                1 - Create views to search within for our open trade
    
                The views are our search space for the next Stop or Sell
                Numpy view is employed as:
                1,000 faster than pandas searches
                Pandas cannot assure it will always return a view, it may make a slow copy.
    
                The view contains columns:
                buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
    
                Requires: np_bslap is our numpy array of the ticker DataFrame
                Requires: t_open_ind is the index row with the  buy.
                Provides: np_t_open_v View of array after buy.
                Provides: np_t_open_v_stop View of array after buy +1  
                          (Stop will search in here to prevent stopping in the past)
                """
                np_t_open_v = np_bslap[t_open_ind:]
                np_t_open_v_stop = np_bslap[t_open_ind + 1:]

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
    
                Requires: np_t_open_v_stop   Numpy view of ticker_data after buy row +1 (when trade was opened)
                Requires: np_sto        User Var(STO)StopTriggeredOn. Typically set to "low" or "close"
                Requires: np_t_stop_pri The stop-loss price STO must fall under to trigger stop
                Provides: np_t_stop_ind The first candle after trade open where STO is under stop-loss
                '''
                np_t_stop_ind = utf1st.find_1st(np_t_open_v_stop[:, np_sto],
                                                np_t_stop_pri,
                                                utf1st.cmp_smaller)

                # plus 1 as np_t_open_v_stop is 1 ahead of view np_t_open_v, used from here on out.
                np_t_stop_ind = np_t_stop_ind + 1

                if debug:
                    print("\n(3) numpy debug\nNext view index with STO (stop trigger on) under Stop-Loss is",
                          np_t_stop_ind - 1,
                          ". STO is using field", np_sto,
                          "\nFrom key: buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5\n")

                    print(
                        "If -1 or 0 returned there is no stop found to end of view, then next two array lines are garbage")
                    print("Row", np_t_stop_ind - 1, np_t_open_v[np_t_stop_ind])
                    print("Row", np_t_stop_ind, np_t_open_v[np_t_stop_ind + 1])
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
                Stop tiggering and closing in 96-1, the candle we bought at OPEN in, is valid.
    
                Buys and sells are triggered at candle close
                Both will open their postions at the open of the next candle. i/e  + 1 index
    
                Stop and buy Indexes are on the view. To map to the ticker dataframe
                the t_open_ind index should be summed.
    
                np_t_stop_ind: Stop Found index in view
                t_exit_ind   : Sell found in view
                t_open_ind   : Where view was started on ticker_data
    
                TODO: fix this frig for logic test,, case/switch/dictionary would be better...
                      more so when later testing many options, dynamic stop / roi etc
                cludge - Setting np_t_sell_ind as 9999999999 when -1 (not found)
                cludge - Setting np_t_stop_ind as 9999999999 when -1 (not found)
    
                '''
                if debug:
                    print("\n(5) numpy debug\nStop or Sell Logic Processing")

                # cludge for logic test (-1) means it was not found, set crazy high to lose < test
                np_t_sell_ind = 99999999 if np_t_sell_ind <= 0 else np_t_sell_ind
                np_t_stop_ind = 99999999 if np_t_stop_ind <= 0 else np_t_stop_ind

                # Stoploss trigger found before a sell =1
                if np_t_stop_ind < 99999999 and np_t_stop_ind <= np_t_sell_ind:
                    t_exit_ind = t_open_ind + np_t_stop_ind  # Set Exit row index
                    t_exit_type = SellType.STOP_LOSS  # Set Exit type (stop)
                    np_t_exit_pri = np_sco  # The price field our STOP exit will use
                    if debug:
                        print("Type STOP is first exit condition. "
                              "At view index:", np_t_stop_ind, ". Ticker data exit index is", t_exit_ind)

                # Buy = 1 found before a stoploss triggered
                elif np_t_sell_ind < 99999999 and np_t_sell_ind < np_t_stop_ind:
                    # move sell onto next candle, we only look back on sell
                    # will use the open price later.
                    t_exit_ind = t_open_ind + np_t_sell_ind # Set Exit row index
                    t_exit_type = SellType.SELL_SIGNAL  # Set Exit type (sell)
                    np_t_exit_pri = np_open  # The price field our SELL exit will use
                    if debug:
                        print("Type SELL is first exit condition. "
                              "At view index", np_t_sell_ind, ". Ticker data exit index is", t_exit_ind)

                # No stop or buy left in view - set t_exit_last -1 to handle gracefully
                else:
                    t_exit_last: int = -1  # Signal loop to exit, no buys or sells found.
                    t_exit_type = SellType.NONE
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
                if t_exit_type == SellType.STOP_LOSS:
                    if np_t_exit_pri == 6:
                        np_trade_exit_price = np_t_stop_pri
                        t_exit_ind = t_exit_ind + 1
                    else:
                        np_trade_exit_price = np_bslap[t_exit_ind, np_t_exit_pri]
                if t_exit_type == SellType.SELL_SIGNAL:
                    np_trade_exit_price = np_bslap[t_exit_ind, np_t_exit_pri]

                # Catch no exit found
                if t_exit_type == SellType.NONE:
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

                # We added +1 to t_exit_ind if the exit was a stop-loss, to not exit early in the IF of this ELSE
                # removing the +1 here so prices match.
                if t_exit_type == SellType.STOP_LOSS:
                    t_exit_ind = t_exit_ind - 1

                # Build trade dictionary
                ## In general if a field can be calculated later from other fields leave blank here
                ## Its X(number of trades faster) to calc all in a single vector than 1 trade at a time

                # create a new dict
                close_index: int = t_exit_ind
                bslap_result = {}  # Must have at start or we end up with a list of multiple same last result
                bslap_result["pair"] = pair
                bslap_result["stoploss"] = stop
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
                bslap_result["sell_reason"] = t_exit_type #duplicated, but I don't care
                # append the dict to the list and print list
                bslap_pair_results.append(bslap_result)

                if t_exit_type is SellType.STOP_LOSS:
                    stop_stops_count = stop_stops_count + 1

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

    def stake_amount(self, pair: str) -> str:
        info = [x for x in self._cached_pairs if x[0] == pair][0]
        stoploss = info[1]
        
        allowed_capital_at_risk = round(self._total_capital * self._allowed_risk, 5)
        position_size = abs(round((allowed_capital_at_risk / stoploss), 5))

        return (allowed_dollars_at_risk / symbol_strategy_stop_loss)

    
    def stoploss(self, pair: str) -> float:
        info = [x for x in self._cached_pairs if x[0] == pair][0]
        return info[1]