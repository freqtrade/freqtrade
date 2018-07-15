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
from pandas import DataFrame
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

    @profile
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
        headers = ['date', 'buy', 'open', 'close', 'sell', 'high', 'low']
        processed = args['processed']
        max_open_trades = args.get('max_open_trades', 0)
        realistic = args.get('realistic', False)
        trades = []
        trade_count_lock: Dict = {}
        ########################### Call out BSlap instead of using FT
        bslap_results: list = []
        last_bslap_results: list = []

        for pair, pair_data in processed.items():
            ticker_data = self.populate_sell_trend(
                    self.populate_buy_trend(pair_data))[headers].copy()

            ticker_data.drop(ticker_data.head(1).index, inplace=True)

            # #dump same DFs to disk for offline testing in scratch
            # f_pair:str = pair
            # csv = f_pair.replace("/", "_")
            # csv="/Users/creslin/PycharmProjects/freqtrade_new/frames/" + csv
            # ticker_data.to_csv(csv, sep='\t', encoding='utf-8')

            #call bslap - results are a list of dicts
            bslap_pair_results = self.backslap_pair(ticker_data, pair)
            last_bslap_results = bslap_results
            bslap_results = last_bslap_results + bslap_pair_results

        bslap_results_df = DataFrame(bslap_results, columns=BacktestResult._fields)
        return bslap_results_df

        ########################### Original BT loop
        # for pair, pair_data in processed.items():
        #     pair_data['buy'], pair_data['sell'] = 0, 0  # cleanup from previous run
        #
        #     ticker_data = self.populate_sell_trend(
        #         self.populate_buy_trend(pair_data))[headers].copy()
        #
        #     # to avoid using data from future, we buy/sell with signal from previous candle
        #     ticker_data.loc[:, 'buy'] = ticker_data['buy'].shift(1)
        #     ticker_data.loc[:, 'sell'] = ticker_data['sell'].shift(1)
        #
        #     ticker_data.drop(ticker_data.head(1).index, inplace=True)
        #
        #     # Convert from Pandas to list for performance reasons
        #     # (Looping Pandas is slow.)
        #     ticker = [x for x in ticker_data.itertuples()]
        #
        #     lock_pair_until = None
        #     for index, row in enumerate(ticker):
        #         if row.buy == 0 or row.sell == 1:
        #             continue  # skip rows where no buy signal or that would immediately sell off
        #
        #         if realistic:
        #             if lock_pair_until is not None and row.date <= lock_pair_until:
        #                 continue
        #         if max_open_trades > 0:
        #             # Check if max_open_trades has already been reached for the given date
        #             if not trade_count_lock.get(row.date, 0) < max_open_trades:
        #                 continue
        #
        #             trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1
        #
        #         trade_entry = self._get_sell_trade_entry(pair, row, ticker[index + 1:],
        #                                                  trade_count_lock, args)
        #
        #
        #         if trade_entry:
        #             lock_pair_until = trade_entry.close_time
        #             trades.append(trade_entry)
        #         else:
        #             # Set lock_pair_until to end of testing period if trade could not be closed
        #             # This happens only if the buy-signal was with the last candle
        #             lock_pair_until = ticker_data.iloc[-1].date
        #
        # return DataFrame.from_records(trades, columns=BacktestResult._fields)
        ######################## Original BT loop end

    def np_get_t_open_ind(self, np_buy_arr, t_exit_ind: int):
        import utils_find_1st as utf1st
        """
        The purpose of this def is to return the next "buy" = 1
        after t_exit_ind.

        t_exit_ind is the index the last trade exited on
        or 0 if first time around this loop.
        """
        t_open_ind: int

        # Create a view on our buy index starting after last trade exit
        # Search for next buy
        np_buy_arr_v = np_buy_arr[t_exit_ind:]
        t_open_ind = utf1st.find_1st(np_buy_arr_v, 1, utf1st.cmp_equal)
        t_open_ind = t_open_ind + t_exit_ind  # Align numpy index
        return t_open_ind

    def backslap_pair(self, ticker_data, pair):
        import pandas as pd
        import numpy as np
        import timeit
        import utils_find_1st as utf1st
        from datetime import datetime

        ### backslap debug wrap
        debug_2loops = False  # only loop twice, for faster debug
        debug_timing = False  # print timing for each step
        debug = False  # print values, to check accuracy

        # Read Stop Loss Values and Stake
        stop = self.stop_loss_value
        p_stop = (stop + 1)  # What stop really means, e.g 0.01 is 0.99 of price
        stake = self.config.get('stake_amount')

        # Set fees
        # TODO grab these from the environment, do not hard set
        # Fees
        open_fee = 0.05
        close_fee = 0.05

        if debug:
            print("Stop is ", stop, "value from stragey file")
            print("p_stop is", p_stop, "value used to multiply to entry price")
            print("Stake is,", stake, "the sum of currency to spend per trade")
            print("The open fee is", open_fee, "The close fee is", close_fee)

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
        """
        A couple legacy Pandas vars still used for pretty debug output.
        If have debug enabled the code uses these fields for dataframe output

        Ensure bto, sto, sco are aligned with Numpy values next
        to align debug and actual. Options are:
        buy - open  - close  - sell  - high  - low  - np_stop_pri
        """
        bto = buys_triggered_on = "close"
        # sto = stops_triggered_on = "low"  ## Should be low, FT uses close
        # sco = stops_calculated_on = "np_stop_pri"  ## should use np_stop_pri, FT uses close
        sto = stops_triggered_on = "close"  ## Should be low, FT uses close
        sco = stops_calculated_on = "close"  ## should use np_stop_pri, FT uses close
        '''
        Numpy arrays are used for 100x speed up
        We requires setting Int values for
        buy stop triggers and stop calculated on
        # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5 - stop 6
        '''
        np_buy: int = 0
        np_open: int = 1
        np_close: int = 2
        np_sell: int = 3
        np_high: int = 4
        np_low: int = 5
        np_stop: int = 6
        np_bto: int = np_close  # buys_triggered_on - should be close
        np_bco: int = np_open  # buys calculated on - open of the next candle.
        #np_sto: int = np_low  # stops_triggered_on - Should be low, FT uses close
        #np_sco: int = np_stop  # stops_calculated_on - Should be stop, FT uses close
        np_sto: int = np_close  # stops_triggered_on - Should be low, FT uses close
        np_sco: int = np_close  # stops_calculated_on - Should be stop, FT uses close
        #
        ### End Config

        pair: str = pair
        loop: int = 1

        #ticker_data: DataFrame = ticker_dfs[t_file]
        bslap: DataFrame = ticker_data

        # Build a single dimension numpy array from "buy" index for faster search
        # (500x faster than pandas)
        np_buy_arr = bslap['buy'].values
        np_buy_arr_len: int = len(np_buy_arr)

        # use numpy array for faster searches in loop, 20x faster than pandas
        # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
        np_bslap = np.array(bslap[['buy', 'open', 'close', 'sell', 'high', 'low']])

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
              c) Chose first intance as trade exit

            Phase 2
             2) Manage dynamic Stop and ROI Exit
              a) Create trade slice from 1
              b) search within trade slice for dynamice stop hit
              c) search within trade slice for ROI hit
            '''

            '''
            Finds index for first buy = 1 flag, use .values numpy array for speed

            Create a slice, from first buy index onwards.
            Slice will be used to find exit conditions after trade open
            '''
            if debug_timing:
                st = s()

            t_open_ind = self.np_get_t_open_ind(np_buy_arr, t_exit_ind)

            if debug_timing:
                t_t = f(st)
                print("0-numpy", str.format('{0:.17f}', t_t))
                st = s()

            '''
            Calculate np_t_stop_pri (Trade Stop Price) based on the buy price

            As stop in based on buy price we are interested in buy
            - Buys are Triggered On np_bto, typically the CLOSE of candle
            - Buys are Calculated On np_bco, default is OPEN of the next candle.
            as we only see the CLOSE after it has happened.
            The assumption is we have bought at first available price, the OPEN
            '''
            np_t_stop_pri = (np_bslap[t_open_ind + 1, np_bco] * p_stop)

            if debug_timing:
                t_t = f(st)
                print("1-numpy", str.format('{0:.17f}', t_t))
                st = s()

            """
            1)Create a View from our open trade  forward

            The view is our search space for the next Stop or Sell
            We use a numpy view:
            Using a numpy for speed on views, 1,000 faster than pandas
            Pandas cannot assure it will always return a view, copies are
            3 orders of magnitude slower

            The view contains columns:
            buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
            """
            np_t_open_v = np_bslap[t_open_ind:]

            if debug_timing:
                t_t = f(st)
                print("2-numpy", str.format('{0:.17f}', t_t))
                st = s()

            '''
            Find first stop index after Trade Open:

            First index in np_t_open_v (numpy view of bslap dataframe)
            Using a numpy view a orders of magnitude faster

            where [np_sto] (stop tiggered on variable: "close", "low" etc) < np_t_stop_pri
            '''
            np_t_stop_ind = utf1st.find_1st(np_t_open_v[:, np_sto],
                                            np_t_stop_pri,
                                            utf1st.cmp_smaller) \
                            + t_open_ind

            if debug_timing:
                t_t = f(st)
                print("3-numpy", str.format('{0:.17f}', t_t))
                st = s()

            '''
            Find first sell index after trade open

            First index in t_open_slice where ['sell'] = 1
            '''
            # Use numpy array for faster search for sell
            # Sell uses column 3.
            # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
            # Numpy searches 25-35x quicker than pandas on this data

            np_t_sell_ind = utf1st.find_1st(np_t_open_v[:, np_sell],
                                            1, utf1st.cmp_equal) \
                            + t_open_ind

            if debug_timing:
                t_t = f(st)
                print("4-numpy", str.format('{0:.17f}', t_t))
                st = s()

            '''
            Determine which was hit first stop or sell, use as exit

            STOP takes priority over SELL as would be 'in candle' from tick data
            Sell would use Open from Next candle.
            So in a draw Stop would be hit first on ticker data in live
            '''
            if np_t_stop_ind <= np_t_sell_ind:
                t_exit_ind = np_t_stop_ind  # Set Exit row index
                t_exit_type = 'stop'  # Set Exit type (sell|stop)
                np_t_exit_pri = np_sco  # The price field our STOP exit will use
            else:
                # move sell onto next candle, we only look back on sell
                # will use the open price later.
                t_exit_ind = np_t_sell_ind  # Set Exit row index
                t_exit_type = 'sell'  # Set Exit type (sell|stop)
                np_t_exit_pri = np_open  # The price field our SELL exit will use

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
                # life easy. This is the currenct stop price based on buy price_
                # Don't care about performance in debug
                # (add an extra column if printing as df has date in col1 not in npy)
                bslap['np_stop_pri'] = np_t_stop_pri

                # Buy
                print("=================== BUY ", pair)
                print("Numpy Array BUY Index is:", t_open_ind)
                print("DataFrame BUY Index is:", t_open_ind + 1, "displaying DF \n")
                print("HINT, BUY trade should use OPEN price from next candle, i.e ", t_open_ind + 2, "\n")
                op_is = t_open_ind - 1  # Print open index start, line before
                op_if = t_open_ind + 3  # Print open index finish, line after
                print(bslap.iloc[op_is:op_if], "\n")
                print(bslap.iloc[t_open_ind + 1]['date'])

                # Stop - Stops trigger price sto, and price received sco. (Stop Trigger|Calculated On)
                print("=================== STOP  ", pair)
                print("Numpy Array STOP Index is:", np_t_stop_ind)
                print("DataFrame STOP Index is:", np_t_stop_ind + 1, "displaying DF \n")
                print("First Stop after Trade open in candle", t_open_ind + 1, "is ", np_t_stop_ind + 1,": \n",
                      str.format('{0:.17f}', bslap.iloc[np_t_stop_ind][sto]),
                      "is less than", str.format('{0:.17f}', np_t_stop_pri))
                print("If stop is first exit match sell rate is :", str.format('{0:.17f}', bslap.iloc[np_t_stop_ind][sco]))
                print("HINT, STOPs should close in-candle, i.e", np_t_stop_ind + 1,
                      ": As live STOPs are not linked to O-C times")

                st_is = np_t_stop_ind - 1  # Print stop index start, line before
                st_if = np_t_stop_ind + 2  # Print stop index finish, line after
                print(bslap.iloc[st_is:st_if], "\n")

                # Sell
                print("=================== SELL ", pair)
                print("Numpy Array SELL Index is:", np_t_sell_ind)
                print("DataFrame SELL Index is:", np_t_sell_ind + 1, "displaying DF \n")
                print("First Sell Index after Trade open in in candle", np_t_sell_ind + 1)
                print("HINT, if exit is SELL (not stop) trade should use OPEN price from next candle",
                      np_t_sell_ind + 2, "\n")
                sl_is = np_t_sell_ind - 1  # Print sell index start, line before
                sl_if = np_t_sell_ind + 3  # Print sell index finish, line after
                print(bslap.iloc[sl_is:sl_if], "\n")

                # Chosen Exit (stop or sell)
                print("=================== EXIT ", pair)
                print("Exit type is :", t_exit_type)
                # print((bslap.iloc[t_exit_ind], "\n"))
                print("trade exit price field is", np_t_exit_pri, "\n")

            '''
            Trade entry is always the next candles "open" price
            We trigger on close, so cannot see that till after
            its closed.

            The exception to this is a STOP which is calculated in candle
            '''
            if debug_timing:
                t_t = f(st)
                print("6-depra", str.format('{0:.17f}', t_t))
                st = s()

            ## use numpy view "np_t_open_v" for speed. Columns are
            # buy 0 - open 1 - close 2 - sell 3 - high 4 - low 5
            # exception is 6 which is use the stop value.

            np_trade_enter_price = np_bslap[t_open_ind + 1, np_open]
            if t_exit_type == 'stop':
                if np_t_exit_pri == 6:
                    np_trade_exit_price = np_t_stop_pri
                else:
                    np_trade_exit_price = np_bslap[t_exit_ind, np_t_exit_pri]
            if t_exit_type == 'sell':
                np_trade_exit_price = np_bslap[t_exit_ind + 1, np_t_exit_pri]

            if debug_timing:
                t_t = f(st)
                print("7-numpy", str.format('{0:.17f}', t_t))
                st = s()

            if debug:
                print("//////////////////////////////////////////////")
                print("+++++++++++++++++++++++++++++++++ Trade Enter ")
                print("np_trade Enterprice is ", str.format('{0:.17f}', np_trade_enter_price))
                print("--------------------------------- Trade Exit ")
                print("Trade Exit Type is ", t_exit_type)
                print("np_trade Exit Price is", str.format('{0:.17f}', np_trade_exit_price))
                print("//////////////////////////////////////////////")

            # Loop control - catch no closed trades.
            if debug:
                print("---------------------------------------- end of loop", loop,
                      " Dataframe Exit Index is: ", t_exit_ind)
                print("Exit Index Last, Exit Index Now Are: ", t_exit_last, t_exit_ind)

            if t_exit_last >= t_exit_ind:
                """
                When last trade exit equals index of last exit, there is no
                opportunity to close any more trades.

                Break loop and go on to next pair.

                TODO
                add handing here to record none closed open trades
                """

                if debug:
                    print(bslap_pair_results)

                break
            else:
                """
                Add trade to backtest looking results list of dicts
                Loop back to look for more trades.
                """
                # Index will change if incandle stop or look back as close Open and Sell
                if t_exit_type == 'stop':
                    close_index: int = t_exit_ind + 1
                elif t_exit_type == 'sell':
                    close_index: int = t_exit_ind + 2
                else:
                    close_index: int = t_exit_ind + 1

                # Munge the date / delta (bt already date formats...just subract)
                trade_start = bslap.iloc[t_open_ind + 1]['date']
                trade_end = bslap.iloc[close_index]['date']
                # def __datetime(date_str):
                #     return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S+00:00')
                trade_mins = (trade_end - trade_start).total_seconds() / 60

                # Profit ABS.
                # sumrecieved((rate * numTokens) * fee) - sumpaid ((rate * numTokens) * fee)
                sumpaid: float = (np_trade_enter_price * stake)
                sumpaid_fee: float = sumpaid * open_fee
                sumrecieved: float = (np_trade_exit_price * stake)
                sumrecieved_fee: float = sumrecieved * close_fee
                profit_abs: float = sumrecieved - sumpaid - sumpaid_fee - sumrecieved_fee

                # build trade dictionary
                bslap_result["pair"] = pair
                bslap_result["profit_percent"] = (np_trade_exit_price - np_trade_enter_price) / np_trade_enter_price
                bslap_result["profit_abs"] = round(profit_abs, 15)
                bslap_result["open_time"] = trade_start
                bslap_result["close_time"] = trade_end
                bslap_result["open_index"] = t_open_ind + 2 # +1 between np and df, +1 as we buy on next.
                bslap_result["close_index"] = close_index
                bslap_result["trade_duration"] = trade_mins
                bslap_result["open_at_end"] = False
                bslap_result["open_rate"] = round(np_trade_enter_price, 15)
                bslap_result["close_rate"] = round(np_trade_exit_price, 15)
                #bslap_result["exit_type"] = t_exit_type
                # Add trade dictionary to list
                bslap_pair_results.append(bslap_result)
                if debug:
                    print(bslap_pair_results)

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

        logger.info(
            '\n================================================= '
            'BACKTESTING REPORT'
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
