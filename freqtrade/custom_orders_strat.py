# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

from datetime import datetime
import time
import timeit


# --------------------------------
class customorders(IStrategy):
    """
    author@: Creslin

    """
    minimal_roi = {
        "0": 0.20
    }
    # Optimal stoploss designed for the strategy
    stoploss = -0.01

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'

    def get_ticker_indicator(self):
        return int(self.ticker_interval[:-1])

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        #

        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        time.sleep(1)

        return dataframe

    # # # If using Berlins expose pairs to strategy
    # def advise_indicators(self, dataframe: DataFrame, pair: str) -> DataFrame:
    #
    #     # RSI
    #     dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
    #
    #     return dataframe

    #####
    #  EXAMPLE Strat - buys every even minute sells every odd
    ####
    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        if int(datetime.now().strftime('%M')) % 2 == 0: #buy = 1 every even minute
            print('----------------we are in strategy', datetime.now().time())
            b: float = 9999999.99
            print( "buy = 1")
        else:
            b = 0.00000001
        dataframe.loc[
            (
                # L-RSI When below 0.01 / vfi -26
                #(dataframe['lrsi'] < 0.5) &
                #(dataframe['vfi'] <  -4)
                (dataframe['close'] < b )
            ),
            'buy'] = 1
        return dataframe

    #####
    #  EXAMPLE Strat - buys every even minute sells every odd
    ####
    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        if int(datetime.now().strftime('%M'))% 2  != 0:  ## sell = 1 every ODD minute
            print('----------------we are in strategy', datetime.now().time())
            s: float = 9999999.99
            print("sell = 1")
        else:
            s = 0.00000001
        dataframe.loc[
            (
                # vfi 0.99
                #(dataframe['vfi'] > 0.41)
                (dataframe['close'] < s )
            ),
            'sell'] = 1
        return dataframe

    def stop_stops_plugin(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        stop_stops_pligin. To stop a buy where X trades from Y have stopped out.
        i.e to remove a pair from being bought that is on a losing streak.

        Only processed if buy=1
        """
        """
        Called before a trade executes if exists, 
        :param dataframe:
        :param pair:
        :return:
        """

        from freqtrade.persistence import Trade
        import pandas as pd
        df = dataframe

        # Only check buys, else return DF untouched

        if df.iloc[-1]['buy'] == 1:

            print("STOP_STOPS STOP_STOPS STOP_STOPS STOP_STOP STOP_STOPS STOP_STOP STOP_STOPS")

            from pandas import set_option
            set_option('display.max_rows', 2000)
            set_option('display.max_columns', 8)

            def closed_pair_trades(pair):
                """
                To return list of closed trades for a pair
                enriched with open/closd dates, profit, and stake

                :param pair:
                :return: df_c pair open, close, profit, stake
                """
                pair = pair
                df_c = pd.DataFrame(columns=['pair', 'profit', 'open_date', 'close_date', 'stake'])

                pair_trades = Trade.query.filter(Trade.pair.is_(pair)).all()
                for pair_trade in pair_trades:
                    if pair_trade.is_open == False:
                        # print("closed trade for", pair, "closed with ", pair_trade.close_profit )

                        p = pair
                        pcp = pair_trade.close_profit
                        od = pair_trade.open_date
                        cd = pair_trade.close_date
                        sa = pair_trade.stake_amount

                        df_c = df_c.append([{'pair':p, 'profit':pcp, 'open_date':od,
                                             'close_date':cd, 'stake':sa}], ignore_index=True)
                return df_c

            def has_past_perf(df_c, lost_the_last=1, out_of=1):
                """
                Return if pair lost_the_last X trades out_of Y trades
                :param df_c: dataframe of closed trades for the pair
                :param lost_the_last: number of bad trades to look for
                :param out_of: within the last number over trades closed
                :return: bool
                """
                lost_the_last =lost_the_last
                out_of = out_of
                df_c = df_c

                df_c_tail = df_c.tail(out_of)
                lost_from_last_count = df_c_tail[df_c_tail['profit'] < 0].shapec
                if lost_from_last_count[0] >= lost_the_last:
                    return True
                else:
                    return False

            # load this pairs closed trades into a df
            # df_c = closed_pair_trades(pair=pair)
            # print("df_c", df_c)

            # Get bool if on pairs past profitable trades count, cancel the buy if limit hit
            # past_perf = has_past_perf(df_c, lost_the_last=9, out_of=10)
            # if past_perf:
            #     #set buy to 0
            #     print("STOP STOP SETTING BUY TO ZERO ")
            #     df.loc['buy'] = 0

            dataframe  = df
        return dataframe

    #def money_mgt_plugin(self, dataframe: DataFrame, pair: str) -> DataFrame:
    def money_mgt_plugin(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        pre_trade_mgt.
        To allow money mamagement and or other checks to be put against dataframe
        with buy or sell set to "1"

        Returns a list of custom order trades.
        Each custom order to include, above the needed fields for exchange

        dict['order_type'] = 'a_open_limit'

        order_types are later ordered and executed in alphabetical order.
        order to enter market are executed before stops and take profits are played in.
        Supported types are:
        - a_open_limit   # enter market with limit order
        - a_open_market  # enter market with market order
        - b_stop_limit   # set a stop_limit order
        - b_stop_market  # set a stop_market order
        - c_take_limit   # set a take profit limit order
        - c_take_market  # set a take profit market order

        Multipe orders can be processed but an asset may only have the sum of its open value
        in either stop or take profit orders.

        i.e we can open a $500 for 500 tokens position and have any combination of
        stop or take profit orders at any prices that COMBINED do not break 500 tokens.

        "one" common scenario may be an 'a_open_limit' or 'a_open_market' order
        and 'b_stop_limit' or 'b_stop_market' order to protect capital.

        """
        print("MONEY_MGT MONEY_MGT MONEY_MGT MONEY_MGT MONEY_MGT MONEY_MGT MONEY_MGT MONEY_MGT MONEY_MGT")
        df = dataframe

        # If buy = 1 and Exchange is GDAX
        if df.iloc[-1]['buy'] == 1:
            exchange = self.config['exchange']['name']

            # GDAX Stop-Limit Order
            if exchange == "gdax":

                custom_orders = []
                limit = {}
                take_profit = {}

                # Take profit Market example "b_take_market"
                # open market order example "a_open_market"

                # Open limit order example
                amount = 0.5
                limit_price = df.iloc[-1]['close'] * 1.05 # pay 5% for testing order are made

                limit = {}
                limit["order_type"] = "a_open_limit"
                limit["symbol"] = pair
                limit["type"] = "limit"
                limit["side"] = "buy"
                limit["amount"] = amount
                limit["price"] = limit_price
                params: dict = {}
                limit["params"] = params

                custom_orders.append(limit)

                # Market Example
                market_amount = 1

                market = {}
                market_price = 1.05  # pay 5% for testing order are made
                market["order_type"] = "a_open_market"
                market["symbol"] = pair
                market["type"] = "market"
                market["side"] = "buy"
                market["amount"] = market_amount
                params: dict = {}
                market["params"] = params

                custom_orders.append(market)

                # Stop Limit order
                stop_limit = {}
                stop_limit_amount = 0.01
                stop_limit_price = df.iloc[-1]['close'] * 0.90  # take 10% hard code for testing
                stop_price = df.iloc[-1]['close'] * 0.95  # trigger at 7% loss, hard code for testing

                stop_limit['order_type'] = 'b_stop_limit'
                stop_limit['symbol'] = pair
                stop_limit['type'] = 'limit'
                stop_limit['side'] = 'sell'
                stop_limit['amount'] = stop_limit_amount
                stop_limit['price'] = stop_limit_price
                params: dict = {}
                params['stop'] = 'loss'
                params['stop_price'] = stop_price
                stop_limit['params'] = params

                custom_orders.append(stop_limit)

                # Take profit Limit example
                take_profit_limit_price = df.iloc[-1]['close'] * 1.2  # take_proft at 20% - for demo
                take_profit_amount = 0.01

                take_profit["order_type"] = "c_take_profit_limit"
                take_profit["order_template"] = (pair, amount, take_profit_limit_price)
                take_profit["symbol"] = pair
                take_profit["type"] = "limit"
                take_profit["side"] = "sell"
                take_profit["amount"] = take_profit_amount
                take_profit["price"] = take_profit_limit_price
                take_profit_params: dict = {}
                take_profit["params"] = take_profit_params

                # custom_orders.append(take_profit)

                #todo Add library of trade templates for binance, polo, gdax...
                #todo Code to check have all the fields and sense check

                # Cancel Legacy buy/sell signals if there are custom orders.
                # Set buy / sell = 0 if we have custom orders
                dataframe.loc['buy'] = 0
                dataframe.loc['sell'] = 0

                return dataframe, custom_orders

        return dataframe, None
