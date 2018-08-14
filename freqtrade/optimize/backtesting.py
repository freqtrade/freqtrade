# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
from argparse import Namespace
from typing import Any, Dict, List, Optional

from pandas import DataFrame

from freqtrade.optimize.optimize import IOptimize, BacktestResult, setup_configuration
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import SellType

logger = logging.getLogger(__name__)


class Backtesting(IOptimize):
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

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
            sell = self.strategy.should_sell(trade, sell_row.open, sell_row.date, buy_signal,
                                             sell_row.sell)
            if sell.sell_flag:

                return BacktestResult(pair=pair,
                                      profit_percent=trade.calc_profit_percent(rate=sell_row.open),
                                      profit_abs=trade.calc_profit(rate=sell_row.open),
                                      open_time=buy_row.date,
                                      close_time=sell_row.date,
                                      trade_duration=int((
                                          sell_row.date - buy_row.date).total_seconds() // 60),
                                      open_index=buy_row.Index,
                                      close_index=sell_row.Index,
                                      open_at_end=False,
                                      open_rate=buy_row.open,
                                      close_rate=sell_row.open,
                                      sell_reason=sell.sell_type
                                      )
        if partial_ticker:
            # no sell condition found - trade stil open at end of backtest period
            sell_row = partial_ticker[-1]
            btr = BacktestResult(pair=pair,
                                 profit_percent=trade.calc_profit_percent(rate=sell_row.open),
                                 profit_abs=trade.calc_profit(rate=sell_row.open),
                                 open_time=buy_row.date,
                                 close_time=sell_row.date,
                                 trade_duration=int((
                                     sell_row.date - buy_row.date).total_seconds() // 60),
                                 open_index=buy_row.Index,
                                 close_index=sell_row.Index,
                                 open_at_end=True,
                                 open_rate=buy_row.open,
                                 close_rate=sell_row.open,
                                 sell_reason=SellType.FORCE_SELL
                                 )
            logger.debug('Force_selling still open trade %s with %s perc - %s', btr.pair,
                         btr.profit_percent, btr.profit_abs)
            return btr
        return None

    def run(self, args: Dict) -> DataFrame:
        """
        Implements backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid, logging on this method

        :param args: a dict containing:
            stake_amount: btc amount to use for each trade
            processed: a processed dictionary with format {pair, data}
            max_open_trades: maximum number of concurrent trades (default: 0, disabled)
            position_stacking: do we allow position stacking? (default: False)
        :return: DataFrame
        """
        headers = ['date', 'buy', 'open', 'close', 'sell']
        processed = args['processed']
        max_open_trades = args.get('max_open_trades', 0)
        position_stacking = args.get('position_stacking', False)
        trades = []
        trade_count_lock: Dict = {}
        for pair, pair_data in processed.items():
            pair_data['buy'], pair_data['sell'] = 0, 0  # cleanup from previous run

            ticker_data = self.advise_sell(
                self.advise_buy(pair_data, {'pair': pair}), {'pair': pair})[headers].copy()

            # to avoid using data from future, we buy/sell with signal from previous candle
            ticker_data.loc[:, 'buy'] = ticker_data['buy'].shift(1)
            ticker_data.loc[:, 'sell'] = ticker_data['sell'].shift(1)

            ticker_data.drop(ticker_data.head(1).index, inplace=True)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            ticker = [x for x in ticker_data.itertuples()]

            lock_pair_until = None
            for index, row in enumerate(ticker):
                if row.buy == 0 or row.sell == 1:
                    continue  # skip rows where no buy signal or that would immediately sell off

                if not position_stacking:
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

        return DataFrame.from_records(trades, columns=BacktestResult._fields)


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
