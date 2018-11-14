# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import logging
from freqtrade.execution.interface import IExecution
from typing import Dict

logger = logging.getLogger(__name__)


class DefaultExecution(IExecution):

    def _get_target_bid(self, pair: str, ticker: Dict[str, float]) -> float:
        """
        Calculates bid target between current ask price and last price
        :param ticker: Ticker to use for getting Ask and Last Price
        :return: float: Price
        """
        if ticker['ask'] < ticker['last']:
            ticker_rate = ticker['ask']
        else:
            balance = self.config['bid_strategy']['ask_last_balance']
            ticker_rate = ticker['ask'] + balance * (ticker['last'] - ticker['ask'])

        used_rate = ticker_rate
        config_bid_strategy = self.config.get('bid_strategy', {})
        if 'use_order_book' in config_bid_strategy and\
                config_bid_strategy.get('use_order_book', False):
            logger.info('Getting price from order book')
            order_book_top = config_bid_strategy.get('order_book_top', 1)
            order_book = self.exchange.get_order_book(pair, order_book_top)
            logger.debug('order_book %s', order_book)
            # top 1 = index 0
            order_book_rate = order_book['bids'][order_book_top - 1][0]
            # if ticker has lower rate, then use ticker ( usefull if down trending )
            logger.info('...top %s order book buy rate %0.8f', order_book_top, order_book_rate)
            if ticker_rate < order_book_rate:
                logger.info('...using ticker rate instead %0.8f', ticker_rate)
                used_rate = ticker_rate
            else:
                used_rate = order_book_rate
        else:
            logger.info('Using Last Ask / Last Price')
            used_rate = ticker_rate

        return used_rate

    def execute_buy(self, pair: str, stake_amount: float, price: float) -> str:
        pair_s = pair.replace('_', '/')

        if price:
            buy_limit = price
        else:
            # Calculate amount
            buy_limit = self._get_target_bid(pair, self.exchange.get_ticker(pair))

        min_stake_amount = self._get_min_pair_stake_amount(pair_s, buy_limit)
        if min_stake_amount is not None and min_stake_amount > stake_amount:
            logger.warning(
                f'Can\'t open a new trade for {pair_s}: stake amount'
                f' is too small ({stake_amount} < {min_stake_amount})'
            )
            return False

        amount = stake_amount / buy_limit

        order_id = self.exchange.buy(pair, buy_limit, amount)['id']
        return order_id or None
