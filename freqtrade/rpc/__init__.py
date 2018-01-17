import logging
import re
import arrow
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.misc import State, get_state
from freqtrade import exchange
from . import telegram

logger = logging.getLogger(__name__)


REGISTERED_MODULES = []


def init(config: dict) -> None:
    """
    Initializes all enabled rpc modules
    :param config: config to use
    :return: None
    """

    if config['telegram'].get('enabled', False):
        logger.info('Enabling rpc.telegram ...')
        REGISTERED_MODULES.append('telegram')
        telegram.init(config)


def cleanup() -> None:
    """
    Stops all enabled rpc modules
    :return: None
    """
    if 'telegram' in REGISTERED_MODULES:
        logger.debug('Cleaning up rpc.telegram ...')
        telegram.cleanup()


def send_msg(msg: str) -> None:
    """
    Send given markdown message to all registered rpc modules
    :param msg: message
    :return: None
    """
    logger.info(msg)
    if 'telegram' in REGISTERED_MODULES:
        telegram.send_msg(msg)


def shorten_date(_date):
    """
    Trim the date so it fits on small screens
    """
    new_date = re.sub('seconds?', 'sec', _date)
    new_date = re.sub('minutes?', 'min', new_date)
    new_date = re.sub('hours?', 'h', new_date)
    new_date = re.sub('days?', 'd', new_date)
    new_date = re.sub('^an?', '1', new_date)
    return new_date


#
# Below follows the RPC backend
# it is prefixed with rpc_
# to raise awareness that it is
# a remotely exposed function


def rpc_trade_status():
    # Fetch open trade
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    if get_state() != State.RUNNING:
        return (True, '*Status:* `trader is not running`')
    elif not trades:
        return (True, '*Status:* `no active trade`')
    else:
        result = []
        for trade in trades:
            order = None
            if trade.open_order_id:
                order = exchange.get_order(trade.open_order_id)
            # calculate profit and send message to user
            current_rate = exchange.get_ticker(trade.pair, False)['bid']
            current_profit = trade.calc_profit_percent(current_rate)
            fmt_close_profit = '{:.2f}%'.format(
                round(trade.close_profit * 100, 2)
            ) if trade.close_profit else None
            message = """
*Trade ID:* `{trade_id}`
*Current Pair:* [{pair}]({market_url})
*Open Since:* `{date}`
*Amount:* `{amount}`
*Open Rate:* `{open_rate:.8f}`
*Close Rate:* `{close_rate}`
*Current Rate:* `{current_rate:.8f}`
*Close Profit:* `{close_profit}`
*Current Profit:* `{current_profit:.2f}%`
*Open Order:* `{open_order}`
            """.format(
                trade_id=trade.id,
                pair=trade.pair,
                market_url=exchange.get_pair_detail_url(trade.pair),
                date=arrow.get(trade.open_date).humanize(),
                open_rate=trade.open_rate,
                close_rate=trade.close_rate,
                current_rate=current_rate,
                amount=round(trade.amount, 8),
                close_profit=fmt_close_profit,
                current_profit=round(current_profit * 100, 2),
                open_order='({} rem={:.8f})'.format(
                    order['type'], order['remaining']
                ) if order else None,
            )
            result.append(message)
        return (False, result)


def rpc_status_table():
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    if get_state() != State.RUNNING:
        return (True, '*Status:* `trader is not running`')
    elif not trades:
        return (True, '*Status:* `no active order`')
    else:
        trades_list = []
        for trade in trades:
            # calculate profit and send message to user
            current_rate = exchange.get_ticker(trade.pair, False)['bid']
            trades_list.append([
                trade.id,
                trade.pair,
                shorten_date(arrow.get(trade.open_date).humanize(only_distance=True)),
                '{:.2f}%'.format(100 * trade.calc_profit_percent(current_rate))
            ])

        columns = ['ID', 'Pair', 'Since', 'Profit']
        df_statuses = DataFrame.from_records(trades_list, columns=columns)
        df_statuses = df_statuses.set_index(columns[0])
        # The style used throughout is to return a tuple
        # consisting of (error_occured?, result)
        # Another approach would be to just return the
        # result, or raise error
        return (False, df_statuses)
