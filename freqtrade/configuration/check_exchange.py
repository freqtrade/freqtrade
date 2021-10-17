import logging
from typing import Any, Dict

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import (available_exchanges, is_exchange_known_ccxt,
                                is_exchange_officially_supported, validate_exchange)


logger = logging.getLogger(__name__)


def check_exchange(config: Dict[str, Any], check_for_bad: bool = True) -> bool:
    """
    Check if the exchange name in the config file is supported by Freqtrade
    :param check_for_bad: if True, check the exchange against the list of known 'bad'
                          exchanges
    :return: False if exchange is 'bad', i.e. is known to work with the bot with
             critical issues or does not work at all, crashes, etc. True otherwise.
             raises an exception if the exchange if not supported by ccxt
             and thus is not known for the Freqtrade at all.
    """

    if (config['runmode'] in [RunMode.PLOT, RunMode.UTIL_NO_EXCHANGE, RunMode.OTHER]
       and not config.get('exchange', {}).get('name')):
        # Skip checking exchange in plot mode, since it requires no exchange
        return True
    logger.info("Checking exchange...")

    exchange = config.get('exchange', {}).get('name').lower()
    if not exchange:
        raise OperationalException(
            f'This command requires a configured exchange. You should either use '
            f'`--exchange <exchange_name>` or specify a configuration file via `--config`.\n'
            f'The following exchanges are available for Freqtrade: '
            f'{", ".join(available_exchanges())}'
        )

    if not is_exchange_known_ccxt(exchange):
        raise OperationalException(
            f'Exchange "{exchange}" is not known to the ccxt library '
            f'and therefore not available for the bot.\n'
            f'The following exchanges are available for Freqtrade: '
            f'{", ".join(available_exchanges())}'
        )

    valid, reason = validate_exchange(exchange)
    if not valid:
        if check_for_bad:
            raise OperationalException(f'Exchange "{exchange}"  will not work with Freqtrade. '
                                       f'Reason: {reason}')
        else:
            logger.warning(f'Exchange "{exchange}"  will not work with Freqtrade. Reason: {reason}')

    if is_exchange_officially_supported(exchange):
        logger.info(f'Exchange "{exchange}" is officially supported '
                    f'by the Freqtrade development team.')
    else:
        logger.warning(f'Exchange "{exchange}" is known to the the ccxt library, '
                       f'available for the bot, but not officially supported '
                       f'by the Freqtrade development team. '
                       f'It may work flawlessly (please report back) or have serious issues. '
                       f'Use it at your own discretion.')

    return True
