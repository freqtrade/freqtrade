import json
import logging

from wrapt import synchronized

logger = logging.getLogger(__name__)

_cur_conf = None


@synchronized
def get_conf(filename='config.json'):
    """
    Loads the config into memory and returns the instance of it
    :return: dict
    """
    global _cur_conf
    if not _cur_conf:
        with open(filename) as fp:
            _cur_conf = json.load(fp)
            validate_conf(_cur_conf)
    return _cur_conf


def validate_conf(conf):
    """
    Validates if the minimal possible config is provided
    :param conf: config as dict
    :return: None, raises ValueError if something is wrong
    """
    if not isinstance(conf.get('max_open_trades'), int):
        raise ValueError('max_open_trades must be a int')
    if not isinstance(conf.get('stake_currency'), str):
        raise ValueError('stake_currency must be a str')
    if not isinstance(conf.get('stake_amount'), float):
        raise ValueError('stake_amount  must be a float')
    if not isinstance(conf.get('dry_run'), bool):
        raise ValueError('dry_run must be a boolean')
    if not isinstance(conf.get('minimal_roi'), dict):
        raise ValueError('minimal_roi must be a dict')

    for i, (minutes, threshold) in enumerate(conf.get('minimal_roi').items()):
        if not isinstance(minutes, str):
            raise ValueError('minimal_roi[{}].key must be a string'.format(i))
        if not isinstance(threshold, float):
            raise ValueError('minimal_roi[{}].value must be a float'.format(i))

    if conf.get('telegram'):
        telegram = conf.get('telegram')
        if not isinstance(telegram.get('token'), str):
            raise ValueError('telegram.token must be a string')
        if not isinstance(telegram.get('chat_id'), str):
            raise ValueError('telegram.chat_id must be a string')

    if conf.get('poloniex'):
        poloniex = conf.get('poloniex')
        if not isinstance(poloniex.get('key'), str):
            raise ValueError('poloniex.key must be a string')
        if not isinstance(poloniex.get('secret'), str):
            raise ValueError('poloniex.secret must be a string')
        if not isinstance(poloniex.get('pair_whitelist'), list):
            raise ValueError('poloniex.pair_whitelist must be a list')
        if poloniex.get('enabled', False):
            if not poloniex.get('pair_whitelist'):
                raise ValueError('poloniex.pair_whitelist must contain some pairs')

    if conf.get('bittrex'):
        bittrex = conf.get('bittrex')
        if not isinstance(bittrex.get('key'), str):
            raise ValueError('bittrex.key must be a string')
        if not isinstance(bittrex.get('secret'), str):
            raise ValueError('bittrex.secret must be a string')
        if not isinstance(bittrex.get('pair_whitelist'), list):
            raise ValueError('bittrex.pair_whitelist must be a list')
        if bittrex.get('enabled', False):
            if not bittrex.get('pair_whitelist'):
                raise ValueError('bittrex.pair_whitelist must contain some pairs')

    if conf.get('poloniex', {}).get('enabled', False) \
            and conf.get('bittrex', {}).get('enabled', False):
        raise ValueError('Cannot use poloniex and bittrex at the same time')

    logger.info('Config is valid ...')
