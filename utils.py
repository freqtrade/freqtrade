import json
import logging

logger = logging.getLogger(__name__)

_CUR_CONF = None


def get_conf():
    """
    Loads the config into memory and returns the instance of it
    :return: dict
    """
    global _CUR_CONF
    if not _CUR_CONF:
        with open('config.json') as fp:
            _CUR_CONF = json.load(fp)
            validate_conf(_CUR_CONF)
    return _CUR_CONF


def validate_conf(conf):
    """
    Validates if the minimal possible config is provided
    :param conf: config as dict
    :return: None, raises ValueError if something is wrong
    """
    if not isinstance(conf.get('stake_amount'), float):
        raise ValueError('stake_amount  must be a float')
    if not isinstance(conf.get('dry_run'), bool):
        raise ValueError('dry_run must be a boolean')
    if not isinstance(conf.get('trade_thresholds'), dict):
        raise ValueError('trade_thresholds must be a dict')
    if not isinstance(conf.get('trade_thresholds'), dict):
        raise ValueError('trade_thresholds must be a dict')

    for i, (minutes, threshold) in enumerate(conf.get('trade_thresholds').items()):
        if not isinstance(minutes, str):
            raise ValueError('trade_thresholds[{}].key must be a string'.format(i))
        if not isinstance(threshold, float):
            raise ValueError('trade_thresholds[{}].value must be a float'.format(i))

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

    if conf.get('bittrex'):
        bittrex = conf.get('bittrex')
        if not isinstance(bittrex.get('key'), str):
            raise ValueError('bittrex.key must be a string')
        if not isinstance(bittrex.get('secret'), str):
            raise ValueError('bittrex.secret must be a string')
        if not isinstance(bittrex.get('pair_whitelist'), list):
            raise ValueError('bittrex.pair_whitelist must be a list')

    if conf.get('poloniex', {}).get('enabled', False) \
            and conf.get('bittrex', {}).get('enabled', False):
        raise ValueError('Cannot use poloniex and bittrex at the same time')

    logger.info('Config is valid ...')
