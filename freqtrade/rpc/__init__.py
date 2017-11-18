import logging

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
