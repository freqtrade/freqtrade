from . import telegram

REGISTERED_MODULES = []


def init(config: dict) -> None:
    """
    Initializes all enabled rpc modules
    :param config: config to use
    :return: None
    """

    if config['telegram'].get('enabled', False):
        REGISTERED_MODULES.append('telegram')
        telegram.init(config)


def cleanup() -> None:
    """
    Stops all enabled rpc modules
    :return: None
    """
    if 'telegram' in REGISTERED_MODULES:
        telegram.cleanup()


def send_msg(msg: str) -> None:
    """
    Send given markdown message to all registered rpc modules
    :param msg: message
    :return: None
    """
    if 'telegram' in REGISTERED_MODULES:
        telegram.send_msg(msg)
