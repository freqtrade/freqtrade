# pragma pylint: disable=too-few-public-methods

"""
This module contains the class for logger and logging messages
"""

import logging


class Logger(object):
    """
    Logging class
    """
    def __init__(self, name='', level=logging.INFO) -> None:
        """
        Init the logger class
        :param name: Name of the Logger scope
        :param level: Logger level that should be used
        :return: None
        """
        self.name = name
        self.level = level
        self._init_logger()

    def _init_logger(self) -> logging:
        """
        Setup the bot logger configuration
        :return: logging object
        """
        logging.basicConfig(
            level=self.level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

    def get_logger(self) -> logging.RootLogger:
        """
        Return the logger instance to use for sending message
        :return: the logger instance
        """
        return logging.getLogger(self.name)
