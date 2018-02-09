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
        self.logger = None

        self._init_logger()

    def _init_logger(self) -> None:
        """
        Setup the bot logger configuration
        :return: None
        """
        logging.basicConfig(
            level=self.level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

        self.logger = self.get_logger()
        self.set_level(self.level)

    def get_logger(self) -> logging.RootLogger:
        """
        Return the logger instance to use for sending message
        :return: the logger instance
        """
        return logging.getLogger(self.name)

    def set_name(self, name: str) -> logging.RootLogger:
        """
        Set the name of the logger
        :param name: Name of the logger
        :return: None
        """
        self.name = name
        self.logger = self.get_logger()
        return self.logger

    def set_level(self, level) -> None:
        """
        Set the level of the logger
        :param level:
        :return: None
        """
        self.level = level
        self.logger.setLevel(self.level)

    def set_format(self, log_format: str, propagate: bool = False) -> None:
        """
        Set a new logging format
        :return: None
        """
        handler = logging.StreamHandler()

        len_handlers = len(self.logger.handlers)
        if len_handlers:
            self.logger.removeHandler(handler)

        handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(handler)

        self.logger.propagate = propagate
