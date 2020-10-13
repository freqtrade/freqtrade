

from cachetools import TTLCache, cached


class LoggingMixin():
    """
    Logging Mixin
    Shows similar messages only once every `refresh_period`.
    """
    def __init__(self, logger, refresh_period: int = 3600):
        """
        :param refresh_period: in seconds - Show identical messages in this intervals
        """
        self.logger = logger
        self.refresh_period = refresh_period
        self._log_cache: TTLCache = TTLCache(maxsize=1024, ttl=self.refresh_period)

    def log_on_refresh(self, logmethod, message: str) -> None:
        """
        Logs message - not more often than "refresh_period" to avoid log spamming
        Logs the log-message as debug as well to simplify debugging.
        :param logmethod: Function that'll be called. Most likely `logger.info`.
        :param message: String containing the message to be sent to the function.
        :return: None.
        """
        @cached(cache=self._log_cache)
        def _log_on_refresh(message: str):
            logmethod(message)

        # Log as debug first
        self.logger.debug(message)
        # Call hidden function.
        _log_on_refresh(message)
