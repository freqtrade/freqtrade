""" FreqTrade bot """
__version__ = '0.16.0'


class DependencyException(BaseException):
    """
    Indicates that a assumed dependency is not met.
    This could happen when there is currently not enough money on the account.
    """


class OperationalException(BaseException):
    """
    Requires manual intervention.
    This happens when an exchange returns an unexpected error during runtime.
    """


class NotEnoughFundsExeption(BaseException):
    """
    This happens when the exchange reports that not enough funds where available. We do not want to stop
    the bot in this case and just keep it going and suppress this message
    """
