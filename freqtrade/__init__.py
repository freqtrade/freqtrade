""" FreqTrade bot """
__version__ = '0.18.5-dev'


class DependencyException(BaseException):
    """
    Indicates that a assumed dependency is not met.
    This could happen when there is currently not enough money on the account.
    """


class OperationalException(BaseException):
    """
    Requires manual intervention.
    This happens when an exchange returns an unexpected error during runtime
    or given configuration is invalid.
    """


class InvalidOrderException(BaseException):
    """
    This is returned when the order is not valid. Example:
    If stoploss on exchange order is hit, then trying to cancel the order
    should return this exception.
    """


class TemporaryError(BaseException):
    """
    Temporary network or exchange related error.
    This could happen when an exchange is congested, unavailable, or the user
    has networking problems. Usually resolves itself after a time.
    """
