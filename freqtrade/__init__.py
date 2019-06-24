""" FreqTrade bot """
__version__ = '2019.6'


class DependencyException(Exception):
    """
    Indicates that an assumed dependency is not met.
    This could happen when there is currently not enough money on the account.
    """


class OperationalException(Exception):
    """
    Requires manual intervention.
    This happens when an exchange returns an unexpected error during runtime
    or given configuration is invalid.
    """


class InvalidOrderException(Exception):
    """
    This is returned when the order is not valid. Example:
    If stoploss on exchange order is hit, then trying to cancel the order
    should return this exception.
    """


class TemporaryError(Exception):
    """
    Temporary network or exchange related error.
    This could happen when an exchange is congested, unavailable, or the user
    has networking problems. Usually resolves itself after a time.
    """
