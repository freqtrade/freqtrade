

class FreqtradeException(Exception):
    """
    Freqtrade base exception. Handled at the outermost level.
    All other exception types are subclasses of this exception type.
    """


class OperationalException(FreqtradeException):
    """
    Requires manual intervention and will stop the bot.
    Most of the time, this is caused by an invalid Configuration.
    """


class DependencyException(FreqtradeException):
    """
    Indicates that an assumed dependency is not met.
    This could happen when there is currently not enough money on the account.
    """


class PricingError(DependencyException):
    """
    Subclass of DependencyException.
    Indicates that the price could not be determined.
    Implicitly a buy / sell operation.
    """


class InvalidOrderException(FreqtradeException):
    """
    This is returned when the order is not valid. Example:
    If stoploss on exchange order is hit, then trying to cancel the order
    should return this exception.
    """


class TemporaryError(FreqtradeException):
    """
    Temporary network or exchange related error.
    This could happen when an exchange is congested, unavailable, or the user
    has networking problems. Usually resolves itself after a time.
    """


class StrategyError(FreqtradeException):
    """
    Errors with custom user-code deteced.
    Usually caused by errors in the strategy.
    """
