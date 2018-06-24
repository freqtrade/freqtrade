import logging
from copy import deepcopy

from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


def import_strategy(strategy: IStrategy) -> IStrategy:
    """
    Imports given Strategy instance to global scope
    of freqtrade.strategy and returns an instance of it
    """
    # Copy all attributes from base class and class
    attr = deepcopy({**strategy.__class__.__dict__, **strategy.__dict__})
    # Adjust module name
    attr['__module__'] = 'freqtrade.strategy'

    name = strategy.__class__.__name__
    clazz = type(name, (IStrategy,), attr)

    logger.debug(
        'Imported strategy %s.%s as %s.%s',
        strategy.__module__, strategy.__class__.__name__,
        clazz.__module__, strategy.__class__.__name__,
    )

    # Modify global scope to declare class
    globals()[name] = clazz

    return clazz()
