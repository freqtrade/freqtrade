import logging
import sys
from copy import deepcopy

from freqtrade.strategy.interface import IStrategy
# Import Default-Strategy to have hyperopt correctly resolve
from freqtrade.strategy.default_strategy import DefaultStrategy  # noqa: F401


logger = logging.getLogger(__name__)


def import_strategy(strategy: IStrategy, config: dict) -> IStrategy:
    """
    Imports given Strategy instance to global scope
    of freqtrade.strategy and returns an instance of it
    """

    # Copy all attributes from base class and class
    comb = {**strategy.__class__.__dict__, **strategy.__dict__}

    # Delete '_abc_impl' from dict as deepcopy fails on 3.7 with
    # `TypeError: can't pickle _abc_data objects``
    # This will only apply to python 3.7
    if sys.version_info.major == 3 and sys.version_info.minor == 7 and '_abc_impl' in comb:
        del comb['_abc_impl']

    attr = deepcopy(comb)

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

    return clazz(config)
