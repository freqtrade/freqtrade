# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom strategies
"""
import logging
import tempfile
from base64 import urlsafe_b64decode
from collections import OrderedDict
from inspect import getfullargspec
from pathlib import Path
from typing import Dict, Optional

from freqtrade import constants, OperationalException
from freqtrade.resolvers import IResolver
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)


class StrategyResolver(IResolver):
    """
    This class contains all the logic to load custom strategy class
    """

    __slots__ = ['strategy']

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        config = config or {}

        if not config.get('strategy'):
            raise OperationalException("No strategy set. Please use `--strategy` to specify "
                                       "the strategy class to use.")

        strategy_name = config['strategy']
        self.strategy: IStrategy = self._load_strategy(strategy_name,
                                                       config=config,
                                                       extra_dir=config.get('strategy_path'))

        # make sure ask_strategy dict is available
        if 'ask_strategy' not in config:
            config['ask_strategy'] = {}

        # Set attributes
        # Check if we need to override configuration
        #             (Attribute name,                    default,     ask_strategy)
        attributes = [("minimal_roi",                     {"0": 10.0}, False),
                      ("ticker_interval",                 None,        False),
                      ("stoploss",                        None,        False),
                      ("trailing_stop",                   None,        False),
                      ("trailing_stop_positive",          None,        False),
                      ("trailing_stop_positive_offset",   0.0,         False),
                      ("trailing_only_offset_is_reached", None,        False),
                      ("process_only_new_candles",        None,        False),
                      ("order_types",                     None,        False),
                      ("order_time_in_force",             None,        False),
                      ("stake_currency",                  None,        False),
                      ("stake_amount",                    None,        False),
                      ("startup_candle_count",            None,        False),
                      ("use_sell_signal",                 True,        True),
                      ("sell_profit_only",                False,       True),
                      ("ignore_roi_if_buy_signal",        False,       True),
                      ]
        for attribute, default, ask_strategy in attributes:
            if ask_strategy:
                self._override_attribute_helper(config['ask_strategy'], attribute, default)
            else:
                self._override_attribute_helper(config, attribute, default)

        # Loop this list again to have output combined
        for attribute, _, exp in attributes:
            if exp and attribute in config['ask_strategy']:
                logger.info("Strategy using %s: %s", attribute, config['ask_strategy'][attribute])
            elif attribute in config:
                logger.info("Strategy using %s: %s", attribute, config[attribute])

        # Sort and apply type conversions
        self.strategy.minimal_roi = OrderedDict(sorted(
            {int(key): value for (key, value) in self.strategy.minimal_roi.items()}.items(),
            key=lambda t: t[0]))
        self.strategy.stoploss = float(self.strategy.stoploss)

        self._strategy_sanity_validations()

    def _override_attribute_helper(self, config, attribute: str, default):
        """
        Override attributes in the strategy.
        Prevalence:
        - Configuration
        - Strategy
        - default (if not None)
        """
        if attribute in config:
            setattr(self.strategy, attribute, config[attribute])
            logger.info("Override strategy '%s' with value in config file: %s.",
                        attribute, config[attribute])
        elif hasattr(self.strategy, attribute):
            val = getattr(self.strategy, attribute)
            # None's cannot exist in the config, so do not copy them
            if val is not None:
                config[attribute] = val
        # Explicitly check for None here as other "falsy" values are possible
        elif default is not None:
            setattr(self.strategy, attribute, default)
            config[attribute] = default

    def _strategy_sanity_validations(self):
        if not all(k in self.strategy.order_types for k in constants.REQUIRED_ORDERTYPES):
            raise ImportError(f"Impossible to load Strategy '{self.strategy.__class__.__name__}'. "
                              f"Order-types mapping is incomplete.")

        if not all(k in self.strategy.order_time_in_force for k in constants.REQUIRED_ORDERTIF):
            raise ImportError(f"Impossible to load Strategy '{self.strategy.__class__.__name__}'. "
                              f"Order-time-in-force mapping is incomplete.")

    def _load_strategy(
            self, strategy_name: str, config: dict, extra_dir: Optional[str] = None) -> IStrategy:
        """
        Search and loads the specified strategy.
        :param strategy_name: name of the module to import
        :param config: configuration for the strategy
        :param extra_dir: additional directory to search for the given strategy
        :return: Strategy instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('strategy').resolve()

        abs_paths = self.build_search_paths(config, current_path=current_path,
                                            user_subdir='strategies', extra_dir=extra_dir)

        if ":" in strategy_name:
            logger.info("loading base64 encoded strategy")
            strat = strategy_name.split(":")

            if len(strat) == 2:
                temp = Path(tempfile.mkdtemp("freq", "strategy"))
                name = strat[0] + ".py"

                temp.joinpath(name).write_text(urlsafe_b64decode(strat[1]).decode('utf-8'))
                temp.joinpath("__init__.py").touch()

                strategy_name = strat[0]

                # register temp path with the bot
                abs_paths.insert(0, temp.resolve())

        strategy = self._load_object(paths=abs_paths, object_type=IStrategy,
                                     object_name=strategy_name, kwargs={'config': config})
        if strategy:
            strategy._populate_fun_len = len(getfullargspec(strategy.populate_indicators).args)
            strategy._buy_fun_len = len(getfullargspec(strategy.populate_buy_trend).args)
            strategy._sell_fun_len = len(getfullargspec(strategy.populate_sell_trend).args)
            if any([x == 2 for x in [strategy._populate_fun_len,
                                     strategy._buy_fun_len,
                                     strategy._sell_fun_len]]):
                strategy.INTERFACE_VERSION = 1

            return strategy

        raise OperationalException(
            f"Impossible to load Strategy '{strategy_name}'. This class does not exist "
            "or contains Python code errors."
        )
