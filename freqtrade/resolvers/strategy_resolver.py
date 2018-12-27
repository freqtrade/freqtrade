# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom strategies
"""
import inspect
import logging
import tempfile
from base64 import urlsafe_b64decode
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

from freqtrade import constants
from freqtrade.resolvers import IResolver
from freqtrade.strategy import import_strategy
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

        # Verify the strategy is in the configuration, otherwise fallback to the default strategy
        strategy_name = config.get('strategy') or constants.DEFAULT_STRATEGY
        self.strategy: IStrategy = self._load_strategy(strategy_name,
                                                       config=config,
                                                       extra_dir=config.get('strategy_path'))

        # Set attributes
        # Check if we need to override configuration
        if 'minimal_roi' in config:
            self.strategy.minimal_roi = config['minimal_roi']
            logger.info("Override strategy 'minimal_roi' with value in config file: %s.",
                        config['minimal_roi'])
        else:
            config['minimal_roi'] = self.strategy.minimal_roi

        if 'stoploss' in config:
            self.strategy.stoploss = config['stoploss']
            logger.info(
                "Override strategy 'stoploss' with value in config file: %s.", config['stoploss']
            )
        else:
            config['stoploss'] = self.strategy.stoploss

        if 'ticker_interval' in config:
            self.strategy.ticker_interval = config['ticker_interval']
            logger.info(
                "Override strategy 'ticker_interval' with value in config file: %s.",
                config['ticker_interval']
            )
        else:
            config['ticker_interval'] = self.strategy.ticker_interval

        if 'process_only_new_candles' in config:
            self.strategy.process_only_new_candles = config['process_only_new_candles']
            logger.info(
                "Override process_only_new_candles 'process_only_new_candles' "
                "with value in config file: %s.", config['process_only_new_candles']
            )
        else:
            config['process_only_new_candles'] = self.strategy.process_only_new_candles

        if 'order_types' in config:
            self.strategy.order_types = config['order_types']
            logger.info(
                "Override strategy 'order_types' with value in config file: %s.",
                config['order_types']
            )
        else:
            config['order_types'] = self.strategy.order_types

        if 'order_time_in_force' in config:
            self.strategy.order_time_in_force = config['order_time_in_force']
            logger.info(
                "Override strategy 'order_time_in_force' with value in config file: %s.",
                config['order_time_in_force']
            )
        else:
            config['order_time_in_force'] = self.strategy.order_time_in_force

        if not all(k in self.strategy.order_types for k in constants.REQUIRED_ORDERTYPES):
            raise ImportError(f"Impossible to load Strategy '{self.strategy.__class__.__name__}'. "
                              f"Order-types mapping is incomplete.")

        if not all(k in self.strategy.order_time_in_force for k in constants.REQUIRED_ORDERTIF):
            raise ImportError(f"Impossible to load Strategy '{self.strategy.__class__.__name__}'. "
                              f"Order-time-in-force mapping is incomplete.")

        # Sort and apply type conversions
        self.strategy.minimal_roi = OrderedDict(sorted(
            {int(key): value for (key, value) in self.strategy.minimal_roi.items()}.items(),
            key=lambda t: t[0]))
        self.strategy.stoploss = float(self.strategy.stoploss)

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

        abs_paths = [
            Path.cwd().joinpath('user_data/strategies'),
            current_path,
        ]

        if extra_dir:
            # Add extra strategy directory on top of search paths
            abs_paths.insert(0, Path(extra_dir).resolve())

        if ":" in strategy_name:
            logger.info("loading base64 endocded strategy")
            strat = strategy_name.split(":")

            if len(strat) == 2:
                temp = Path(tempfile.mkdtemp("freq", "strategy"))
                name = strat[0] + ".py"

                temp.joinpath(name).write_text(urlsafe_b64decode(strat[1]).decode('utf-8'))
                temp.joinpath("__init__.py").touch()

                strategy_name = strat[0]

                # register temp path with the bot
                abs_paths.insert(0, temp.resolve())

        for _path in abs_paths:
            try:
                strategy = self._search_object(directory=_path, object_type=IStrategy,
                                               object_name=strategy_name, kwargs={'config': config})
                if strategy:
                    logger.info('Using resolved strategy %s from \'%s\'', strategy_name, _path)
                    strategy._populate_fun_len = len(
                        inspect.getfullargspec(strategy.populate_indicators).args)
                    strategy._buy_fun_len = len(
                        inspect.getfullargspec(strategy.populate_buy_trend).args)
                    strategy._sell_fun_len = len(
                        inspect.getfullargspec(strategy.populate_sell_trend).args)

                    return import_strategy(strategy, config=config)
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist', _path.relative_to(Path.cwd()))

        raise ImportError(
            "Impossible to load Strategy '{}'. This class does not exist"
            " or contains Python code errors".format(strategy_name)
        )
