# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom strategies
"""

import logging
import tempfile
from base64 import urlsafe_b64decode
from inspect import getfullargspec
from os import walk
from pathlib import Path
from typing import Any, Optional

from freqtrade.configuration.config_validation import validate_migrated_strategy_settings
from freqtrade.constants import REQUIRED_ORDERTIF, REQUIRED_ORDERTYPES, USERPATH_STRATEGIES, Config
from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.resolvers import IResolver
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class StrategyResolver(IResolver):
    """
    This class contains the logic to load custom strategy class
    """

    object_type = IStrategy
    object_type_str = "Strategy"
    user_subdir = USERPATH_STRATEGIES
    initial_search_path = None
    extra_path = "strategy_path"

    @staticmethod
    def load_strategy(config: Optional[Config] = None) -> IStrategy:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        config = config or {}

        if not config.get("strategy"):
            raise OperationalException(
                "No strategy set. Please use `--strategy` to specify the strategy class to use."
            )

        strategy_name = config["strategy"]
        strategy: IStrategy = StrategyResolver._load_strategy(
            strategy_name, config=config, extra_dir=config.get("strategy_path")
        )
        strategy.ft_load_params_from_file()
        # Set attributes
        # Check if we need to override configuration
        #             (Attribute name,                    default,     subkey)
        attributes = [
            ("minimal_roi", {"0": 10.0}),
            ("timeframe", None),
            ("stoploss", None),
            ("trailing_stop", None),
            ("trailing_stop_positive", None),
            ("trailing_stop_positive_offset", 0.0),
            ("trailing_only_offset_is_reached", None),
            ("use_custom_stoploss", None),
            ("process_only_new_candles", None),
            ("order_types", None),
            ("order_time_in_force", None),
            ("stake_currency", None),
            ("stake_amount", None),
            ("startup_candle_count", None),
            ("unfilledtimeout", None),
            ("use_exit_signal", True),
            ("exit_profit_only", False),
            ("ignore_roi_if_entry_signal", False),
            ("exit_profit_offset", 0.0),
            ("disable_dataframe_checks", False),
            ("ignore_buying_expired_candle_after", 0),
            ("position_adjustment_enable", False),
            ("max_entry_position_adjustment", -1),
            ("max_open_trades", -1),
        ]
        for attribute, default in attributes:
            StrategyResolver._override_attribute_helper(strategy, config, attribute, default)

        # Loop this list again to have output combined
        for attribute, _ in attributes:
            if attribute in config:
                logger.info("Strategy using %s: %s", attribute, config[attribute])

        StrategyResolver._normalize_attributes(strategy)

        StrategyResolver._strategy_sanity_validations(strategy)
        return strategy

    @staticmethod
    def _override_attribute_helper(strategy, config: Config, attribute: str, default: Any):
        """
        Override attributes in the strategy.
        Prevalence:
        - Configuration
        - Strategy
        - default (if not None)
        """
        if attribute in config and not isinstance(
            getattr(type(strategy), attribute, None), property
        ):
            # Ensure Properties are not overwritten
            setattr(strategy, attribute, config[attribute])
            logger.info(
                "Override strategy '%s' with value in config file: %s.",
                attribute,
                config[attribute],
            )
        elif hasattr(strategy, attribute):
            val = getattr(strategy, attribute)
            # None's cannot exist in the config, so do not copy them
            if val is not None:
                # max_open_trades set to -1 in the strategy will be copied as infinity in the config
                if attribute == "max_open_trades" and val == -1:
                    config[attribute] = float("inf")
                else:
                    config[attribute] = val
        # Explicitly check for None here as other "falsy" values are possible
        elif default is not None:
            setattr(strategy, attribute, default)
            config[attribute] = default

    @staticmethod
    def _normalize_attributes(strategy: IStrategy) -> IStrategy:
        """
        Normalize attributes to have the correct type.
        """
        # Sort and apply type conversions
        if hasattr(strategy, "minimal_roi"):
            strategy.minimal_roi = dict(
                sorted(
                    {int(key): value for (key, value) in strategy.minimal_roi.items()}.items(),
                    key=lambda t: t[0],
                )
            )
        if hasattr(strategy, "stoploss"):
            strategy.stoploss = float(strategy.stoploss)
        if hasattr(strategy, "max_open_trades") and strategy.max_open_trades < 0:
            strategy.max_open_trades = float("inf")
        return strategy

    @staticmethod
    def _strategy_sanity_validations(strategy: IStrategy):
        # Ensure necessary migrations are performed first.
        validate_migrated_strategy_settings(strategy.config)

        if not all(k in strategy.order_types for k in REQUIRED_ORDERTYPES):
            raise ImportError(
                f"Impossible to load Strategy '{strategy.__class__.__name__}'. "
                f"Order-types mapping is incomplete."
            )
        if not all(k in strategy.order_time_in_force for k in REQUIRED_ORDERTIF):
            raise ImportError(
                f"Impossible to load Strategy '{strategy.__class__.__name__}'. "
                f"Order-time-in-force mapping is incomplete."
            )
        trading_mode = strategy.config.get("trading_mode", TradingMode.SPOT)

        if strategy.can_short and trading_mode == TradingMode.SPOT:
            raise ImportError(
                "Short strategies cannot run in spot markets. Please make sure that this "
                "is the correct strategy and that your trading mode configuration is correct. "
                "You can run this strategy in spot markets by setting `can_short=False`"
                " in your strategy. Please note that short signals will be ignored in that case."
            )

    @staticmethod
    def validate_strategy(strategy: IStrategy) -> IStrategy:
        if strategy.config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            # Require new method
            warn_deprecated_setting(strategy, "sell_profit_only", "exit_profit_only", True)
            warn_deprecated_setting(strategy, "sell_profit_offset", "exit_profit_offset", True)
            warn_deprecated_setting(strategy, "use_sell_signal", "use_exit_signal", True)
            warn_deprecated_setting(
                strategy, "ignore_roi_if_buy_signal", "ignore_roi_if_entry_signal", True
            )

            if not check_override(strategy, IStrategy, "populate_entry_trend"):
                raise OperationalException("`populate_entry_trend` must be implemented.")
            if not check_override(strategy, IStrategy, "populate_exit_trend"):
                raise OperationalException("`populate_exit_trend` must be implemented.")
            if check_override(strategy, IStrategy, "check_buy_timeout"):
                raise OperationalException(
                    "Please migrate your implementation "
                    "of `check_buy_timeout` to `check_entry_timeout`."
                )
            if check_override(strategy, IStrategy, "check_sell_timeout"):
                raise OperationalException(
                    "Please migrate your implementation "
                    "of `check_sell_timeout` to `check_exit_timeout`."
                )

            if check_override(strategy, IStrategy, "custom_sell"):
                raise OperationalException(
                    "Please migrate your implementation of `custom_sell` to `custom_exit`."
                )

        else:
            # TODO: Implementing one of the following methods should show a deprecation warning
            #  buy_trend and sell_trend, custom_sell
            warn_deprecated_setting(strategy, "sell_profit_only", "exit_profit_only")
            warn_deprecated_setting(strategy, "sell_profit_offset", "exit_profit_offset")
            warn_deprecated_setting(strategy, "use_sell_signal", "use_exit_signal")
            warn_deprecated_setting(
                strategy, "ignore_roi_if_buy_signal", "ignore_roi_if_entry_signal"
            )

            if not check_override(strategy, IStrategy, "populate_buy_trend") and not check_override(
                strategy, IStrategy, "populate_entry_trend"
            ):
                raise OperationalException(
                    "`populate_entry_trend` or `populate_buy_trend` must be implemented."
                )
            if not check_override(
                strategy, IStrategy, "populate_sell_trend"
            ) and not check_override(strategy, IStrategy, "populate_exit_trend"):
                raise OperationalException(
                    "`populate_exit_trend` or `populate_sell_trend` must be implemented."
                )

            _populate_fun_len = len(getfullargspec(strategy.populate_indicators).args)
            _buy_fun_len = len(getfullargspec(strategy.populate_buy_trend).args)
            _sell_fun_len = len(getfullargspec(strategy.populate_sell_trend).args)
            if any(x == 2 for x in [_populate_fun_len, _buy_fun_len, _sell_fun_len]):
                raise OperationalException(
                    "Strategy Interface v1 is no longer supported. "
                    "Please update your strategy to implement "
                    "`populate_indicators`, `populate_entry_trend` and `populate_exit_trend` "
                    "with the metadata argument. "
                )

        has_after_fill = "after_fill" in getfullargspec(
            strategy.custom_stoploss
        ).args and check_override(strategy, IStrategy, "custom_stoploss")
        if has_after_fill:
            strategy._ft_stop_uses_after_fill = True

        return strategy

    @staticmethod
    def _load_strategy(
        strategy_name: str, config: Config, extra_dir: Optional[str] = None
    ) -> IStrategy:
        """
        Search and loads the specified strategy.
        :param strategy_name: name of the module to import
        :param config: configuration for the strategy
        :param extra_dir: additional directory to search for the given strategy
        :return: Strategy instance or None
        """
        if config.get("recursive_strategy_search", False):
            extra_dirs: list[str] = [
                path[0] for path in walk(f"{config['user_data_dir']}/{USERPATH_STRATEGIES}")
            ]  # sub-directories
        else:
            extra_dirs = []

        if extra_dir:
            extra_dirs.append(extra_dir)

        abs_paths = StrategyResolver.build_search_paths(
            config, user_subdir=USERPATH_STRATEGIES, extra_dirs=extra_dirs
        )

        if ":" in strategy_name:
            logger.info("loading base64 encoded strategy")
            strat = strategy_name.split(":")

            if len(strat) == 2:
                temp = Path(tempfile.mkdtemp("freq", "strategy"))
                name = strat[0] + ".py"

                temp.joinpath(name).write_text(urlsafe_b64decode(strat[1]).decode("utf-8"))
                temp.joinpath("__init__.py").touch()

                strategy_name = strat[0]

                # register temp path with the bot
                abs_paths.insert(0, temp.resolve())

        strategy = StrategyResolver._load_object(
            paths=abs_paths,
            object_name=strategy_name,
            add_source=True,
            kwargs={"config": config},
        )

        if strategy:
            return StrategyResolver.validate_strategy(strategy)

        raise OperationalException(
            f"Impossible to load Strategy '{strategy_name}'. This class does not exist "
            "or contains Python code errors."
        )


def warn_deprecated_setting(strategy: IStrategy, old: str, new: str, error=False):
    if hasattr(strategy, old):
        errormsg = f"DEPRECATED: Using '{old}' moved to '{new}'."
        if error:
            raise OperationalException(errormsg)
        logger.warning(errormsg)
        setattr(strategy, new, getattr(strategy, f"{old}"))


def check_override(obj, parentclass, attribute: str):
    """
    Checks if a object overrides the parent class attribute.
    :returns: True if the object is overridden.
    """
    return getattr(type(obj), attribute) != getattr(parentclass, attribute)
