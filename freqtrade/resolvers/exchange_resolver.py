"""
This module loads custom exchanges
"""

import logging
from inspect import isclass
from typing import Any

import freqtrade.exchange as exchanges
from freqtrade.constants import Config, ExchangeConfig
from freqtrade.exchange import MAP_EXCHANGE_CHILDCLASS, Exchange
from freqtrade.resolvers.iresolver import IResolver


logger = logging.getLogger(__name__)


class ExchangeResolver(IResolver):
    """
    This class contains all the logic to load a custom exchange class
    """

    object_type = Exchange

    @staticmethod
    def load_exchange(
        config: Config,
        *,
        exchange_config: ExchangeConfig | None = None,
        validate: bool = True,
        load_leverage_tiers: bool = False,
    ) -> Exchange:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        :param exchange_config: Optional exchange configuration. When provided, its "name"
            is used to select the exchange class - this allows loading a secondary exchange
            that differs from the main ``config["exchange"]`` (e.g. for informative data).
        """
        # Prefer the name from the explicit exchange_config (secondary exchanges),
        # falling back to the main exchange configuration.
        exchange_name: str = (exchange_config or config["exchange"])["name"]
        # Map exchange name to avoid duplicate classes for identical exchanges
        exchange_name = MAP_EXCHANGE_CHILDCLASS.get(exchange_name, exchange_name)
        exchange_name = exchange_name.title()
        exchange = None
        try:
            exchange = ExchangeResolver._load_exchange(
                exchange_name,
                kwargs={
                    "config": config,
                    "validate": validate,
                    "exchange_config": exchange_config,
                    "load_leverage_tiers": load_leverage_tiers,
                },
            )
        except ImportError:
            logger.info(
                f"No {exchange_name} specific subclass found. "
                "Using the generic exchange class instead."
            )
        if not exchange:
            exchange = Exchange(
                config,
                validate=validate,
                exchange_config=exchange_config,
            )
        return exchange

    @staticmethod
    def load_data_exchange(config: Config, exchange_config: ExchangeConfig) -> Exchange:
        """
        Load a secondary exchange used purely as a data source (informative pairs).
        These exchanges are never used for trading, so config validation and leverage
        tier loading are skipped.
        :param config: configuration dictionary (used for shared settings like runmode)
        :param exchange_config: exchange configuration for this data source. Must contain
            at least a "name".
        :return: Exchange instance
        """
        return ExchangeResolver.load_exchange(
            config,
            exchange_config=exchange_config,
            validate=False,
            load_leverage_tiers=False,
        )

    @staticmethod
    def _load_exchange(exchange_name: str, kwargs: dict) -> Exchange:
        """
        Loads the specified exchange.
        Only checks for exchanges exported in freqtrade.exchanges
        :param exchange_name: name of the module to import
        :return: Exchange instance or None
        """

        try:
            ex_class = getattr(exchanges, exchange_name)

            exchange = ex_class(**kwargs)
            if exchange:
                logger.info(f"Using resolved exchange class '{exchange_name}' ...")
                return exchange
        except AttributeError:
            # Pass and raise ImportError instead
            pass

        raise ImportError(
            f"Impossible to load Exchange '{exchange_name}'. This class does not exist "
            "or contains Python code errors."
        )

    @classmethod
    def search_all_objects(
        cls, config: Config, enum_failed: bool, recursive: bool = False
    ) -> list[dict[str, Any]]:
        """
        Searches for valid objects
        :param config: Config object
        :param enum_failed: If True, will return None for modules which fail.
            Otherwise, failing modules are skipped.
        :param recursive: Recursively walk directory tree searching for strategies
        :return: List of dicts containing 'name', 'class' and 'location' entries
        """
        result = []
        for exchange_name in dir(exchanges):
            exchange = getattr(exchanges, exchange_name)
            if isclass(exchange) and issubclass(exchange, Exchange):
                result.append(
                    {
                        "name": exchange_name,
                        "class": exchange,
                        "location": exchange.__module__,
                        "location_rel: ": exchange.__module__.replace("freqtrade.", ""),
                    }
                )
        return result
