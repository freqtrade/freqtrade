# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the edge backtesting interface
"""
import logging
from typing import Any, Dict

from freqtrade import constants
from freqtrade.configuration import (TimeRange, remove_credentials,
                                     validate_config_consistency)
from freqtrade.edge import Edge
from freqtrade.optimize.optimize_reports import generate_edge_table
from freqtrade.resolvers import ExchangeResolver, StrategyResolver

logger = logging.getLogger(__name__)


class EdgeCli:
    """
    EdgeCli class, this class contains all the logic to run edge backtesting

    To run a edge backtest:
    edge = EdgeCli(config)
    edge.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Reset keys for edge
        remove_credentials(self.config)
        self.config['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT
        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)
        self.strategy = StrategyResolver.load_strategy(self.config)

        validate_config_consistency(self.config)

        self.edge = Edge(config, self.exchange, self.strategy)
        # Set refresh_pairs to false for edge-cli (it must be true for edge)
        self.edge._refresh_pairs = False

        self.edge._timerange = TimeRange.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))

    def start(self) -> None:
        result = self.edge.calculate()
        if result:
            print('')  # blank line for readability
            print(generate_edge_table(self.edge._cached_pairs))
