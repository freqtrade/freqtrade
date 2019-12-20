# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the edge backtesting interface
"""
import logging
from typing import Any, Dict

from tabulate import tabulate

from freqtrade import constants
from freqtrade.configuration import (TimeRange, remove_credentials,
                                     validate_config_consistency)
from freqtrade.edge import Edge
from freqtrade.exchange import Exchange
from freqtrade.resolvers import StrategyResolver

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
        self.exchange = Exchange(self.config)
        self.strategy = StrategyResolver(self.config).strategy

        validate_config_consistency(self.config)

        self.edge = Edge(config, self.exchange, self.strategy)
        # Set refresh_pairs to false for edge-cli (it must be true for edge)
        self.edge._refresh_pairs = False

        self.edge._timerange = TimeRange.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))

    def _generate_edge_table(self, results: dict) -> str:

        floatfmt = ('s', '.10g', '.2f', '.2f', '.2f', '.2f', 'd', '.d')
        tabular_data = []
        headers = ['pair', 'stoploss', 'win rate', 'risk reward ratio',
                   'required risk reward', 'expectancy', 'total number of trades',
                   'average duration (min)']

        for result in results.items():
            if result[1].nb_trades > 0:
                tabular_data.append([
                    result[0],
                    result[1].stoploss,
                    result[1].winrate,
                    result[1].risk_reward_ratio,
                    result[1].required_risk_reward,
                    result[1].expectancy,
                    result[1].nb_trades,
                    round(result[1].avg_trade_duration)
                ])

        # Ignore type as floatfmt does allow tuples but mypy does not know that
        return tabulate(tabular_data, headers=headers,
                        floatfmt=floatfmt, tablefmt="pipe")  # type: ignore

    def start(self) -> None:
        result = self.edge.calculate()
        if result:
            print('')  # blank line for readability
            print(self._generate_edge_table(self.edge._cached_pairs))
