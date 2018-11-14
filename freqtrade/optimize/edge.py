# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
import operator
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from freqtrade.edge import Edge
from tabulate import tabulate

import freqtrade.optimize as optimize
from freqtrade import DependencyException, constants
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Exchange
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.resolver import IStrategy, StrategyResolver
import pdb

logger = logging.getLogger(__name__)


class EdgeCli(object):
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Reset keys for edge
        self.config['exchange']['key'] = ''
        self.config['exchange']['secret'] = ''
        self.config['exchange']['password'] = ''
        self.config['exchange']['uid'] = ''
        self.config['dry_run'] = True
        self.exchange = Exchange(self.config)
        self.strategy = StrategyResolver(self.config).strategy

        self.edge = Edge(config, self.exchange, self.strategy)
        self.edge._refresh_pairs = self.config.get('refresh_pairs', False)

    def _generate_edge_table(self, results: dict) -> str:

        floatfmt = ('s', '.10g', '.2f', '.2f', '.2f', '.2f', 'd', '.d')
        tabular_data = []
        headers = ['pair', 'stoploss', 'win rate', 'risk reward ratio',
                   'required risk reward', 'expectancy', 'total number of trades', 'average duration (min)']

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

        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt, tablefmt="pipe")

    def start(self) -> None:
        self.edge.calculate()
        print('')  # blank like for readability
        print(self._generate_edge_table(self.edge._cached_pairs))


def setup_configuration(args: Namespace) -> Dict[str, Any]:
    """
    Prepare the configuration for the backtesting
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args)
    config = configuration.get_config()

    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    return config


def start(args: Namespace) -> None:
    """
    Start Edge script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Initialize configuration
    config = setup_configuration(args)
    logger.info('Starting freqtrade in Edge mode')

    # Initialize Edge object
    edge_cli = EdgeCli(config)
    edge_cli.start()
