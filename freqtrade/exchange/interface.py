from abc import ABC, abstractmethod
from typing import List, Optional

import arrow


class Exchange(ABC):
    @property
    def name(self) -> str:
        """
        Name of the exchange.
        :return: str representation of the class name
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def sleep_time(self) -> float:
        """
        Sleep time in seconds for the main loop to avoid API rate limits.
        :return: float
        """

    @abstractmethod
    def buy(self, pair: str, rate: float, amount: float) -> str:
        """
        Places a limit buy order.
        :param pair: Pair as str, format: BTC_ETH
        :param rate: Rate limit for order
        :param amount: The amount to purchase
        :return: order_id of the placed buy order
        """

    @abstractmethod
    def sell(self, pair: str, rate: float, amount: float) -> str:
        """
        Places a limit sell order.
        :param pair: Pair as str, format: BTC_ETH
        :param rate: Rate limit for order
        :param amount: The amount to sell
        :return: order_id of the placed sell order
        """

    @abstractmethod
    def get_balance(self, currency: str) -> float:
        """
        Gets account balance.
        :param currency: Currency as str, format: BTC
        :return: float
        """

    @abstractmethod
    def get_balances(self) -> List[dict]:
        """
        Gets account balances across currencies
        :return: List of dicts, format: [
          {
            'Currency': str,
            'Balance': float,
            'Available': float,
            'Pending': float,
          }
          ...
        ]
        """

    @abstractmethod
    def get_ticker(self, pair: str) -> dict:
        """
        Gets ticker for given pair.
        :param pair: Pair as str, format: BTC_ETC
        :return: dict, format: {
            'bid': float,
            'ask': float,
            'last': float
        }
        """

    @abstractmethod
    def get_ticker_history(self, pair: str, minimum_date: Optional[arrow.Arrow] = None) -> dict:
        """
        Gets ticker history for given pair.
        :param pair: Pair as str, format: BTC_ETC
        :param minimum_date: Minimum date (optional)
        :return: dict, format: {
            'success': bool,
            'message': str,
            'result': [
                {
                    'O': float,       (Open)
                    'H': float,       (High)
                    'L': float,       (Low)
                    'C': float,       (Close)
                    'V': float,       (Volume)
                    'T': datetime,    (Time)
                    'BV': float,      (Base Volume)
                },
                ...
            ]
        }
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """
        Cancels order for given order_id.
        :param order_id: ID as str
        :return: None
        """

    @abstractmethod
    def get_open_orders(self, pair: str) -> List[dict]:
        """
        Gets all open orders for given pair.
        :param pair: Pair as str, format: BTC_ETC
        :return: List of dicts, format: [
            {
                'id': str,
                'type': str,
                'opened': datetime,
                'rate': float,
                'amount': float,
                'remaining': int,
            },
            ...
        ]
        """

    @abstractmethod
    def get_pair_detail_url(self, pair: str) -> str:
        """
        Returns the market detail url for the given pair.
        :param pair: Pair as str, format: BTC_ETC
        :return: URL as str
        """

    @abstractmethod
    def get_markets(self) -> List[str]:
        """
        Returns all available markets.
        :return: List of all available pairs
        """
