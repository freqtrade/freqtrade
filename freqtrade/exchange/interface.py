from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Exchange(ABC):
    @property
    def name(self) -> str:
        """
        Name of the exchange.
        :return: str representation of the class name
        """
        return self.__class__.__name__

    @property
    def fee(self) -> float:
        """
        Fee for placing an order
        :return: percentage in float
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
    def get_ticker(self, pair: str, refresh: Optional[bool] = True) -> dict:
        """
        Gets ticker for given pair.
        :param pair: Pair as str, format: BTC_ETC
        :param refresh: Shall we query a new value or a cached value is enough
        :return: dict, format: {
            'bid': float,
            'ask': float,
            'last': float
        }
        """

    @abstractmethod
    def get_ticker_history(self, pair: str, tick_interval: int) -> List[Dict]:
        """
        Gets ticker history for given pair.
        :param pair: Pair as str, format: BTC_ETC
        :param tick_interval: ticker interval in minutes
        :return: list, format: [
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
        """

    def get_order(self, order_id: str) -> Dict:
        """
        Get order details for the given order_id.
        :param order_id: ID as str
        :return: dict, format: {
            'id': str,
            'type': str,
            'pair': str,
            'opened': str ISO 8601 datetime,
            'closed': str ISO 8601 datetime,
            'rate': float,
            'amount': float,
            'remaining': int
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

    @abstractmethod
    def get_market_summaries(self) -> List[Dict]:
        """
        Returns a 24h market summary for all available markets
        :return: list, format: [
            {
                'MarketName': str,
                'High': float,
                'Low': float,
                'Volume': float,
                'Last': float,
                'TimeStamp': datetime,
                'BaseVolume': float,
                'Bid': float,
                'Ask': float,
                'OpenBuyOrders': int,
                'OpenSellOrders': int,
                'PrevDay': float,
                'Created': datetime
            },
            ...
        ]
        """

    @abstractmethod
    def get_wallet_health(self) -> List[Dict]:
        """
        Returns a list of all wallet health information
        :return: list, format: [
            {
                'Currency': str,
                'IsActive': bool,
                'LastChecked': str,
                'Notice': str
            },
            ...
        """
