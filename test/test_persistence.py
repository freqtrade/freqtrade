import unittest
from unittest.mock import patch

from exchange import Exchange
from persistence import Trade


class TestTrade(unittest.TestCase):
    def test_1_exec_sell_order(self):
        with patch('main.exchange.sell', side_effect='mocked_order_id') as api_mock:
            trade = Trade(
                pair='BTC_ETH',
                btc_amount=1.00,
                open_rate=0.50,
                amount=10.00,
                exchange=Exchange.BITTREX,
                open_order_id='mocked'
            )
            profit = trade.exec_sell_order(1.00, 10.00)
            api_mock.assert_called_once_with('BTC_ETH', 1.0, 10.0)
            self.assertEqual(profit, 100.0)
            self.assertEqual(trade.close_rate, 1.0)
            self.assertEqual(trade.close_profit, profit)
            self.assertIsNotNone(trade.close_date)


if __name__ == '__main__':
    unittest.main()
