from math import isclose

import pytest

from freqtrade.enums import TradingMode, Collateral
from freqtrade.leverage import liquidation_price


@pytest.mark.parametrize(
    'exchange_name, open_rate, is_short, leverage, trading_mode, collateral, wallet_balance, maintenance_margin_ex_1, '
    'unrealized_pnl_ex_1, maintenance_amount, position_1, entry_price_1, maintenance_margin_rate, '
    'expected',
    [
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.ISOLATED, 1535443.01, 71200.81144, -56354.57,
         135365.00, 3683.979, 1456.84, 0.10, 1114.78),
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.ISOLATED, 1535443.01, 356512.508, -448192.89,
         16300.000, 109.488, 32481.980, 0.025, 18778.73),
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.CROSS, 1535443.01, 71200.81144, -56354.57, 135365.00,
         3683.979, 1456.84, 0.10, 1153.26),
        ("binance", 0.0, False, 1, TradingMode.FUTURES, Collateral.CROSS, 1535443.01, 356512.508, -448192.89, 16300.000,
         109.488, 32481.980, 0.025, 26316.89)
    ])
def test_liquidation_price(exchange_name, open_rate, is_short, leverage, trading_mode, collateral, wallet_balance,
                           maintenance_margin_ex_1, unrealized_pnl_ex_1, maintenance_amount, position_1,
                           entry_price_1, maintenance_margin_rate, expected):
    assert isclose(round(liquidation_price(
        exchange_name=exchange_name,
        open_rate=open_rate,
        is_short=is_short,
        leverage=leverage,
        trading_mode=trading_mode,
        collateral=collateral,
        wallet_balance=wallet_balance,
        maintenance_margin_ex_1=maintenance_margin_ex_1,
        unrealized_pnl_ex_1=unrealized_pnl_ex_1,
        maintenance_amount=maintenance_amount,
        position_1=position_1,
        entry_price_1=entry_price_1,
        maintenance_margin_rate=maintenance_margin_rate
    ), 2), expected)
