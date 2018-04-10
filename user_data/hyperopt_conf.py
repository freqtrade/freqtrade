"""
File that contains the configuration for Hyperopt
"""


def hyperopt_optimize_conf() -> dict:
    """
    This function is used to define which parameters Hyperopt must used.
    The "pair_whitelist" is only used is your are using Hyperopt with MongoDB,
    without MongoDB, Hyperopt will use the pair your have set in your config file.
    :return:
    """
    return {
        'max_open_trades': 3,
        'stake_currency': 'BTC',
        'stake_amount': 0.01,
        "minimal_roi": {
            '40': 0.0,
            '30': 0.01,
            '20': 0.02,
            '0': 0.04,
        },
        'stoploss': -0.10,
        "bid_strategy": {
            "ask_last_balance": 0.0
        },
        "exchange": {
            "name": "bittrex",
            "pair_whitelist": [
                "ETH/BTC",
                "LTC/BTC",
                "ETC/BTC",
                "DASH/BTC",
                "ZEC/BTC",
                "XLM/BTC",
                "NXT/BTC",
                "POWR/BTC",
                "ADA/BTC",
                "XMR/BTC"
            ]
        }
    }
