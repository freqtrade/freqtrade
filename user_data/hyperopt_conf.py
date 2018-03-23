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
        'max_open_trades': 15,
        'stake_currency': 'BTC',
        'stake_amount': 0.00075,
        'ticker_interval': 5,
        "bid_strategy": {
            "ask_last_balance": 0.0
        },
        "exchange": {
            "pair_whitelist": [
            'BTC_ETH',
            'BTC_XRP',
            'BTC_BCC',
            'BTC_LTC',
            'BTC_ADA',
            'BTC_XMR',
            'BTC_DASH',
            'BTC_TRX',
            'BTC_ETC',
            'BTC_ZEC',
            'BTC_WAVES',
            'BTC_STEEM',
            'BTC_STRAT',
            'BTC_DCR',
            'BTC_REP',
            'BTC_SNT',
            'BTC_KMD',
            'BTC_ARK',
            'BTC_ARDR',
            'BTC_MONA',
            'BTC_DGB',
            'BTC_PIVX',
            'BTC_SYS',
            'BTC_FCT',
            'BTC_BAT',
            'BTC_GNT',
            'BTC_XZC',
            'BTC_EMC',
            'BTC_NXT',
            'BTC_SALT',
            'BTC_PAY',
            'BTC_PART',
            'BTC_GBYTE',
            'BTC_BNT',
            'BTC_POWR',
            'BTC_NXS',
            'BTC_SRN'
            ]
        }
    }
