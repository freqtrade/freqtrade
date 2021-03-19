from pathlib import Path
from freqtrade.configuration import Configuration
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import load_pair_history
from freqtrade.resolvers import StrategyResolver
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.data.btanalysis import load_trades_from_db
from freqtrade.data.btanalysis import analyze_trade_parallelism
from freqtrade.plot.plotting import generate_candlestick_graph

import matplotlib

# # Customize these according to your needs.
#
# # Initialize empty configuration object
from freqtrade.strategy import IStrategy

config = Configuration.from_files(["user_data/config_ltcusdt_1h.json"])
# # Optionally, use existing configuration file
# config = Configuration.from_files(["config.json"])
#
# # Define some constants
# config["timeframe"] = "1m"
# # Name of the strategy class
config["strategy"] = "ltcusdt_1h"
# # Location of the data
data_location = Path(config['user_data_dir'], 'data', 'binance')
# # Pair to analyze - Only use one pair here
pair = "LTC/USDT"
#
# # Load data using values set above
#
candles = load_pair_history(datadir=data_location,
                            timeframe=config["timeframe"],
                            pair=pair)
#
# # Confirm success
# print("Loaded " + str(len(candles)) + f" rows of data for {pair} from {data_location}")
#
# # Load strategy using values set above
#
dataprovider = DataProvider(config, config['exchange'])
IStrategy.dp = dataprovider
strategy = StrategyResolver.load_strategy(config)
#
# # Generate buy/sell signals using strategy
df = strategy.analyze_ticker(candles, {'pair': pair})
#
# col = df.columns
#
# # Report results
# print(f"Generated {df['buy'].sum()} buy signals")
data = df.set_index('date', drop=False)


# # if backtest_dir points to a directory, it'll automatically load the last backtest file.
# stats = load_backtest_stats(backtest_dir)
# # You can get the full backtest statistics by using the following command.
# # This contains all information used to generate the backtest result.
#
# strategy = config["strategy"]
# # All statistics are available per strategy, so if `--strategy-list` was used during backtest, this will be reflected here as well.
# # Example usages:
# print(stats['strategy'][strategy]['results_per_pair'])
# # Get pairlist used for this backtest
# print(stats['strategy'][strategy]['pairlist'])
# # Get market change (average change of all pairs from start to end of the backtest period)
# print(stats['strategy'][strategy]['market_change'])
# # Maximum drawdown ()
# print(stats['strategy'][strategy]['max_drawdown'])
# # Maximum drawdown start and end
# print(stats['strategy'][strategy]['drawdown_start'])
# print(stats['strategy'][strategy]['drawdown_end'])
#
# # Get strategy comparison (only relevant if multiple strategies were compared)
# print(stats['strategy_comparison'])
#
# # Load backtested trades as dataframe
# print(f"backtest {backtest_dir}")
# trades = load_backtest_data(backtest_dir)

# # Show value-counts per pair
# trades.groupby("pair")["sell_reason"].value_counts()

# Fetch trades from database


# Display results
# trades.groupby("pair")["sell_reason"].value_counts()


def plot_db(path):
    trades = load_trades_from_db(path)
    # Analyze the above
    parallel_trades = analyze_trade_parallelism(trades, '1m')
    parallel_trades.plot()
    trades_red = trades.loc[trades['pair'] == pair]
    # Limit graph period to keep plotly quick and reactive
    # Filter trades to one pair
    data_red = data['2021-01-15':'2021-01-23']
    # Generate candlestick graph
    graph = generate_candlestick_graph(pair=pair,
                                       data=data_red,
                                       trades=trades_red,
                                       indicators1=['tsf_mid'],
                                       indicators2=['correl_tsf_mid_close', 'correl_angle_short_close',
                                                    'correl_angle_long_close', 'correl_hist_close']
                                       )
    graph.show()


def plot_backtest(start_date, stop_date):
    backtest_dir = config["user_data_dir"] / "backtest_results"
    strategy = config["strategy"]

    trades = load_backtest_data(backtest_dir)
    stats = load_backtest_stats(backtest_dir)
    parallel_trades = analyze_trade_parallelism(trades, '1h')
    parallel_trades = parallel_trades[start_date:stop_date]
    parallel_trades.plot()
    # All statistics are available per strategy, so if `--strategy-list` was used during backtest, this will be reflected here as well.
    # Example usages:
    print(stats['strategy'][strategy]['results_per_pair'])
    # Get pairlist used for this backtest
    print(stats['strategy'][strategy]['pairlist'])
    # Get market change (average change of all pairs from start to end of the backtest period)
    print(stats['strategy'][strategy]['market_change'])
    # Maximum drawdown ()
    print(stats['strategy'][strategy]['max_drawdown'])
    # Maximum drawdown start and end
    print(stats['strategy'][strategy]['drawdown_start'])
    print(stats['strategy'][strategy]['drawdown_end'])

    # Get strategy comparison (only relevant if multiple strategies were compared)
    print(stats['strategy_comparison'])

    # Load backtested trades as dataframe
    print(f"backtest {backtest_dir}")
    # Show value-counts per pair
    trades.groupby("pair")["sell_reason"].value_counts()

    # trades_red = trades[start_date:stop_date]

    # Limit graph period to keep plotly quick and reactive
    # Filter trades to one pair
    data_red = data[start_date:stop_date]
    # Generate candlestick graph
    graph = generate_candlestick_graph(pair=pair,
                                       data=data_red,
                                       trades=trades,
                                       indicators1=['sar', 'tsf_mid'],
                                       indicators2=['sine', 'leadsine', 'angle_trend_mid', 'inphase', 'quadrature' ]
                                       )

    graph.show()


plot_backtest(start_date='2020-01-01', stop_date='2021-03-18')

