# Analyzing bot data

After performing backtests, or after running the bot for some time, it will be interresting to analyze the results your bot generated.

A good way for this is using Jupyter (notebook or lab) - which provides an interactive environment to analyze the data.

The following helpers will help you loading the data into Pandas DataFrames, and may also give you some starting points in analyzing the results.

## Backtesting

To analyze your backtest results, you can [export the trades](#exporting-trades-to-file).
You can then load the trades to perform further analysis.

A good way for this is using Jupyter (notebook or lab) - which provides an interactive environment to analyze the data.

Freqtrade provides an easy to load the backtest results, which is `load_backtest_data` - and takes a path to the backtest-results file.

``` python
from freqtrade.data.btanalysis import load_backtest_data
df = load_backtest_data("user_data/backtest-result.json")

# Show value-counts per pair
df.groupby("pair")["sell_reason"].value_counts()

```

This will allow you to drill deeper into your backtest results, and perform analysis which would make the regular backtest-output unreadable.

If you have some ideas for interesting / helpful backtest data analysis ideas, please submit a PR so the community can benefit from it.

## Live data

To analyze the trades your bot generated, you can load them to a DataFrame as follwos:

``` python
from freqtrade.data.btanalysis import load_trades_from_db

df = load_trades_from_db("sqlite:///tradesv3.sqlite")

df.groupby("pair")["sell_reason"].value_counts()

```

Feel free to submit an issue or Pull Request if you would like to share ideas on how to best analyze the data.
