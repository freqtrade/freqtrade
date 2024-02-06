# Advanced Orderflow

This page explains some advanced tasks and configuration options that can be performed to use orderflow data by downloading public trade data.


## Quickstart

- enable using public trades in `config.json`
```
"exchange": {
   ...
   "use_public_trades": true,
}
```
- set orderflow processing configuration in `config.json`:
 ```
"orderflow": {
    "scale": 0.5, 
    "stacked_imbalance_range": 3, # needs at least this amount of imblance next to each other
    "imbalance_volume": 1, # filters out below
    "imbalance_ratio": 300 # filters out ratio lower than
  },
```

## Downloading data for backtesting

- use `--dl-trades` to fetch trades for timerange

For example
``` bash
freqtrade download-data -p BTC/USDT:USDT --timerange 20230101-  --trading-mode futures  --timeframes 5m --dl-trades
```


## Accessing orderflow data

Several new columns are available when activated.
``` python

    dataframe['trades']
    dataframe['orderflow']
    dataframe['bid']
    dataframe['ask']
    dataframe['delta']
    dataframe['min_delta']
    dataframe['max_delta']
    dataframe['total_trades']
    dataframe['stacked_imbalances_bid']
    dataframe['stacked_imbalances_ask']
```

These can be accessed like this:
``` python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # calculating cumulative delta
    dataframe['cum_delta'] = cumulative_delta(dataframe['delta'])

def cumulative_delta(delta: Series):
    cumdelta = delta.cumsum()
    return cumdelta

```
