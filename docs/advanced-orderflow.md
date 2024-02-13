# Advanced Orderflow

This page explains some advanced tasks and configuration options that can be performed to use orderflow data by downloading public trade data.


## Quickstart

enable using public trades in `config.json`
```
"exchange": {
   ...
   "use_public_trades": true,
}
```
set orderflow processing configuration in `config.json`:
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

    dataframe['trades'] # every single trade
    dataframe['orderflow'] # footprint chart: see below
    dataframe['bid'] # bid sum
    dataframe['ask'] # ask sum
    dataframe['delta'] # ask - bid
    dataframe['min_delta'] # minimum delta reached within candle
    dataframe['max_delta'] # maximum delta reached within candle
    dataframe['total_trades'] # amount of trades
    dataframe['stacked_imbalances_bid'] # price level stacked imbalance bid occurred
    dataframe['stacked_imbalances_ask'] # price level stacked imbalance ask occurred
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
### dataframe['orderflow']

This includes a dataframe that represents a Footprint chart of the Bid vs Ask type. Footprint charts are a type of candlestick chart that provides additional information, such as trade volume and order flow, in addition to price.
The scale of the price is set by `orderflow.scale` (see above) and thus binned per price level.

Following columns are available:
```python

orderflow_df['bid_amount'] # how much bids were traded
orderflow_df['ask_amount'] # how much asks were traded
orderflow_df['bid'] # how many bids trades
orderflow_df['ask'] # how many asks trades
orderflow_df['delta'] # ask amount - bid amount
orderflow_df['total_volume'] # ask amount + bid amount
orderflow_df['total_trades'] # ask + bid trades 
```



