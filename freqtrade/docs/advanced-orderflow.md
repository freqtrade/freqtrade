# Orderflow data

This guide walks you through utilizing public trade data for advanced orderflow analysis in Freqtrade.

!!! Warning "Experimental Feature"
    The orderflow feature is currently in beta and may be subject to changes in future releases. Please report any issues or feedback on the [Freqtrade GitHub repository](https://github.com/freqtrade/freqtrade/issues).
    It's also currently not been tested with freqAI - and combining these two features is considered out of scope at this point.

!!! Warning "Performance"
    Orderflow requires raw trades data. This data is rather large, and can cause a slow initial startup, when freqtrade needs to download the trades data for the last X candles. Additionally, enabling this feature will cause increased memory usage. Please ensure to have sufficient resources available.

## Getting Started

### Enable Public Trades

In your `config.json` file, set the `use_public_trades` option to true under the `exchange` section.

```json
"exchange": {
   ...
   "use_public_trades": true,
}
```

### Configure Orderflow Processing

Define your desired settings for orderflow processing within the orderflow section of config.json. Here, you can adjust factors like:

- `cache_size`: How many previous orderflow candles are saved into cache instead of calculated every new candle
- `max_candles`: Filter how many candles would you like to get trades data for.
- `scale`: This controls the price bin size for the footprint chart.
- `stacked_imbalance_range`: Defines the minimum consecutive imbalanced price levels required for consideration.
- `imbalance_volume`: Filters out imbalances with volume below this threshold.
- `imbalance_ratio`: Filters out imbalances with a ratio (difference between ask and bid volume) lower than this value.

```json
"orderflow": {
    "cache_size": 1000, 
    "max_candles": 1500, 
    "scale": 0.5, 
    "stacked_imbalance_range": 3, //  needs at least this amount of imbalance next to each other
    "imbalance_volume": 1, //  filters out below
    "imbalance_ratio": 3 //  filters out ratio lower than
  },
```

## Downloading Trade Data for Backtesting

To download historical trade data for backtesting, use the --dl-trades flag with the freqtrade download-data command.

```bash
freqtrade download-data -p BTC/USDT:USDT --timerange 20230101- --trading-mode futures --timeframes 5m --dl-trades
```

!!! Warning "Data availability"
    Not all exchanges provide public trade data. For supported exchanges, freqtrade will warn you if public trade data is not available if you start downloading data with the `--dl-trades` flag.

## Accessing Orderflow Data

Once activated, several new columns become available in your dataframe:

``` python

dataframe["trades"] # Contains information about each individual trade.
dataframe["orderflow"] # Represents a footprint chart dict (see below)
dataframe["imbalances"] # Contains information about imbalances in the order flow.
dataframe["bid"] # Total bid volume 
dataframe["ask"] # Total ask volume
dataframe["delta"] # Difference between ask and bid volume.
dataframe["min_delta"] # Minimum delta within the candle
dataframe["max_delta"] # Maximum delta within the candle
dataframe["total_trades"] # Total number of trades
dataframe["stacked_imbalances_bid"] # List of price levels of stacked bid imbalance range beginnings
dataframe["stacked_imbalances_ask"] # List of price levels of stacked ask imbalance range beginnings
```

You can access these columns in your strategy code for further analysis. Here's an example:

``` python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Calculating cumulative delta
    dataframe["cum_delta"] = cumulative_delta(dataframe["delta"])
    # Accessing total trades
    total_trades = dataframe["total_trades"]
    ...

def cumulative_delta(delta: Series):
    cumdelta = delta.cumsum()
    return cumdelta

```

### Footprint chart (`dataframe["orderflow"]`)

This column provides a detailed breakdown of buy and sell orders at different price levels, offering valuable insights into order flow dynamics. The `scale` parameter in your configuration determines the price bin size for this representation

The `orderflow` column contains a dict with the following structure:

``` output
{
    "price": {
        "bid_amount": 0.0,
        "ask_amount": 0.0,
        "bid": 0,
        "ask": 0,
        "delta": 0.0,
        "total_volume": 0.0,
        "total_trades": 0
    }
}
```

#### Orderflow column explanation

- key: Price bin - binned at `scale` intervals
- `bid_amount`: Total volume bought at each price level.
- `ask_amount`: Total volume sold at each price level.
- `bid`: Number of buy orders at each price level.
- `ask`: Number of sell orders at each price level.
- `delta`: Difference between ask and bid volume at each price level.
- `total_volume`: Total volume (ask amount + bid amount) at each price level.
- `total_trades`: Total number of trades (ask + bid) at each price level.

By leveraging these features, you can gain valuable insights into market sentiment and potential trading opportunities based on order flow analysis.

### Raw trades data (`dataframe["trades"]`)

List with the individual trades that occurred during the candle. This data can be used for more granular analysis of order flow dynamics.

Each individual entry contains a dict with the following keys:

- `timestamp`: Timestamp of the trade.
- `date`: Date of the trade.
- `price`: Price of the trade.
- `amount`: Volume of the trade.
- `side`: Buy or sell.
- `id`: Unique identifier for the trade.
- `cost`: Total cost of the trade (price * amount).

### Imbalances (`dataframe["imbalances"]`)

This column provides a dict with information about imbalances in the order flow. An imbalance occurs when there is a significant difference between the ask and bid volume at a given price level.

Each row looks as follows - with price as index, and the corresponding bid and ask imbalance values as columns

``` output
{
    "price": {
        "bid_imbalance": False,
        "ask_imbalance": False
    }
}
```
