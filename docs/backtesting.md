# Backtesting
This page explains how to validate your strategy performance by using 
Backtesting.

## Table of Contents
- [Test your strategy with Backtesting](#test-your-strategy-with-backtesting)
- [Understand the backtesting result](#understand-the-backtesting-result)

## Test your strategy with Backtesting
Now you have good Buy and Sell strategies, you want to test it against
real data. This is what we call 
[backtesting](https://en.wikipedia.org/wiki/Backtesting).


Backtesting will use the crypto-currencies (pair) from your config file
and load static tickers located in 
[/freqtrade/tests/testdata](https://github.com/gcarq/freqtrade/tree/develop/freqtrade/tests/testdata).  
If the 5 min and 1 min ticker for the crypto-currencies to test is not 
already in the `testdata` folder, backtesting will download them 
automatically. Testdata files will not be updated until you specify it.

The result of backtesting will confirm you if your bot as more chance to
make a profit than a loss.


The backtesting is very easy with freqtrade.

### Run a backtesting against the currencies listed in your config file
**With 5 min tickers (Per default)**
```bash
python3 ./freqtrade/main.py backtesting --realistic-simulation
```

**With 1 min tickers**
```bash
python3 ./freqtrade/main.py backtesting --realistic-simulation --ticker-interval 1
```

**Reload your testdata files**
```bash
python3 ./freqtrade/main.py backtesting --realistic-simulation --refresh-pairs-cached
```

**With live data (do not alter your testdata files)**
```bash
python3 ./freqtrade/main.py backtesting --realistic-simulation --live
```

**Using a different on-disk ticker-data source**
```bash
python3 ./freqtrade/main.py backtesting --datadir freqtrade/tests/testdata-20180101
```

**With a (custom) strategy file**
```bash
python3 ./freqtrade/main.py -s currentstrategy backtesting
```
Where `-s currentstrategy` refers to a filename `currentstrategy.py` in `freqtrade/user_data/strategies`

**Exporting trades to file**
```bash
python3 ./freqtrade/main.py backtesting --export trades
```

**Running backtest with smaller testset**  
Use the `--timerange` argument to change how much of the testset
you want to use. The last N ticks/timeframes will be used.

Example:
```bash
python3 ./freqtrade/main.py backtesting --timerange=-200
```

***Advanced use of timerange***  
Doing `--timerange=-200` will get the last 200 timeframes
from your inputdata. You can also specify specific dates,
or a range span indexed by start and stop.

The full timerange specification:
- Use last 123 tickframes of data: `--timerange=-123`
- Use first 123 tickframes of data: `--timerange=123-`
- Use tickframes from line 123 through 456: `--timerange=123-456`


Incoming feature, not implemented yet:
- `--timerange=-20180131`
-  `--timerange=20180101-`
- `--timerange=20180101-20181231`


**Update testdata directory**
To update your testdata directory, or download into another testdata directory:
```bash
mkdir -p user_data/data/testdata-20180113
cp freqtrade/tests/testdata/pairs.json user_data/data-20180113
cd user_data/data-20180113
```

Possibly edit pairs.json file to include/exclude pairs

```bash
python3 freqtrade/tests/testdata/download_backtest_data.py -p pairs.json
```

The script will read your pairs.json file, and download ticker data
into the current working directory.


For help about backtesting usage, please refer to 
[Backtesting commands](#backtesting-commands).

## Understand the backtesting result
The most important in the backtesting is to understand the result.

A backtesting result will look like that:
```
====================== BACKTESTING REPORT ================================
pair        buy count    avg profit %    total profit BTC    avg duration
--------  -----------  --------------  ------------------  --------------
BTC_ETH            56           -0.67         -0.00075455            62.3
BTC_LTC            38           -0.48         -0.00036315            57.9
BTC_ETC            42           -1.15         -0.00096469            67.0
BTC_DASH           72           -0.62         -0.00089368            39.9
BTC_ZEC            45           -0.46         -0.00041387            63.2
BTC_XLM            24           -0.88         -0.00041846            47.7
BTC_NXT            24            0.68          0.00031833            40.2
BTC_POWR           35            0.98          0.00064887            45.3
BTC_ADA            43           -0.39         -0.00032292            55.0
BTC_XMR            40           -0.40         -0.00032181            47.4
TOTAL             419           -0.41         -0.00348593            52.9
```

The last line will give you the overall performance of your strategy,
here:
```
TOTAL             419           -0.41         -0.00348593            52.9
```

We understand the bot has made `419` trades for an average duration of
`52.9` min, with a performance of `-0.41%` (loss), that means it has 
lost a total of `-0.00348593 BTC`.
 
As you will see your strategy performance will be influenced by your buy 
strategy, your sell strategy, and also by the `minimal_roi` and 
`stop_loss` you have set. 

As for an example if your minimal_roi is only `"0":  0.01`. You cannot
expect the bot to make more profit than 1% (because it will sell every 
time a trade will reach 1%).
```json
"minimal_roi": {
    "0":  0.01
},
```

On the other hand, if you set a too high `minimal_roi` like `"0":  0.55`
(55%), there is a lot of chance that the bot will never reach this 
profit. Hence, keep in mind that your performance is a mix of your 
strategies, your configuration, and the crypto-currency you have set up.

## Next step
Great, your strategy is profitable. What if the bot can give your the
optimal parameters to use for your strategy?  
Your next step is to learn [how to find optimal parameters with Hyperopt](https://github.com/gcarq/freqtrade/blob/develop/docs/hyperopt.md)
