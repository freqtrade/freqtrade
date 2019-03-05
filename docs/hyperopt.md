# Hyperopt

This page explains how to tune your strategy by finding the optimal
parameters, a process called hyperparameter optimization. The bot uses several
algorithms included in the `scikit-optimize` package to accomplish this. The
search will burn all your CPU cores, make your laptop sound like a fighter jet
and still take a long time.

!!! Bug
    Hyperopt will crash when used with only 1 CPU Core as found out in [Issue #1133](https://github.com/freqtrade/freqtrade/issues/1133)

## Prepare Hyperopting

Before we start digging into Hyperopt, we recommend you to take a look at
an example hyperopt file located into [user_data/hyperopts/](https://github.com/freqtrade/freqtrade/blob/develop/user_data/hyperopts/test_hyperopt.py)

Configuring hyperopt is similar to writing your own strategy, and many tasks will be similar and a lot of code can be copied across from the strategy.

### Checklist on all tasks / possibilities in hyperopt

Depending on the space you want to optimize, only some of the below are required.

* fill `populate_indicators` - probably a copy from your strategy
* fill `buy_strategy_generator` - for buy signal optimization
* fill `indicator_space` - for buy signal optimzation
* fill `sell_strategy_generator` - for sell signal optimization
* fill `sell_indicator_space` - for sell signal optimzation
* fill `roi_space` - for ROI optimization
* fill `generate_roi_table` - for ROI optimization (if you need more than 3 entries)
* fill `stoploss_space` - stoploss optimization
* Optional but recommended
  * copy `populate_buy_trend` from your strategy - otherwise default-strategy will be used
  * copy `populate_sell_trend` from your strategy - otherwise default-strategy will be used

### 1. Install a Custom Hyperopt File

Put your hyperopt file into the folder`user_data/hyperopts`.

Let assume you want a hyperopt file `awesome_hyperopt.py`:  
Copy the file `user_data/hyperopts/sample_hyperopt.py` into `user_data/hyperopts/awesome_hyperopt.py`

### 2. Configure your Guards and Triggers

There are two places you need to change in your hyperopt file to add a new buy hyperopt for testing:

- Inside `indicator_space()` - the parameters hyperopt shall be optimizing.
- Inside `populate_buy_trend()` - applying the parameters.

There you have two different types of indicators: 1. `guards` and 2. `triggers`.

1. Guards are conditions like "never buy if ADX < 10", or never buy if current price is over EMA10.
2. Triggers are ones that actually trigger buy in specific moment, like "buy when EMA5 crosses over EMA10" or "buy when close price touches lower bollinger band".

Hyperoptimization will, for each eval round, pick one trigger and possibly
multiple guards. The constructed strategy will be something like
"*buy exactly when close price touches lower bollinger band, BUT only if
ADX > 10*".

If you have updated the buy strategy, ie. changed the contents of
`populate_buy_trend()` method you have to update the `guards` and
`triggers` hyperopts must use.

#### Sell optimization

Similar to the buy-signal above, sell-signals can also be optimized. 
Place the corresponding settings into the following methods

* Inside `sell_indicator_space()` - the parameters hyperopt shall be optimizing.
* Inside `populate_sell_trend()` - applying the parameters.

The configuration and rules are the same than for buy signals.
To avoid naming collisions in the search-space, please prefix all sell-spaces with `sell-`.

## Solving a Mystery

Let's say you are curious: should you use MACD crossings or lower Bollinger
Bands to trigger your buys. And you also wonder should you use RSI or ADX to
help with those buy decisions. If you decide to use RSI or ADX, which values
should I use for them? So let's use hyperparameter optimization to solve this
mystery.

We will start by defining a search space:

```python
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching strategy parameters
        """
        return [
            Integer(20, 40, name='adx-value'),
            Integer(20, 40, name='rsi-value'),
            Categorical([True, False], name='adx-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['bb_lower', 'macd_cross_signal'], name='trigger')
        ]
```

Above definition says: I have five parameters I want you to randomly combine
to find the best combination. Two of them are integer values (`adx-value`
and `rsi-value`) and I want you test in the range of values 20 to 40.
Then we have three category variables. First two are either `True` or `False`.
We use these to either enable or disable the ADX and RSI guards. The last
one we call `trigger` and use it to decide which buy trigger we want to use.

So let's write the buy strategy using these values:

``` python
        def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
            conditions = []
            # GUARDS AND TRENDS
            if 'adx-enabled' in params and params['adx-enabled']:
                conditions.append(dataframe['adx'] > params['adx-value'])
            if 'rsi-enabled' in params and params['rsi-enabled']:
                conditions.append(dataframe['rsi'] < params['rsi-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'bb_lower':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['macd'], dataframe['macdsignal']
                    ))

            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

            return dataframe

        return populate_buy_trend
```

Hyperopting will now call this `populate_buy_trend` as many times you ask it (`epochs`)
with different value combinations. It will then use the given historical data and make
buys based on the buy signals generated with the above function and based on the results
it will end with telling you which paramter combination produced the best profits.

The search for best parameters starts with a few random combinations and then uses a
regressor algorithm (currently ExtraTreesRegressor) to quickly find a parameter combination
that minimizes the value of the objective function `calculate_loss` in `hyperopt.py`.

The above setup expects to find ADX, RSI and Bollinger Bands in the populated indicators.
When you want to test an indicator that isn't used by the bot currently, remember to
add it to the `populate_indicators()` method in `hyperopt.py`.

## Execute Hyperopt

Once you have updated your hyperopt configuration you can run it.
Because hyperopt tries a lot of combinations to find the best parameters it will take time you will have the result (more than 30 mins).

We strongly recommend to use `screen` or `tmux` to prevent any connection loss.

```bash
python3 ./freqtrade/main.py -c config.json hyperopt --customhyperopt <hyperoptname> -e 5000 --spaces all
```

Use  `<hyperoptname>` as the name of the custom hyperopt used.

The `-e` flag will set how many evaluations hyperopt will do. We recommend
running at least several thousand evaluations.

The `--spaces all` flag determines that all possible parameters should be optimized. Possibilities are listed below.

!!! Warning
When switching parameters or changing configuration options, the file `user_data/hyperopt_results.pickle` should be removed. It's used to be able to continue interrupted calculations, but does not detect changes to settings or the hyperopt file.

### Execute Hyperopt with Different Ticker-Data Source

If you would like to hyperopt parameters using an alternate ticker data that
you have on-disk, use the `--datadir PATH` option. Default hyperopt will
use data from directory `user_data/data`.

### Running Hyperopt with Smaller Testset

Use the `--timerange` argument to change how much of the testset
you want to use. The last N ticks/timeframes will be used.
Example:

```bash
python3 ./freqtrade/main.py hyperopt --timerange -200
```

### Running Hyperopt with Smaller Search Space

Use the `--spaces` argument to limit the search space used by hyperopt.
Letting Hyperopt optimize everything is a huuuuge search space. Often it
might make more sense to start by just searching for initial buy algorithm.
Or maybe you just want to optimize your stoploss or roi table for that awesome
new buy strategy you have.

Legal values are:

- `all`: optimize everything
- `buy`: just search for a new buy strategy
- `sell`: just search for a new sell strategy
- `roi`: just optimize the minimal profit table for your strategy
- `stoploss`: search for the best stoploss value
- space-separated list of any of the above values for example `--spaces roi stoploss`

## Understand the Hyperopt Result

Once Hyperopt is completed you can use the result to create a new strategy.
Given the following result from hyperopt:

```
Best result:
   135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722Σ%). Avg duration 180.4 mins.
with values:
{    'adx-value': 44,
     'rsi-value': 29,
     'adx-enabled': False,
     'rsi-enabled': True,
     'trigger': 'bb_lower'}
```

You should understand this result like:

- The buy trigger that worked best was `bb_lower`.
- You should not use ADX because `adx-enabled: False`)
- You should **consider** using the RSI indicator (`rsi-enabled: True` and the best value is `29.0` (`rsi-value: 29.0`)

You have to look inside your strategy file into `buy_strategy_generator()`
method, what those values match to.

So for example you had `rsi-value: 29.0` so we would look at `rsi`-block, that translates to the following code block:

```
(dataframe['rsi'] < 29.0)
```

Translating your whole hyperopt result as the new buy-signal
would then look like:

```python
def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['rsi'] < 29.0) &  # rsi-value
            dataframe['close'] < dataframe['bb_lowerband']  # trigger
        ),
        'buy'] = 1
    return dataframe
```

### Understand Hyperopt ROI results

If you are optimizing ROI, you're result will look as follows and include a ROI table.

```
Best result:
   135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722Σ%). Avg duration 180.4 mins.
with values:
{   'adx-value': 44,
    'rsi-value': 29,
    'adx-enabled': false,
    'rsi-enabled': True,
    'trigger': 'bb_lower',
    'roi_t1': 40,
    'roi_t2': 57,
    'roi_t3': 21,
    'roi_p1': 0.03634636907306948,
    'roi_p2': 0.055237357937802885,
    'roi_p3': 0.015163796015548354,
    'stoploss': -0.37996664668703606
}
ROI table:
{   0: 0.10674752302642071,
    21: 0.09158372701087236,
    78: 0.03634636907306948,
    118: 0}
```

This would translate to the following ROI table:

``` python
 minimal_roi = {
        "118": 0,
        "78": 0.0363463,
        "21": 0.0915,
        "0": 0.106
    }
```

### Validate backtest result

Once the optimized strategy has been implemented into your strategy, you should backtest this strategy to make sure everything is working as expected.
To archive the same results (number of trades, ...) than during hyperopt, please use the command line flags `--disable-max-market-positions` and `--enable-position-stacking` for backtesting.

This configuration is the default in hyperopt for performance reasons.

You can overwrite position stacking in the configuration by explicitly setting `"position_stacking"=false` or by changing the relevant line in your hyperopt file [here](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/optimize/hyperopt.py#L191).

Enabling the market-position for hyperopt is currently not possible.

!!! Note:
Dry/live runs will **NOT** use position stacking - therefore it does make sense to also validate the strategy without this as it's closer to reality.

## Next Step

Now you have a perfect bot and want to control it from Telegram. Your
next step is to learn the [Telegram usage](telegram-usage.md).
