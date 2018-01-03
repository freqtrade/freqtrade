# Hyperopt
This page explains how to tune your strategy by finding the optimal
parameters with Hyperopt.

## Table of Contents
- [Prepare your Hyperopt](#prepare-hyperopt)
    - [1. Configure your Guards and Triggers](#1-configure-your-guards-and-triggers)
    - [2. Update the hyperopt config file](#2-update-the-hyperopt-config-file)
- [Advanced Hyperopt notions](#advanced-notions)
    - [Understand the Guards and Triggers](#understand-the-guards-and-triggers)
- [Execute Hyperopt](#execute-hyperopt)
    - [Hyperopt with MongoDB](#hyperopt-with-mongoDB)
- [Understand the hyperopts result](#understand-the-backtesting-result)

## Prepare Hyperopt
Before we start digging in Hyperopt, we recommend you to take a look at 
out Hyperopt file 
[freqtrade/optimize/hyperopt.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/optimize/hyperopt.py)
 
### 1. Configure your Guards and Triggers
There are two places you need to change to add a new buy strategy for 
testing:
- Inside the [populate_buy_trend()](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/optimize/hyperopt.py#L167-L207).
- Inside the [SPACE dict](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/optimize/hyperopt.py#L47-L94).

There you have two different type of indicators: 1. `guards` and 2. 
`triggers`.
1. Guards are conditions like "never buy if ADX < 10", or never buy if 
current price is over EMA10.
2. Triggers are ones that actually trigger buy in specific moment, like 
"buy when EMA5 crosses over EMA10" or buy when close price touches lower 
bollinger band.

HyperOpt will, for each eval round, pick just ONE trigger, and possibly 
multiple guards. So that the constructed strategy will be something like 
"*buy exactly when close price touches lower bollinger band, BUT only if 
ADX > 10*".


If you have updated the buy strategy, means change the content of
`populate_buy_trend()` function you have to update the `guards` and 
`triggers` hyperopts must used.

As for an example if your `populate_buy_trend()` function is:
```python
def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
    dataframe.loc[
        (dataframe['rsi'] < 35) &
        (dataframe['adx'] > 65),
        'buy'] = 1

    return dataframe
```

Your hyperopt file must contains `guards` to find the right value for 
`(dataframe['adx'] > 65)` & and `(dataframe['plus_di'] > 0.5)`. That 
means you will need to enable/disable triggers.
 
In our case the `SPACE` and `populate_buy_trend` in hyperopt.py file 
will be look like:
```python
SPACE = {
    'rsi': hp.choice('rsi', [
        {'enabled': False},
        {'enabled': True, 'value': hp.quniform('rsi-value', 20, 40, 1)}
    ]),
    'adx': hp.choice('adx', [
        {'enabled': False},
        {'enabled': True, 'value': hp.quniform('adx-value', 15, 50, 1)}
    ]),
    'trigger': hp.choice('trigger', [
        {'type': 'lower_bb'},
        {'type': 'faststoch10'},
        {'type': 'ao_cross_zero'},
        {'type': 'ema5_cross_ema10'},
        {'type': 'macd_cross_signal'},
        {'type': 'sar_reversal'},
        {'type': 'stochf_cross'},
        {'type': 'ht_sine'},
    ]),
}

... 

def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS
        if params['adx']['enabled']:
            conditions.append(dataframe['adx'] > params['adx']['value'])
        if params['rsi']['enabled']:
            conditions.append(dataframe['rsi'] < params['rsi']['value'])

        # TRIGGERS
        triggers = {
            'lower_bb': dataframe['tema'] <= dataframe['blower'],
            'faststoch10': (crossed_above(dataframe['fastd'], 10.0)),
            'ao_cross_zero': (crossed_above(dataframe['ao'], 0.0)),
            'ema5_cross_ema10': (crossed_above(dataframe['ema5'], dataframe['ema10'])),
            'macd_cross_signal': (crossed_above(dataframe['macd'], dataframe['macdsignal'])),
            'sar_reversal': (crossed_above(dataframe['close'], dataframe['sar'])),
            'stochf_cross': (crossed_above(dataframe['fastk'], dataframe['fastd'])),
            'ht_sine': (crossed_above(dataframe['htleadsine'], dataframe['htsine'])),
        }
        ...
```


### 2. Update the hyperopt config file
Hyperopt is using a dedicated config file. At this moment hyperopt 
cannot use your config file. It is also made on purpose to allow you
testing your strategy with different configurations.

The Hyperopt configuration is located in 
[freqtrade/optimize/hyperopt_conf.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/optimize/hyperopt_conf.py).


## Advanced notions
### Understand the Guards and Triggers
When you need to add the new guards and triggers to be hyperopt 
parameters, you do this by adding them into the [SPACE dict](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/optimize/hyperopt.py#L47-L94).

If it's a trigger, you add one line to the 'trigger' choice group and that's it.

If it's a guard, you will add a line like this:
```
'rsi': hp.choice('rsi', [
    {'enabled': False},
    {'enabled': True, 'value': hp.quniform('rsi-value', 20, 40, 1)}
]),
```
This says, "*one of guards is RSI, it can have two values, enabled or 
disabled. If it is enabled, try different values for it between 20 and 40*".

So, the part of the strategy builder using the above setting looks like 
this:
```
if params['rsi']['enabled']:
    conditions.append(dataframe['rsi'] < params['rsi']['value'])
```
It checks if Hyperopt wants the RSI guard to be enabled for this 
round `params['rsi']['enabled']` and if it is, then it will add a 
condition that says RSI must be < than the value hyperopt picked 
for this evaluation, that is given in the `params['rsi']['value']`.

That's it. Now you can add new parts of strategies to Hyperopt and it 
will try all the combinations with all different values in the search 
for best working algo.


### Add a new Indicators
If you want to test an indicator that isn't used by the bot currently, 
you need to add it to 
[freqtrade/analyze.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/analyze.py#L40-L70)
inside the `populate_indicators` function.

## Execute Hyperopt
Once you have updated your hyperopt configuration you can run it. 
Because hyperopt tries a lot of combination to find the best parameters
it will take time you will have the result (more than 30 mins).

We strongly recommend to use `screen` to prevent any connection loss.
```bash
python3 ./freqtrade/main.py -c config.json hyperopt
```

### Hyperopt with MongoDB
Hyperopt with MongoDB, is like Hyperopt under steroids. As you saw by
executing the previous command is the execution takes a long time. 
To accelerate it you can use hyperopt with MongoDB.

To run hyperopt with MongoDb you will need 3 terminals.

**Terminal 1: Start MongoDB**
```bash
cd <freqtrade> 
source .env/bin/activate
python3 scripts/start-mongodb.py
```

**Terminal 2: Start Hyperopt worker**
```bash
cd <freqtrade> 
source .env/bin/activate
python3 scripts/start-hyperopt-worker.py
```

**Terminal 3: Start Hyperopt with MongoDB**
```bash
cd <freqtrade> 
source .env/bin/activate
python3 ./freqtrade/main.py -c config.json hyperopt --use-mongodb
```

**Re-run an Hyperopt**
To re-run Hyperopt you have to delete the existing MongoDB table.
```bash
cd <freqtrade> 
rm -rf .hyperopt/mongodb/
```

## Understand the hyperopts result 
Once Hyperopt is completed you can use the result to adding new buy 
signal. Given following result from hyperopt:
```
Best parameters:
{
  "adx": 1,
  "adx-value": 15.0,
  "fastd": 1,
  "fastd-value": 40.0,
  "green_candle": 1,
  "mfi": 0,
  "over_sar": 0,
  "rsi": 1,
  "rsi-value": 37.0,
  "trigger": 0,
  "uptrend_long_ema": 1,
  "uptrend_short_ema": 0,
  "uptrend_sma": 0
}

Best Result:
  2197 trades. Avg profit  1.84%. Total profit  0.79367541 BTC. Avg duration 241.0 mins.
```

You should understand this result like:
- You should **consider** the guard "adx" (`"adx": 1,` = `adx` is true) 
and the best value is `15.0` (`"adx-value": 15.0,`)
- You should **consider** the guard "fastd" (`"fastd": 1,` = `fastd` 
is true) and the best value is `40.0` (`"fastd-value": 40.0,`)
- You should **consider** to enable the guard "green_candle" 
(`"green_candle": 1,` = `candle` is true) but this guards as no 
customizable value.
- You should **ignore** the guard "mfi" (`"mfi": 0,` = `mfi` is false)
- and so on...


You have to look from 
[freqtrade/optimize/hyperopt.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/optimize/hyperopt.py#L170-L200) 
what those values match to.   
  
So for example you had `adx-value: 15.0` (and `adx: 1` was true) so we 
would look at `adx`-block from 
[freqtrade/optimize/hyperopt.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/optimize/hyperopt.py#L178-L179). 
That translates to the following code block to 
[analyze.populate_buy_trend()](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/analyze.py#L73)
```
(dataframe['adx'] > 15.0)
```
  
So translating your whole hyperopt result to as the new buy-signal 
would be the following:
```
def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['adx'] > 15.0) & # adx-value
            (dataframe['fastd'] < 40.0) & # fastd-value
            (dataframe['close'] > dataframe['open']) & # green_candle
            (dataframe['rsi'] < 37.0) & # rsi-value
            (dataframe['ema50'] > dataframe['ema100']) # uptrend_long_ema
        ),
        'buy'] = 1
    return dataframe
```

## Next step
Now you have a perfect bot and want to control it from Telegram. Your
next step is to learn the [Telegram usage](https://github.com/gcarq/freqtrade/blob/develop/docs/telegram-usage.md).
