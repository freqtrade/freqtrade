#  Edge positioning

This page explains how to use Edge Positioning module in your bot in order to enter into a trade only if the trade has a reasonable win rate and risk reward ratio, and consequently adjust your position size and stoploss.

!!! Warning
    Edge positioning is not compatible with dynamic (volume-based) whitelist.

!!! Note
    Edge does not consider anything else than buy/sell/stoploss signals. So trailing stoploss, ROI, and everything else are ignored in its calculation.

## Introduction
Trading is all about probability. No one can claim that he has a strategy working all the time. You have to assume that sometimes you lose.

But it doesn't mean there is no rule, it only means rules should work "most of the time". Let's play a game: we toss a coin, heads: I give you 10$, tails: you give me 10$. Is it an interesting game? No, it's quite boring, isn't it?

But let's say the probability that we have heads is 80% (because our coin has the displaced distribution of mass or other defect), and the probability that we have tails is 20%. Now it is becoming interesting...

That means 10$ X 80% versus 10$ X 20%. 8$ versus 2$. That means over time you will win 8$ risking only 2$ on each toss of coin.

Let's complicate it more: you win 80% of the time but only 2$, I win 20% of the time but 8$. The calculation is: 80% X 2$ versus 20% X 8$. It is becoming boring again because overtime you win $1.6$ (80% X 2$) and me $1.6 (20% X 8$) too.

The question is: How do you calculate that? How do you know if you wanna play?

The answer comes to two factors:
- Win Rate
- Risk Reward Ratio

### Win Rate
Win Rate (*W*) is is the mean over some amount of trades (*N*) what is the percentage of winning trades to total number of trades (note that we don't consider how much you gained but only if you won or not).

    W = (Number of winning trades) / (Total number of trades) = (Number of winning trades) / N

Complementary Loss Rate (*L*) is defined as

    L = (Number of losing trades) / (Total number of trades) = (Number of losing trades) / N

or, which is the same, as

    L = 1 – W

### Risk Reward Ratio
Risk Reward Ratio (*R*) is a formula used to measure the expected gains of a given investment against the risk of loss. It is basically what you potentially win divided by what you potentially lose:

    R = Profit / Loss

Over time, on many trades, you can calculate your risk reward by dividing your average profit on winning trades by your average loss on losing trades:

    Average profit = (Sum of profits) / (Number of winning trades)

    Average loss = (Sum of losses) / (Number of losing trades)

    R = (Average profit) / (Average loss)

### Expectancy
At this point we can combine *W* and *R* to create an expectancy ratio. This is a simple process of multiplying the risk reward ratio by the percentage of winning trades and subtracting the percentage of losing trades, which is calculated as follows:

    Expectancy Ratio = (Risk Reward Ratio X Win Rate) – Loss Rate = (R X W) – L

So lets say your Win rate is 28% and your Risk Reward Ratio is 5:

    Expectancy = (5 X 0.28) – 0.72 = 0.68

Superficially, this means that on average you expect this strategy’s trades to return .68 times the size of your loses. This is important for two reasons: First, it may seem obvious, but you know right away that you have a positive return. Second, you now have a number you can compare to other candidate systems to make decisions about which ones you employ.

It is important to remember that any system with an expectancy greater than 0 is profitable using past data. The key is finding one that will be profitable in the future.

You can also use this value to evaluate the effectiveness of modifications to this system.

**NOTICE:** It's important to keep in mind that Edge is testing your expectancy using historical data, there's no guarantee that you will have a similar edge in the future. It's still vital to do this testing in order to build confidence in your methodology, but be wary of "curve-fitting" your approach to the historical data as things are unlikely to play out the exact same way for future trades.

## How does it work?
If enabled in config, Edge will go through historical data with a range of stoplosses in order to find buy and sell/stoploss signals. It then calculates win rate and expectancy over *N* trades for each stoploss. Here is an example:

| Pair   |      Stoploss      |  Win Rate | Risk Reward Ratio | Expectancy |
|----------|:-------------:|-------------:|------------------:|-----------:|
| XZC/ETH  |  -0.01        |   0.50       |1.176384           | 0.088      |
| XZC/ETH  |  -0.02        |   0.51       |1.115941           | 0.079      |
| XZC/ETH  |  -0.03        |   0.52       |1.359670           | 0.228      |
| XZC/ETH  |  -0.04        |   0.51       |1.234539           | 0.117      |

The goal here is to find the best stoploss for the strategy in order to have the maximum expectancy. In the above example stoploss at 3% leads to the maximum expectancy according to historical data.

Edge module then forces stoploss value it evaluated to your strategy dynamically.

### Position size
Edge also dictates the stake amount for each trade to the bot according to the following factors:

- Allowed capital at risk
- Stoploss

Allowed capital at risk is calculated as follows:

    Allowed capital at risk = (Capital available_percentage) X (Allowed risk per trade)

Stoploss is calculated as described above against historical data.

Your position size then will be:

    Position size = (Allowed capital at risk) / Stoploss

Example:

Let's say the stake currency is ETH and you have 10 ETH on the exchange, your capital available percentage is 50% and you would allow 1% of risk for each trade. thus your available capital for trading is **10 x 0.5 = 5 ETH** and allowed capital at risk would be **5 x 0.01 = 0.05 ETH**.

Let's assume Edge has calculated that for **XLM/ETH** market your stoploss should be at 2%. So your position size will be **0.05 / 0.02 = 2.5 ETH**.

Bot takes a position of 2.5 ETH on XLM/ETH (call it trade 1). Up next, you receive another buy signal while trade 1 is still open. This time on **BTC/ETH** market. Edge calculated stoploss for this market at 4%. So your position size would be 0.05 / 0.04 = 1.25 ETH (call it trade 2).

Note that available capital for trading didn’t change for trade 2 even if you had already trade 1. The available capital doesn’t mean the free amount on your wallet.

Now you have two trades open. The bot receives yet another buy signal for another market: **ADA/ETH**. This time the stoploss is calculated at 1%. So your position size is **0.05 / 0.01 = 5 ETH**. But there are already 3.75 ETH blocked in two previous trades. So the position size for this third trade would be **5 – 3.75 = 1.25 ETH**.

Available capital doesn’t change before a position is sold. Let’s assume that trade 1 receives a sell signal and it is sold with a profit of 1 ETH. Your total capital on exchange would be 11 ETH and the available capital for trading becomes 5.5 ETH.

So the Bot receives another buy signal for trade 4 with a stoploss at 2% then your position size would be **0.055 / 0.02 = 2.75 ETH**.

## Configurations
Edge module has following configuration options:

#### enabled
If true, then Edge will run periodically.

(defaults to false)

#### process_throttle_secs
How often should Edge run in seconds?

(defaults to 3600 so one hour)

#### calculate_since_number_of_days
Number of days of data against which Edge calculates Win Rate, Risk Reward and Expectancy
Note that it downloads historical data so increasing this number would lead to slowing down the bot.

(defaults to 7)

#### capital_available_percentage
This is the percentage of the total capital on exchange in stake currency.

As an example if you have 10 ETH available in your wallet on the exchange and this value is 0.5 (which is 50%), then the bot will use a maximum amount of 5 ETH for trading and considers it as available capital.

(defaults to 0.5)

#### allowed_risk
Percentage of allowed risk per trade.

(defaults to 0.01 so 1%)

#### stoploss_range_min

Minimum stoploss.

(defaults to -0.01)

#### stoploss_range_max

Maximum stoploss.

(defaults to -0.10)

#### stoploss_range_step

As an example if this is set to -0.01 then Edge will test the strategy for \[-0.01, -0,02, -0,03 ..., -0.09, -0.10\] ranges.
Note than having a smaller step means having a bigger range which could lead to slow calculation.

If you set this parameter to -0.001, you then slow down the Edge calculation by a factor of 10.

(defaults to -0.01)

#### minimum_winrate

It filters out pairs which don't have at least minimum_winrate.

This comes handy if you want to be conservative and don't comprise win rate in favour of risk reward ratio.

(defaults to 0.60)

#### minimum_expectancy

It filters out pairs which have the expectancy lower than this number.

Having an expectancy of 0.20 means if you put 10$ on a trade you expect a 12$ return.

(defaults to 0.20)

#### min_trade_number

When calculating *W*, *R* and *E* (expectancy) against historical data, you always want to have a minimum number of trades. The more this number is the more Edge is reliable.

Having a win rate of 100% on a single trade doesn't mean anything at all. But having a win rate of 70% over past 100 trades means clearly something.

(defaults to 10, it is highly recommended not to decrease this number)

#### max_trade_duration_minute

Edge will filter out trades with long duration. If a trade is profitable after 1 month, it is hard to evaluate the strategy based on it. But if most of trades are profitable and they have maximum duration of 30 minutes, then it is clearly a good sign.

**NOTICE:** While configuring this value, you should take into consideration your ticker interval. As an example filtering out trades having duration less than one day for a strategy which has 4h interval does not make sense. Default value is set assuming your strategy interval is relatively small (1m or 5m, etc.).

(defaults to 1 day, i.e. to 60 * 24 = 1440 minutes)

#### remove_pumps

Edge will remove sudden pumps in a given market while going through historical data. However, given that pumps happen very often in crypto markets, we recommend you keep this off.

(defaults to false)

## Running Edge independently

You can run Edge independently in order to see in details the result. Here is an example:

```bash
freqtrade edge
```

An example of its output:

| pair      |   stoploss |   win rate |   risk reward ratio |   required risk reward |   expectancy |   total number of trades |   average duration (min) |
|:----------|-----------:|-----------:|--------------------:|-----------------------:|-------------:|-------------------------:|-------------------------:|
| AGI/BTC   |      -0.02 |       0.64 |                5.86 |                   0.56 |         3.41 |                       14 |                       54 |
| NXS/BTC   |      -0.03 |       0.64 |                2.99 |                   0.57 |         1.54 |                       11 |                       26 |
| LEND/BTC  |      -0.02 |       0.82 |                2.05 |                   0.22 |         1.50 |                       11 |                       36 |
| VIA/BTC   |      -0.01 |       0.55 |                3.01 |                   0.83 |         1.19 |                       11 |                       48 |
| MTH/BTC   |      -0.09 |       0.56 |                2.82 |                   0.80 |         1.12 |                       18 |                       52 |
| ARDR/BTC  |      -0.04 |       0.42 |                3.14 |                   1.40 |         0.73 |                       12 |                       42 |
| BCPT/BTC  |      -0.01 |       0.71 |                1.34 |                   0.40 |         0.67 |                       14 |                       30 |
| WINGS/BTC |      -0.02 |       0.56 |                1.97 |                   0.80 |         0.65 |                       27 |                       42 |
| VIBE/BTC  |      -0.02 |       0.83 |                0.91 |                   0.20 |         0.59 |                       12 |                       35 |
| MCO/BTC   |      -0.02 |       0.79 |                0.97 |                   0.27 |         0.55 |                       14 |                       31 |
| GNT/BTC   |      -0.02 |       0.50 |                2.06 |                   1.00 |         0.53 |                       18 |                       24 |
| HOT/BTC   |      -0.01 |       0.17 |                7.72 |                   4.81 |         0.50 |                      209 |                        7 |
| SNM/BTC   |      -0.03 |       0.71 |                1.06 |                   0.42 |         0.45 |                       17 |                       38 |
| APPC/BTC  |      -0.02 |       0.44 |                2.28 |                   1.27 |         0.44 |                       25 |                       43 |
| NEBL/BTC  |      -0.03 |       0.63 |                1.29 |                   0.58 |         0.44 |                       19 |                       59 |

### Update cached pairs with the latest data

Edge requires historic data the same way as backtesting does.
Please refer to the [Data Downloading](data-download.md) section of the documentation for details.

### Precising stoploss range

```bash
freqtrade edge --stoplosses=-0.01,-0.1,-0.001 #min,max,step
```

### Advanced use of timerange

```bash
freqtrade edge --timerange=20181110-20181113
```

Doing `--timerange=-20190901` will get all available data until September 1st (excluding September 1st 2019).

The full timerange specification:

* Use tickframes till 2018/01/31: `--timerange=-20180131`
* Use tickframes since 2018/01/31: `--timerange=20180131-`
* Use tickframes since 2018/01/31 till 2018/03/01 : `--timerange=20180131-20180301`
* Use tickframes between POSIX timestamps 1527595200 1527618600: `--timerange=1527595200-1527618600`
