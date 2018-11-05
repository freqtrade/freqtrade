#  Edge positioning

This page explains how to use Edge Positioning module in your bot in order to enter into a trade only of the trade has a reasonable win rate and risk reward ration, and consequently adjust your position size and stoploss.

## Table of Contents

- [Introduction](#introduction)
- [How does it work?](#how-does-it-work?)
- [Configurations](#configurations)

## Introduction
Trading is all about probability. no one can claim that he has the strategy working all the time. you have to assume that sometimes you lose.<br/><br/>
But it doesn't mean there is no rule, it only means rules should work "most of the time". let's play a game: we toss a coin, heads: I give you 10$, tails: You give me 10$. is it an interetsing game ? no, it is quite boring, isn't it?<br/><br/>
But lets say the probabiliy that we have heads is 80%, and the probablilty that we have tails is 20%. now it is becoming interesting ...
That means 10$ x 80% versus 10$ x 20%. 8$ versus 2$. that means over time you will win 8$ risking only 2$ on each toss of coin.<br/><br/>
lets complicate it more: you win 80% of the time but only 2$, I win 20% of the time but 8$. the calculation is: 80% * 2$ versus 20% * 8$. it is becoming boring again because overtime you win $1.6$ (80% x 2$) and me $1.6 (20% * 8$) too.<br/><br/>
The question is: how do you calculate that? how do you know if you wanna play?
The answer comes to two factors:
- Win Rate
- Risk Reward Ratio


### Win Rate
Means over X trades what is the perctange of winning trades to total number of trades (note that we don't consider how much you gained but only If you won or not).


W = (Number of winning trades) / (Number of losing trades)

### Risk Reward Ratio
Risk Reward Ratio is a formula used to measure the expected gains of a given investment against the risk of loss. it is basically what you potentially win divided by what you potentially lose:

R = Profit / Loss

Over time, on many trades, you can calculate your risk reward by dividing your average profit on winning trades by your average loss on losing trades:

average profit = (Sum of profits) / (Number of winning trades)

average loss = (Sum of losses) / (Number of losing trades)

R = (average profit) / (average loss)

### Expectancy

At this point we can combine W and R to create an expectancy ratio. This is a simple process of multiplying the risk reward ratio by the percentage of winning trades, and subtracting the percentage of losing trades, which is calculated as follows:

Expectancy Ratio = (Risk Reward Ratio x Win Rate) – Loss Rate

So lets say your Win rate is 28% and your Risk Reward Ratio is 5:

Expectancy = (5 * 0.28) - 0.72 = 0.68

Superficially, this means that on average you expect this strategy’s trades to return .68 times the size of your losers. This is important for two reasons: First, it may seem obvious, but you know right away that you have a positive return. Second, you now have a number you can compare to other candidate systems to make decisions about which ones you employ.

It is important to remember that any system with an expectancy greater than 0 is profitable using past data. The key is finding one that will be profitable in the future.

You can also use this number to evaluate the effectiveness of modifications to this system.

**NOTICE:** It's important to keep in mind that Edge is testing your expectancy using historical data , there's no guarantee that you will have a similar edge in the future. It's still vital to do this testing in order to build confidence in your methodology, but be wary of "curve-fitting" your approach to the historical data as things are unlikely to play out the exact same way for future trades.

## How does it work?
If enabled in config, Edge will go through historical data with a range of stoplosses in order to find buy and sell/stoploss signals. it then calculates win rate and expectancy over X trades for each stoploss. here is an example:

| Pair   |      Stoploss      |  Win Rate | Risk Reward Ratio | Expectancy |
|----------|:-------------:|-------------:|------------------:|-----------:|
| XZC/ETH  |  -0.03        |   0.52       |1.359670           | 0.228      |
| XZC/ETH  |  -0.01        |   0.50       |1.176384           | 0.088      |
| XZC/ETH  |  -0.02        |   0.51       |1.115941           | 0.079      |

The goal here is to find the best stoploss for the strategy in order to have the maximum expectancy. in the above example stoploss at 3% leads to the maximum expectancy according to historical data.

Edge then forces stoploss to your strategy dynamically.

### Position size
Edge dictates the stake amount for each trade to the bot according to the following factors:

- Allowed capital at risk
- Stoploss

Alowed capital at risk is calculated as follows:

**allowed capital at risk** = **total capital** X **allowed risk per trade**

**Stoploss** is calculated as described above against historical data.

Your position size then will be:

**position size** = **allowed capital at risk** / **stoploss**

## Configurations
Edge has following configurations:

#### enabled
If true, then Edge will run periodically

#### process_throttle_secs
How often should Edge run ? (in seconds)

#### calculate_since_number_of_days
Number of days of data agaist which Edge calculates Win Rate, Risk Reward and Expectancy
Note that it downloads historical data so increasing this number would lead to slowing down the bot

#### total_capital_in_stake_currency
This your total capital at risk. if edge is enabled then stake_amount is ignored in favor of this parameter

#### allowed_risk
Percentage of allowed risk per trade

#### stoploss_range_min
Minimum stoploss (default to -0.01)

#### stoploss_range_max
Maximum stoploss (default to -0.10)

#### stoploss_range_step
As an example if this is set to -0.01 then Edge will test the strategy for [-0.01, -0,02, -0,03 ..., -0.09, -0.10] ranges.
Note than having a smaller step means having a bigger range which could lead to slow calculation. <br/>
if you set this parameter to -0.001, you then slow down the Edge calculatiob by a factor of 10

#### minimum_winrate
It filters pairs which don't have at least minimum_winrate (default to 0.60)
This comes handy if you want to be conservative and don't comprise win rate in favor of risk reward ratio.

#### minimum_expectancy
It filters paris which have an expectancy lower than this number (default to 0.20)
Having an expectancy of 0.20 means if you put 10$ on a trade you expect a 12$ return.

#### min_trade_number
When calulating W and R and E (expectancy) against histoical data, you always want to have a minimum number of trades. the more this number is the more Edge is reliable. having a win rate of 100% on a single trade doesn't mean anything at all. but having a win rate of 70% over past 100 trades means clearly something. <br/>

Default to 10 (it is highly recommanded not to decrease this number)

#### max_trade_duration_minute
Edge will filter out trades with long duration. if a trade is profitable after 1 month, it is hard to evaluate the stratgy based on it. but if most of trades are profitable and they have maximum duration of 30 minutes, then it is clearly a good sign.<br/>
Default to 1 day (1440 = 60 * 24)


#### remove_pumps
Edge will remove sudden pumps in a given market while going through historical data. however, given that pumps happen very often in crypto markets, we recommand you keep this off.<br/>
Default to false