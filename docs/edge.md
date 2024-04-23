# Edge positioning

The `Edge Positioning` module uses probability to calculate your win rate and risk reward ratio. It will use these statistics to control your strategy trade entry points, position size and, stoploss.

!!! Danger "Deprecated functionality"
    `Edge positioning` (or short Edge) is currently in maintenance mode only (we keep existing functionality alive) and should be considered as deprecated.
    It will currently not receive new features until either someone stepped forward to take up ownership of that module - or we'll decide to remove edge from freqtrade.

!!! Warning
    When using `Edge positioning` with a dynamic whitelist (VolumePairList), make sure to also use `AgeFilter` and set it to at least `calculate_since_number_of_days` to avoid problems with missing data.

!!! Note
    `Edge Positioning` only considers *its own* buy/sell/stoploss signals. It ignores the stoploss, trailing stoploss, and ROI settings in the strategy configuration file.
    `Edge Positioning` improves the performance of some trading strategies and *decreases* the performance of others.


## Introduction

Trading strategies are not perfect. They are frameworks that are susceptible to the market and its indicators. Because the market is not at all predictable, sometimes a strategy will win and sometimes the same strategy will lose.

To obtain an edge in the market, a strategy has to make more money than it loses. Making money in trading is not only about *how often* the strategy makes or loses money.

!!! tip "It doesn't matter how often, but how much!"
    A bad strategy might make 1 penny in *ten* transactions but lose 1 dollar in *one* transaction. If one only checks the number of winning trades, it would be misleading to think that the strategy is actually making a profit.

The Edge Positioning module seeks to improve a strategy's winning probability and the money that the strategy will make *on the long run*. 

We raise the following question[^1]:

!!! Question "Which trade is a better option?"
    a) A trade with 80% of chance of losing 100\$ and 20% chance of winning 200\$<br/>
    b) A trade with 100% of chance of losing 30\$

???+ Info "Answer"
    The expected value of *a)* is smaller than the expected value of *b)*.<br/>
    Hence, *b*) represents a smaller loss in the long run.<br/>
    However, the answer is: *it depends*

Another way to look at it is to ask a similar question:

!!! Question "Which trade is a better option?"
    a) A trade with 80% of chance of winning 100\$ and 20% chance of losing 200\$<br/>
    b) A trade with 100% of chance of winning 30\$

Edge positioning tries to answer the hard questions about risk/reward and position size automatically, seeking to minimizes the chances of losing of a given strategy.

### Trading, winning and losing

Let's call $o$ the return of a single transaction $o$ where $o \in \mathbb{R}$. The collection $O = \{o_1, o_2, ..., o_N\}$ is the set of all returns of transactions made during a trading session. We say that $N$ is the cardinality of $O$, or, in lay terms, it is the number of transactions made in a trading session.

!!! Example
    In a session where a strategy made three transactions we can say that $O = \{3.5, -1, 15\}$. That means that $N = 3$ and $o_1 = 3.5$, $o_2 = -1$, $o_3 = 15$.

A winning trade is a trade where a strategy *made* money. Making money means that the strategy closed the position in a value that returned a profit, after all deducted fees. Formally, a winning trade will have a return $o_i > 0$. Similarly, a losing trade will have a return $o_j \leq 0$. With that, we can discover the set of all winning trades, $T_{win}$, as follows:

$$ T_{win} = \{ o \in O | o > 0 \} $$

Similarly, we can discover the set of losing trades $T_{lose}$ as follows:

$$ T_{lose} = \{o \in O | o \leq 0\} $$

!!! Example
    In a section where a strategy made four transactions $O = \{3.5, -1, 15, 0\}$:<br>
    $T_{win} = \{3.5, 15\}$<br>
    $T_{lose} = \{-1, 0\}$<br>

### Win Rate and Lose Rate

The win rate $W$ is the proportion of winning trades with respect to all the trades made by a strategy. We use the following function to compute the win rate:

$$W = \frac{|T_{win}|}{N}$$

Where $W$ is the win rate, $N$ is the number of trades and, $T_{win}$ is the set of all trades where the strategy made money.

Similarly, we can compute the rate of losing trades:

$$ 
    L = \frac{|T_{lose}|}{N} 
$$

Where $L$ is the lose rate, $N$ is the amount of trades made and, $T_{lose}$ is the set of all trades where the strategy lost money. Note that the above formula is the same as calculating $L = 1 – W$ or $W = 1 – L$

### Risk Reward Ratio

Risk Reward Ratio ($R$) is a formula used to measure the expected gains of a given investment against the risk of loss. It is basically what you potentially win divided by what you potentially lose. Formally:

$$ R = \frac{\text{potential_profit}}{\text{potential_loss}} $$

???+ Example "Worked example of $R$ calculation"
    Let's say that you think that the price of *stonecoin* today is 10.0\$. You believe that, because they will start mining stonecoin, it will go up to 15.0\$ tomorrow. There is the risk that the stone is too hard, and the GPUs can't mine it, so the price might go to 0\$ tomorrow. You are planning to invest 100\$, which will give you 10 shares (100 / 10).

    Your potential profit is calculated as:

    $\begin{aligned} 
        \text{potential_profit} &= (\text{potential_price} - \text{entry_price}) * \frac{\text{investment}}{\text{entry_price}} \\
                                &= (15 - 10) * (100 / 10) \\
                                &= 50
    \end{aligned}$

    Since the price might go to 0\$, the 100\$ dollars invested could turn into 0.

    We do however use a stoploss of 15% - so in the worst case, we'll sell 15% below entry price (or at 8.5$\).

    $\begin{aligned}
        \text{potential_loss} &= (\text{entry_price} - \text{stoploss}) * \frac{\text{investment}}{\text{entry_price}} \\
                                &= (10 - 8.5) * (100 / 10)\\
                                &= 15
    \end{aligned}$

    We can compute the Risk Reward Ratio as follows:

    $\begin{aligned}
        R   &= \frac{\text{potential_profit}}{\text{potential_loss}}\\
            &= \frac{50}{15}\\
            &= 3.33
    \end{aligned}$<br>
    What it effectively means is that the strategy have the potential to make 3.33\$ for each 1\$ invested.

On a long horizon, that is, on many trades, we can calculate the risk reward by dividing the strategy' average profit on winning trades by the strategy' average loss on losing trades. We can calculate the average profit, $\mu_{win}$, as follows:

$$ \text{average_profit} = \mu_{win} = \frac{\text{sum_of_profits}}{\text{count_winning_trades}} = \frac{\sum^{o \in T_{win}} o}{|T_{win}|} $$

Similarly, we can calculate the average loss, $\mu_{lose}$, as follows:

$$ \text{average_loss} = \mu_{lose} = \frac{\text{sum_of_losses}}{\text{count_losing_trades}} = \frac{\sum^{o \in T_{lose}} o}{|T_{lose}|} $$

Finally, we can calculate the Risk Reward ratio, $R$, as follows:

$$ R = \frac{\text{average_profit}}{\text{average_loss}} = \frac{\mu_{win}}{\mu_{lose}}\\ $$


???+ Example "Worked example of $R$ calculation using mean profit/loss"
    Let's say the strategy that we are using makes an average win $\mu_{win} = 2.06$ and an average loss $\mu_{loss} = 4.11$.<br>
    We calculate the risk reward ratio as follows:<br>
    $R = \frac{\mu_{win}}{\mu_{loss}} = \frac{2.06}{4.11} = 0.5012...$
    

### Expectancy

By combining the Win Rate $W$ and the Risk Reward ratio $R$ to create an expectancy ratio $E$. A expectance ratio is the expected return of the investment made in a trade. We can compute the value of $E$ as follows:

$$E = R * W - L$$

!!! Example "Calculating $E$"
    Let's say that a strategy has a win rate $W = 0.28$ and a risk reward ratio $R = 5$. What this means is that the strategy is expected to make 5 times the investment around on 28% of the trades it makes. Working out the example:<br>
    $E = R * W - L = 5 * 0.28 - 0.72 = 0.68$
    <br>

The expectancy worked out in the example above means that, on average, this strategy' trades will return 1.68 times the size of its losses. Said another way, the strategy makes 1.68\$ for every 1\$ it loses, on average. 

This is important for two reasons: First, it may seem obvious, but you know right away that you have a positive return. Second, you now have a number you can compare to other candidate systems to make decisions about which ones you employ.

It is important to remember that any system with an expectancy greater than 0 is profitable using past data. The key is finding one that will be profitable in the future.

You can also use this value to evaluate the effectiveness of modifications to this system.

!!! Note
    It's important to keep in mind that Edge is testing your expectancy using historical data, there's no guarantee that you will have a similar edge in the future. It's still vital to do this testing in order to build confidence in your methodology but be wary of "curve-fitting" your approach to the historical data as things are unlikely to play out the exact same way for future trades.

## How does it work?

Edge combines dynamic stoploss, dynamic positions, and whitelist generation into one isolated module which is then applied to the trading strategy. If enabled in config, Edge will go through historical data with a range of stoplosses in order to find buy and sell/stoploss signals. It then calculates win rate and expectancy over *N* trades for each stoploss. Here is an example:

| Pair   |      Stoploss      |  Win Rate | Risk Reward Ratio | Expectancy |
|----------|:-------------:|-------------:|------------------:|-----------:|
| XZC/ETH  |  -0.01        |   0.50       |1.176384           | 0.088      |
| XZC/ETH  |  -0.02        |   0.51       |1.115941           | 0.079      |
| XZC/ETH  |  -0.03        |   0.52       |1.359670           | 0.228      |
| XZC/ETH  |  -0.04        |   0.51       |1.234539           | 0.117      |

The goal here is to find the best stoploss for the strategy in order to have the maximum expectancy. In the above example stoploss at $3%$ leads to the maximum expectancy according to historical data.

Edge module then forces stoploss value it evaluated to your strategy dynamically.

### Position size

Edge dictates the amount at stake for each trade to the bot according to the following factors:

- Allowed capital at risk
- Stoploss

Allowed capital at risk is calculated as follows:

```
Allowed capital at risk = (Capital available_percentage) X (Allowed risk per trade)
```

Stoploss is calculated as described above with respect to historical data.

The position size is calculated as follows:

```
Position size = (Allowed capital at risk) / Stoploss
```

Example:

Let's say the stake currency is **ETH** and there is $10$ **ETH** on the wallet. The capital available percentage is $50%$ and the allowed risk per trade is $1\%$. Thus, the available capital for trading is $10 * 0.5 = 5$ **ETH** and the allowed capital at risk would be $5 * 0.01 = 0.05$ **ETH**.

-   **Trade 1:** The strategy detects a new buy signal in the **XLM/ETH** market. `Edge Positioning` calculates a stoploss of $2\%$ and a position of $0.05 / 0.02 = 2.5$ **ETH**. The bot takes a position of $2.5$ **ETH** in the **XLM/ETH** market.

-   **Trade 2:** The strategy detects a buy signal on the **BTC/ETH** market while **Trade 1** is still open. `Edge Positioning` calculates the stoploss of $4\%$ on this market. Thus, **Trade 2** position size is $0.05 / 0.04 = 1.25$ **ETH**.

!!! Tip "Available Capital $\neq$ Available in wallet"
    The available capital for trading didn't change in **Trade 2** even with **Trade 1** still open. The available capital **is not** the free amount in the wallet.

-   **Trade 3:** The strategy detects a buy signal in the **ADA/ETH** market. `Edge Positioning` calculates a stoploss of $1\%$ and a position of $0.05 / 0.01 = 5$ **ETH**. Since **Trade 1** has $2.5$ **ETH** blocked and **Trade 2** has $1.25$ **ETH** blocked, there is only $5 - 1.25 - 2.5 = 1.25$ **ETH** available. Hence, the position size of **Trade 3** is $1.25$ **ETH**. 

!!! Tip "Available Capital Updates"
    The available capital does not change before a position is sold. After a trade is closed the Available Capital goes up if the trade was profitable or goes down if the trade was a loss.

-   The strategy detects a sell signal in the **XLM/ETH** market. The bot exits **Trade 1** for a profit of $1$ **ETH**. The total capital in the wallet becomes $11$ **ETH** and the available capital for trading becomes $5.5$ **ETH**.

-   **Trade 4** The strategy detects a new buy signal int the **XLM/ETH** market. `Edge Positioning` calculates the stoploss of $2\%$, and the position size of $0.055 / 0.02 = 2.75$ **ETH**.

## Edge command reference

```
usage: freqtrade edge [-h] [-v] [--logfile FILE] [-V] [-c PATH] [-d PATH]
                      [--userdir PATH] [-s NAME] [--strategy-path PATH]
                      [-i TIMEFRAME] [--timerange TIMERANGE]
                      [--data-format-ohlcv {json,jsongz,hdf5}]
                      [--max-open-trades INT] [--stake-amount STAKE_AMOUNT]
                      [--fee FLOAT] [-p PAIRS [PAIRS ...]]
                      [--stoplosses STOPLOSS_RANGE]

optional arguments:
  -h, --help            show this help message and exit
  -i TIMEFRAME, --timeframe TIMEFRAME
                        Specify timeframe (`1m`, `5m`, `30m`, `1h`, `1d`).
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --data-format-ohlcv {json,jsongz,hdf5}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `None`).
  --max-open-trades INT
                        Override the value of the `max_open_trades`
                        configuration setting.
  --stake-amount STAKE_AMOUNT
                        Override the value of the `stake_amount` configuration
                        setting.
  --fee FLOAT           Specify fee ratio. Will be applied twice (on trade
                        entry and exit).
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --stoplosses STOPLOSS_RANGE
                        Defines a range of stoploss values against which edge
                        will assess the strategy. The format is "min,max,step"
                        (without any space). Example:
                        `--stoplosses=-0.01,-0.1,-0.001`

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE        Log to the file specified. Special values are:
                        'syslog', 'journald'. See the documentation for more
                        details.
  -V, --version         show program's version number and exit
  -c PATH, --config PATH
                        Specify configuration file (default:
                        `userdir/config.json` or `config.json` whichever
                        exists). Multiple --config options may be used. Can be
                        set to `-` to read config from stdin.
  -d PATH, --datadir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

Strategy arguments:
  -s NAME, --strategy NAME
                        Specify strategy class name which will be used by the
                        bot.
  --strategy-path PATH  Specify additional strategy lookup path.

```

## Configurations

Edge module has following configuration options:

|  Parameter | Description |
|------------|-------------|
| `enabled` | If true, then Edge will run periodically. <br>*Defaults to `false`.* <br> **Datatype:** Boolean
| `process_throttle_secs` | How often should Edge run in seconds. <br>*Defaults to `3600` (once per hour).* <br> **Datatype:** Integer
| `calculate_since_number_of_days` | Number of days of data against which Edge calculates Win Rate, Risk Reward and Expectancy. <br> **Note** that it downloads historical data so increasing this number would lead to slowing down the bot. <br>*Defaults to `7`.* <br> **Datatype:** Integer
| `allowed_risk` | Ratio of allowed risk per trade. <br>*Defaults to `0.01` (1%)).* <br> **Datatype:** Float
| `stoploss_range_min` | Minimum stoploss. <br>*Defaults to `-0.01`.* <br> **Datatype:** Float
| `stoploss_range_max` | Maximum stoploss. <br>*Defaults to `-0.10`.* <br> **Datatype:** Float
| `stoploss_range_step` | As an example if this is set to -0.01 then Edge will test the strategy for `[-0.01, -0,02, -0,03 ..., -0.09, -0.10]` ranges. <br> **Note** than having a smaller step means having a bigger range which could lead to slow calculation. <br> If you set this parameter to -0.001, you then slow down the Edge calculation by a factor of 10. <br>*Defaults to `-0.001`.* <br> **Datatype:** Float
| `minimum_winrate` | It filters out pairs which don't have at least minimum_winrate. <br>This comes handy if you want to be conservative and don't comprise win rate in favour of risk reward ratio. <br>*Defaults to `0.60`.* <br> **Datatype:** Float
| `minimum_expectancy` | It filters out pairs which have the expectancy lower than this number. <br>Having an expectancy of 0.20 means if you put 10\$ on a trade you expect a 12\$ return. <br>*Defaults to `0.20`.* <br> **Datatype:** Float
| `min_trade_number` | When calculating *W*, *R* and *E* (expectancy) against historical data, you always want to have a minimum number of trades. The more this number is the more Edge is reliable. <br>Having a win rate of 100% on a single trade doesn't mean anything at all. But having a win rate of 70% over past 100 trades means clearly something. <br>*Defaults to `10` (it is highly recommended not to decrease this number).* <br> **Datatype:** Integer
| `max_trade_duration_minute` | Edge will filter out trades with long duration. If a trade is profitable after 1 month, it is hard to evaluate the strategy based on it. But if most of trades are profitable and they have maximum duration of 30 minutes, then it is clearly a good sign.<br>**NOTICE:** While configuring this value, you should take into consideration your timeframe. As an example filtering out trades having duration less than one day for a strategy which has 4h interval does not make sense. Default value is set assuming your strategy interval is relatively small (1m or 5m, etc.).<br>*Defaults to `1440` (one day).* <br> **Datatype:** Integer
| `remove_pumps` | Edge will remove sudden pumps in a given market while going through historical data. However, given that pumps happen very often in crypto markets, we recommend you keep this off.<br>*Defaults to `false`.* <br> **Datatype:** Boolean

## Running Edge independently

You can run Edge independently in order to see in details the result. Here is an example:

``` bash
freqtrade edge
```

An example of its output:

| **pair**     |   **stoploss** |   **win rate** |   **risk reward ratio** |   **required risk reward** |   **expectancy** |   **total number of trades** |   **average duration (min)** |
|:----------|-----------:|-----------:|--------------------:|-----------------------:|-------------:|-----------------:|---------------:|
| **AGI/BTC**   |      -0.02 |       0.64 |                5.86 |                   0.56 |         3.41 |                       14 |                       54 |
| **NXS/BTC**   |      -0.03 |       0.64 |                2.99 |                   0.57 |         1.54 |                       11 |                       26 |
| **LEND/BTC**  |      -0.02 |       0.82 |                2.05 |                   0.22 |         1.50 |                       11 |                       36 |
| **VIA/BTC**   |      -0.01 |       0.55 |                3.01 |                   0.83 |         1.19 |                       11 |                       48 |
| **MTH/BTC**   |      -0.09 |       0.56 |                2.82 |                   0.80 |         1.12 |                       18 |                       52 |
| **ARDR/BTC**  |      -0.04 |       0.42 |                3.14 |                   1.40 |         0.73 |                       12 |                       42 |
| **BCPT/BTC**  |      -0.01 |       0.71 |                1.34 |                   0.40 |         0.67 |                       14 |                       30 |
| **WINGS/BTC** |      -0.02 |       0.56 |                1.97 |                   0.80 |         0.65 |                       27 |                       42 |
| **VIBE/BTC**  |      -0.02 |       0.83 |                0.91 |                   0.20 |         0.59 |                       12 |                       35 |
| **MCO/BTC**   |      -0.02 |       0.79 |                0.97 |                   0.27 |         0.55 |                       14 |                       31 |
| **GNT/BTC**   |      -0.02 |       0.50 |                2.06 |                   1.00 |         0.53 |                       18 |                       24 |
| **HOT/BTC**   |      -0.01 |       0.17 |                7.72 |                   4.81 |         0.50 |                      209 |                        7 |
| **SNM/BTC**   |      -0.03 |       0.71 |                1.06 |                   0.42 |         0.45 |                       17 |                       38 |
| **APPC/BTC**  |      -0.02 |       0.44 |                2.28 |                   1.27 |         0.44 |                       25 |                       43 |
| **NEBL/BTC**  |      -0.03 |       0.63 |                1.29 |                   0.58 |         0.44 |                       19 |                       59 |

Edge produced the above table by comparing `calculate_since_number_of_days` to `minimum_expectancy` to find `min_trade_number` historical information based on the config file. The timerange Edge uses for its comparisons can be further limited by using the `--timerange` switch.

In live and dry-run modes, after the `process_throttle_secs` has passed, Edge will again process `calculate_since_number_of_days` against `minimum_expectancy` to find `min_trade_number`. If no `min_trade_number` is found, the bot will return "whitelist empty". Depending on the trade strategy being deployed, "whitelist empty" may be return much of the time - or *all* of the time. The use of Edge may also cause trading to occur in bursts, though this is rare.

If you encounter "whitelist empty" a lot, condsider tuning `calculate_since_number_of_days`, `minimum_expectancy`  and `min_trade_number` to align to the trading frequency of your strategy.

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


[^1]: Question extracted from MIT Opencourseware S096 - Mathematics with applications in Finance: https://ocw.mit.edu/courses/mathematics/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/
