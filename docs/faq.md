# Freqtrade FAQ

## Supported Markets

Freqtrade supports spot trading only.

### Can I open short positions?

No, Freqtrade does not support trading with margin / leverage, and cannot open short positions.

In some cases, your exchange may provide leveraged spot tokens which can be traded with Freqtrade eg. BTCUP/USD, BTCDOWN/USD, ETHBULL/USD, ETHBEAR/USD, etc...

### Can I trade options or futures?

No, options and futures trading are not supported.

## Beginner Tips & Tricks

* When you work with your strategy & hyperopt file you should use a proper code editor like VSCode or PyCharm. A good code editor will provide syntax highlighting as well as line numbers, making it easy to find syntax errors (most likely pointed out by Freqtrade during startup).

## Freqtrade common issues

### The bot does not start

Running the bot with `freqtrade trade --config config.json` shows the output `freqtrade: command not found`.

This could be caused by the following reasons:

* The virtual environment is not active.
  * Run `source .env/bin/activate` to activate the virtual environment.
* The installation did not work correctly.
  * Please check the [Installation documentation](installation.md).

### I have waited 5 minutes, why hasn't the bot made any trades yet?

* Depending on the buy strategy, the amount of whitelisted coins, the
situation of the market etc, it can take up to hours to find a good entry
position for a trade. Be patient!

* It may be because of a configuration error. It's best to check the logs, they usually tell you if the bot is simply not getting buy signals (only heartbeat messages), or if there is something wrong (errors / exceptions in the log).

### I have made 12 trades already, why is my total profit negative?

I understand your disappointment but unfortunately 12 trades is just
not enough to say anything. If you run backtesting, you can see that our
current algorithm does leave you on the plus side, but that is after
thousands of trades and even there, you will be left with losses on
specific coins that you have traded tens if not hundreds of times. We
of course constantly aim to improve the bot but it will _always_ be a
gamble, which should leave you with modest wins on monthly basis but
you can't say much from few trades.

### I’d like to make changes to the config. Can I do that without having to kill the bot?

Yes. You can edit your config and use the `/reload_config` command to reload the configuration. The bot will stop, reload the configuration and strategy and will restart with the new configuration and strategy.

### I want to improve the bot with a new strategy

That's great. We have a nice backtesting and hyperoptimization setup. See the tutorial [here|Testing-new-strategies-with-Hyperopt](bot-usage.md#hyperopt-commands).

### Is there a setting to only SELL the coins being held and not perform anymore BUYS?

You can use the `/stopbuy` command in Telegram to prevent future buys, followed by `/forcesell all` (sell all open trades).

### I want to run multiple bots on the same machine

Please look at the [advanced setup documentation Page](advanced-setup.md#running-multiple-instances-of-freqtrade).

### I'm getting "Missing data fillup" messages in the log

This message is just a warning that the latest candles had missing candles in them.
Depending on the exchange, this can indicate that the pair didn't have a trade for the timeframe you are using - and the exchange does only return candles with volume.
On low volume pairs, this is a rather common occurrence.

If this happens for all pairs in the pairlist, this might indicate a recent exchange downtime. Please check your exchange's public channels for details.

Irrespectively of the reason, Freqtrade will fill up these candles with "empty" candles, where open, high, low and close are set to the previous candle close - and volume is empty. In a chart, this will look like a `_` - and is aligned with how exchanges usually represent 0 volume candles.

### I'm getting the "RESTRICTED_MARKET" message in the log

Currently known to happen for US Bittrex users.  

Read [the Bittrex section about restricted markets](exchanges.md#restricted-markets) for more information.

### I'm getting the "Exchange Bittrex does not support market orders." message and cannot run my strategy

As the message says, Bittrex does not support market orders and you have one of the [order types](configuration.md/#understand-order_types) set to "market". Your strategy was probably written with other exchanges in mind and sets "market" orders for "stoploss" orders, which is correct and preferable for most of the exchanges supporting market orders (but not for Bittrex).

To fix it for Bittrex, redefine order types in the strategy to use "limit" instead of "market":

```
    order_types = {
        ...
        'stoploss': 'limit',
        ...
    }
```

The same fix should be applied in the configuration file, if order types are defined in your custom config rather than in the strategy.

### How do I search the bot logs for something?

By default, the bot writes its log into stderr stream. This is implemented this way so that you can easily separate the bot's diagnostics messages from Backtesting, Edge and Hyperopt results, output from other various Freqtrade utility sub-commands, as well as from the output of your custom `print()`'s you may have inserted into your strategy. So if you need to search the log messages with the grep utility, you need to redirect stderr to stdout and disregard stdout.

* In unix shells, this normally can be done as simple as:
```shell
$ freqtrade --some-options 2>&1 >/dev/null | grep 'something'
```
(note, `2>&1` and `>/dev/null` should be written in this order)

* Bash interpreter also supports so called process substitution syntax, you can grep the log for a string with it as:
```shell
$ freqtrade --some-options 2> >(grep 'something') >/dev/null
```
or
```shell
$ freqtrade --some-options 2> >(grep -v 'something' 1>&2)
```

* You can also write the copy of Freqtrade log messages to a file with the `--logfile` option:
```shell
$ freqtrade --logfile /path/to/mylogfile.log --some-options
```
and then grep it as:
```shell
$ cat /path/to/mylogfile.log | grep 'something'
```
or even on the fly, as the bot works and the log file grows:
```shell
$ tail -f /path/to/mylogfile.log | grep 'something'
```
from a separate terminal window.

On Windows, the `--logfile` option is also supported by Freqtrade and you can use the `findstr` command to search the log for the string of interest:
```
> type \path\to\mylogfile.log | findstr "something"
```

## Hyperopt module

### How many epochs do I need to get a good Hyperopt result?

Per default Hyperopt called without the `-e`/`--epochs` command line option will only
run 100 epochs, means 100 evaluations of your triggers, guards, ... Too few
to find a great result (unless if you are very lucky), so you probably
have to run it for 10.000 or more. But it will take an eternity to
compute.

Since hyperopt uses Bayesian search, running for too many epochs may not produce greater results.

It's therefore recommended to run between 500-1000 epochs over and over until you hit at least 10.000 epochs in total (or are satisfied with the result). You can best judge by looking at the results - if the bot keeps discovering better strategies, it's best to keep on going.

```bash
freqtrade hyperopt --hyperopt SampleHyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy SampleStrategy -e 1000
```

### Why does it take a long time to run hyperopt?

* Discovering a great strategy with Hyperopt takes time. Study www.freqtrade.io, the Freqtrade Documentation page, join the Freqtrade [Slack community](https://join.slack.com/t/highfrequencybot/shared_invite/zt-mm786y93-Fxo37glxMY9g8OQC5AoOIw) - or the Freqtrade [discord community](https://discord.gg/MA9v74M). While you patiently wait for the most advanced, free crypto bot in the world, to hand you a possible golden strategy specially designed just for you.

* If you wonder why it can take from 20 minutes to days to do 1000 epochs here are some answers:

This answer was written during the release 0.15.1, when we had:

* 8 triggers
* 9 guards: let's say we evaluate even 10 values from each
* 1 stoploss calculation: let's say we want 10 values from that too to be evaluated

The following calculation is still very rough and not very precise
but it will give the idea. With only these triggers and guards there is
already 8\*10^9\*10 evaluations. A roughly total of 80 billion evaluations.
Did you run 100 000 evaluations? Congrats, you've done roughly 1 / 100 000 th
of the search space, assuming that the bot never tests the same parameters more than once.

* The time it takes to run 1000 hyperopt epochs depends on things like: The available cpu, hard-disk, ram, timeframe, timerange, indicator settings, indicator count, amount of coins that hyperopt test strategies on and the resulting trade count - which can be 650 trades in a year or 10.0000 trades depending if the strategy aims for big profits by trading rarely or for many low profit trades. 

Example: 4% profit 650 times vs 0,3% profit a trade 10.000 times in a year. If we assume you set the --timerange to 365 days. 

Example:
`freqtrade --config config.json --strategy SampleStrategy --hyperopt SampleHyperopt -e 1000 --timerange 20190601-20200601`

## Edge module

### Edge implements interesting approach for controlling position size, is there any theory behind it?

The Edge module is mostly a result of brainstorming of [@mishaker](https://github.com/mishaker) and [@creslinux](https://github.com/creslinux) freqtrade team members.

You can find further info on expectancy, win rate, risk management and position size in the following sources:

- https://www.tradeciety.com/ultimate-math-guide-for-traders/
- http://www.vantharp.com/tharp-concepts/expectancy.asp
- https://samuraitradingacademy.com/trading-expectancy/
- https://www.learningmarkets.com/determining-expectancy-in-your-trading/
- http://www.lonestocktrader.com/make-money-trading-positive-expectancy/
- https://www.babypips.com/trading/trade-expectancy-matter
