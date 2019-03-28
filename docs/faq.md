# Freqtrade FAQ

### Freqtrade commons

#### I have waited 5 minutes, why hasn't the bot made any trades yet?!

Depending on the buy strategy, the amount of whitelisted coins, the
situation of the market etc, it can take up to hours to find good entry
position for a trade. Be patient!

#### I have made 12 trades already, why is my total profit negative?!

I understand your disappointment but unfortunately 12 trades is just
not enough to say anything. If you run backtesting, you can see that our
current algorithm does leave you on the plus side, but that is after
thousands of trades and even there, you will be left with losses on
specific coins that you have traded tens if not hundreds of times. We
of course constantly aim to improve the bot but it will _always_ be a
gamble, which should leave you with modest wins on monthly basis but
you can't say much from few trades.

#### I’d like to change the stake amount. Can I just stop the bot with
/stop and then change the config.json and run it again?

Not quite. Trades are persisted to a database but the configuration is
currently only read when the bot is killed and restarted. `/stop` more
like pauses. You can stop your bot, adjust settings and start it again.

#### I want to improve the bot with a new strategy

That's great. We have a nice backtesting and hyperoptimizing setup. See
the tutorial [here|Testing-new-strategies-with-Hyperopt](bot-usage.md#hyperopt-commands).

#### Is there a setting to only SELL the coins being held and not
perform anymore BUYS?

You can use the `/forcesell all` command from Telegram.

### Hyperopt module

#### How many epoch do I need to get a good Hyperopt result?
Per default Hyperopts without `-e` or `--epochs` parameter will only
run 100 epochs, means 100 evals of your triggers, guards, .... Too few
to find a great result (unless if you are very lucky), so you probably
have to run it for 10.000 or more. But it will take an eternity to
compute.

We recommend you to run it at least 10.000 epochs:

```bash
python3 freqtrade hyperopt -e 10000
```

or if you want intermediate result to see

```bash
for i in {1..100}; do python3 freqtrade hyperopt -e 100; done
```

#### Why it is so long to run hyperopt?

Finding a great Hyperopt results takes time.

If you wonder why it takes a while to find great hyperopt results

This answer was written during the under the release 0.15.1, when we had:
- 8 triggers
- 9 guards: let's say we evaluate even 10 values from each
- 1 stoploss calculation: let's say we want 10 values from that too to
be evaluated

The following calculation is still very rough and not very precise
but it will give the idea. With only these triggers and guards there is
already 8\*10^9\*10 evaluations. A roughly total of 80 billion evals.
Did you run 100 000 evals? Congrats, you've done roughly 1 / 100 000 th
of the search space.

### Edge module

#### Edge implements interesting approach for controlling position size, is there any theory behind it?

The Edge module is mostly a result of brainstorming of [@mishaker](https://github.com/mishaker) and [@creslinux](https://github.com/creslinux) freqtrade team members.

You can find further info on expectancy, winrate, risk management and position size in the following sources:
* https://www.tradeciety.com/ultimate-math-guide-for-traders/
* http://www.vantharp.com/tharp-concepts/expectancy.asp
* https://samuraitradingacademy.com/trading-expectancy/
* https://www.learningmarkets.com/determining-expectancy-in-your-trading/
* http://www.lonestocktrader.com/make-money-trading-positive-expectancy/
* https://www.babypips.com/trading/trade-expectancy-matter

