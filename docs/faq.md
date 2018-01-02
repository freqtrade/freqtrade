# freqtrade FAQ

#### I have waited 5 minutes, why hasn't the bot made any trades yet?!

Depending on the buy strategy, the amount of whitelisted coins, the situation of the market etc, it can take up to hours to find good entry position for a trade. Be patient!

#### I have made 12 trades already, why is my total profit negative?!

I understand your disappointment but unfortunately 12 trades is just not enough to say anything. If you run backtesting, you can see that our current algorithm does leave you on the plus side, but that is after thousands of trades and even there, you will be left with losses on specific coins that you have traded tens if not hundreds of times. We of course constantly aim to improve the bot but it will _always_ be a gamble, which should leave you with modest wins on monthly basis but you can't say much from few trades.

#### I’d like to change the stake amount. Can I just stop the bot with /stop and then change the config.json and run it again?

Not quite. Trades are persisted to a database but the configuration is currently only read when the bot is killed and restarted. `/stop` more like pauses. You can stop your bot, adjust settings and start it again.

#### I want to improve the bot with a new strategy

That's great. We have a nice backtesting and hyperoptimizing setup. See the tutorial [[here|Testing-new-strategies-with-Hyperopt]].

#### Is there a setting to only SELL the coins being held and not perform anymore BUYS?

You can use the `/forcesell all` command from Telegram.