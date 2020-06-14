# How to run multiple instances of freqtrade simultaneously

This page is meant to be a quick tip for new users on how to run multiple bots a the same time, on the same computer (or other devices). 

In order to keep track of your trades, profits, etc., freqtrade is using a SQLite database where it stores various types of information such as the trades you performed in the past and the current position(s) you are holding at any time. This allow you to keep track of your profits, but most importantly, keep track of ongoing activity if the bot process would finish unexpectedly for one or another reason.

As for various other things, upon docker or manual install, [freqtrade will create by default two different databases, one for dry-run, and the other for live trades.](https://www.freqtrade.io/en/latest/docker/#create-your-database-file)

By default, executing the trade command in the command line interface, without specifying any database (`--db-url`) argument, freqtrade will store your trades and performance history in one of these two default databases, depending if dry-run mode is enabled or not. These databases are actual .sqlite files which are stored in your main freqtrade folder, under the name `tradesv3.dryrun.sqlite` for the dry-run mode, and `tradesv3.sqlite` for the live mode.

The optional argument to the trade command used to specify the path of these files is `--db-url`. So when you are starting a bot with only the config and strategy arguments in dry-run mode for instance :

```
freqtrade trade -c MyConfig.json -s MyStrategy
```

is equivalent to :

```
freqtrade trade -c MyConfig.json -s MyStrategy --db-url sqlite:///tradesv3.dryrun.sqlite
```

That means that if you are running the trade command in two different terminals, for example to test your strategy both for trades in USDT and in another instance for trades in BTC, you will have to run them with different databases. Even if you are using the same configuration file and strategy.

 If you specify the URL of a database which does not exist, freqtrade will create one with the name you specified. So for example if you want to test your custom strategy in BTC vs USDT, you could type in one terminal :

```
freqtrade trade -c MyConfigBTC.json -s MyCustomStrategy --db-url sqlite://path/tradesBTC.dryrun.sqlite
```

and in the other

```
freqtrade trade -c MyConfigUSDT.json -s MyCustomStrategy --db-url sqlite://path/path/tradesUSDT.dryrun.sqlite
```

Conversely, if you wish to do the same thing in production mode, you will also have to create at least one new database (in addition to the default one) and specify the path to the "live" databases, for example :

```
freqtrade trade -c MyConfigBTC.json -s MyCustomStrategy --db-url sqlite://path/tradesBTC.live.sqlite
```

and in the other

```
freqtrade trade -c MyConfigUSDT.json -s MyCustomStrategy --db-url sqlite://path/path/tradesUSDT.live.sqlite
```

For more information regarding usage of the sqlite databases, for example to manually enter or remove trades, please refer to the following page :

https://www.freqtrade.io/en/latest/sql_cheatsheet/
