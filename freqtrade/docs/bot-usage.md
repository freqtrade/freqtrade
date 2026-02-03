# Start the bot

This page explains the different parameters of the bot and how to run it.

!!! Note
    If you've used `setup.sh`, don't forget to activate your virtual environment (`source .venv/bin/activate`) before running freqtrade commands.

!!! Warning "Up-to-date clock"
    The clock on the system running the bot must be accurate, synchronized to a NTP server frequently enough to avoid problems with communication to the exchanges.

## Bot commands

--8<-- "commands/main.md"

### Bot trading commands

--8<-- "commands/trade.md"

### How to specify which configuration file be used?

The bot allows you to select which configuration file you want to use by means of
the `-c/--config` command line option:

```bash
freqtrade trade -c path/far/far/away/config.json
```

Per default, the bot loads the `config.json` configuration file from the current
working directory.

### How to use multiple configuration files?

The bot allows you to use multiple configuration files by specifying multiple
`-c/--config` options in the command line. Configuration parameters
defined in the latter configuration files override parameters with the same name
defined in the previous configuration files specified in the command line earlier.

For example, you can make a separate configuration file with your key and secret
for the Exchange you use for trading, specify default configuration file with
empty key and secret values while running in the Dry Mode (which does not actually
require them):

```bash
freqtrade trade -c ./config.json
```

and specify both configuration files when running in the normal Live Trade Mode:

```bash
freqtrade trade -c ./config.json -c path/to/secrets/keys.config.json
```

This could help you hide your private Exchange key and Exchange secret on you local machine
by setting appropriate file permissions for the file which contains actual secrets and, additionally,
prevent unintended disclosure of sensitive private data when you publish examples
of your configuration in the project issues or in the Internet.

See more details on this technique with examples in the documentation page on
[configuration](configuration.md).

### Where to store custom data

Freqtrade allows the creation of a user-data directory using `freqtrade create-userdir --userdir someDirectory`.
This directory will look as follows:

```
user_data/
├── backtest_results
├── data
├── hyperopts
├── hyperopt_results
├── plot
└── strategies
```

You can add the entry "user_data_dir" setting to your configuration, to always point your bot to this directory.
Alternatively, pass in `--userdir` to every command.
The bot will fail to start if the directory does not exist, but will create necessary subdirectories.

This directory should contain your custom strategies, custom hyperopts and hyperopt loss functions, backtesting historical data (downloaded using either backtesting command or the download script) and plot outputs.

It is recommended to use version control to keep track of changes to your strategies.

### How to use **--strategy**?

This parameter will allow you to load your custom strategy class.
To test the bot installation, you can use the `SampleStrategy` installed by the `create-userdir` subcommand (usually `user_data/strategy/sample_strategy.py`).

The bot will search your strategy file within `user_data/strategies`.
To use other directories, please read the next section about `--strategy-path`.

To load a strategy, simply pass the class name (e.g.: `CustomStrategy`) in this parameter.

**Example:**
In `user_data/strategies` you have a file `my_awesome_strategy.py` which has
a strategy class called `AwesomeStrategy` to load it:

```bash
freqtrade trade --strategy AwesomeStrategy
```

If the bot does not find your strategy file, it will display in an error
message the reason (File not found, or errors in your code).

Learn more about strategy file in
[Strategy Customization](strategy-customization.md).

### How to use **--strategy-path**?

This parameter allows you to add an additional strategy lookup path, which gets
checked before the default locations (The passed path must be a directory!):

```bash
freqtrade trade --strategy AwesomeStrategy --strategy-path /some/directory
```

#### How to install a strategy?

This is very simple. Copy paste your strategy file into the directory
`user_data/strategies` or use `--strategy-path`. And voila, the bot is ready to use it.

### How to use **--db-url**?

When you run the bot in Dry-run mode, per default no transactions are
stored in a database. If you want to store your bot actions in a DB
using `--db-url`. This can also be used to specify a custom database
in production mode. Example command:

```bash
freqtrade trade -c config.json --db-url sqlite:///tradesv3.dry_run.sqlite
```

## Next step

The optimal strategy of the bot will change with time depending of the market trends. The next step is to
[Strategy Customization](strategy-customization.md).
