# Development Help

This page is intended for developers of FreqTrade, people who want to contribute to the FreqTrade codebase or documentation, or people who want to understand the source code of the application they're running.

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome. We [track issues](https://github.com/freqtrade/freqtrade/issues) on [GitHub](https://github.com) and also have a dev channel in [slack](https://join.slack.com/t/highfrequencybot/shared_invite/enQtNjU5ODcwNjI1MDU3LWEyODBiNzkzNzcyNzU0MWYyYzE5NjIyOTQxMzBmMGUxOTIzM2YyN2Y4NWY1YTEwZDgwYTRmMzE2NmM5ZmY2MTg) where you can ask questions.

## Documentation

Documentation is available at [https://freqtrade.io](https://www.freqtrade.io/) and needs to be provided with every new feature PR.

Special fields for the documentation (like Note boxes, ...) can be found [here](https://squidfunk.github.io/mkdocs-material/extensions/admonition/).

## Developer setup

To configure a development environment, best use the `setup.sh` script and answer "y" when asked "Do you want to install dependencies for dev [y/N]? ".
Alternatively (if your system is not supported by the setup.sh script), follow the manual installation process and run `pip3 install -e .[all]`.

This will install all required tools for development, including `pytest`, `flake8`, `mypy`, and `coveralls`.

### Tests

New code should be covered by basic unittests. Depending on the complexity of the feature, Reviewers may request more in-depth unittests.
If necessary, the Freqtrade team can assist and give guidance with writing good tests (however please don't expect anyone to write the tests for you).

#### Checking log content in tests

Freqtrade uses 2 main methods to check log content in tests, `log_has()` and `log_has_re()` (to check using regex, in case of dynamic log-messages).
These are available from `conftest.py` and can be imported in any test module.

A sample check looks as follows:

``` python
from freqtrade.tests.conftest import log_has, log_has_re

def test_method_to_test(caplog):
    method_to_test()

    assert log_has("This event happened", caplog)
    # Check regex with trailing number ...
    assert log_has_re(r"This dynamic event happened and produced \d+", caplog)
```

## Modules

### Dynamic Pairlist

You have a great idea for a new pair selection algorithm you would like to try out? Great.
Hopefully you also want to contribute this back upstream.

Whatever your motivations are - This should get you off the ground in trying to develop a new Pairlist provider.

First of all, have a look at the [VolumePairList](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/pairlist/VolumePairList.py) provider, and best copy this file with a name of your new Pairlist Provider.

This is a simple provider, which however serves as a good example on how to start developing.

Next, modify the classname of the provider (ideally align this with the Filename).

The base-class provides the an instance of the bot (`self._freqtrade`), as well as the configuration (`self._config`), and initiates both `_blacklist` and `_whitelist`.

```python
        self._freqtrade = freqtrade
        self._config = config
        self._whitelist = self._config['exchange']['pair_whitelist']
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
```


Now, let's step through the methods which require actions:

#### configuration

Configuration for PairListProvider is done in the bot configuration file in the element `"pairlist"`.
This Pairlist-object may contain a `"config"` dict with additional configurations for the configured pairlist.
By convention, `"number_assets"` is used to specify the maximum number of pairs to keep in the whitelist. Please follow this to ensure a consistent user experience.

Additional elements can be configured as needed. `VolumePairList` uses `"sort_key"` to specify the sorting value - however feel free to specify whatever is necessary for your great algorithm to be successfull and dynamic.

#### short_desc

Returns a description used for Telegram messages.
This should contain the name of the Provider, as well as a short description containing the number of assets. Please follow the format `"PairlistName - top/bottom X pairs"`.

#### refresh_pairlist

Override this method and run all calculations needed in this method.
This is called with each iteration of the bot - so consider implementing caching for compute/network heavy calculations.

Assign the resulting whiteslist to `self._whitelist` and `self._blacklist` respectively. These will then be used to run the bot in this iteration. Pairs with open trades will be added to the whitelist to have the sell-methods run correctly.

Please also run `self._validate_whitelist(pairs)` and to check and remove pairs with inactive markets. This function is available in the Parent class (`StaticPairList`) and should ideally not be overwritten.

##### sample

``` python
    def refresh_pairlist(self) -> None:
        # Generate dynamic whitelist
        pairs = self._gen_pair_whitelist(self._config['stake_currency'], self._sort_key)
        # Validate whitelist to only have active market pairs
        self._whitelist = self._validate_whitelist(pairs)[:self._number_pairs]
```

#### _gen_pair_whitelist

This is a simple method used by `VolumePairList` - however serves as a good example.
It implements caching (`@cached(TTLCache(maxsize=1, ttl=1800))`) as well as a configuration option to allow different (but similar) strategies to work with the same PairListProvider.

## Implement a new Exchange (WIP)

!!! Note
    This section is a Work in Progress and is not a complete guide on how to test a new exchange with FreqTrade.

Most exchanges supported by CCXT should work out of the box.

### Stoploss On Exchange

Check if the new exchange supports Stoploss on Exchange orders through their API.

Since CCXT does not provide unification for Stoploss On Exchange yet, we'll need to implement the exchange-specific parameters ourselfs. Best look at `binance.py` for an example implementation of this. You'll need to dig through the documentation of the Exchange's API on how exactly this can be done. [CCXT Issues](https://github.com/ccxt/ccxt/issues) may also provide great help, since others may have implemented something similar for their projects.

### Incomplete candles

While fetching OHLCV data, we're may end up getting incomplete candles (Depending on the exchange).
To demonstrate this, we'll use daily candles (`"1d"`) to keep things simple.
We query the api (`ct.fetch_ohlcv()`) for the timeframe and look at the date of the last entry. If this entry changes or shows the date of a "incomplete" candle, then we should drop this since having incomplete candles is problematic because indicators assume that only complete candles are passed to them, and will generate a lot of false buy signals. By default, we're therefore removing the last candle assuming it's incomplete.

To check how the new exchange behaves, you can use the following snippet:

``` python
import ccxt
from datetime import datetime
from freqtrade.data.converter import parse_ticker_dataframe
ct = ccxt.binance()
timeframe = "1d"
pair = "XLM/BTC"  # Make sure to use a pair that exists on that exchange!
raw = ct.fetch_ohlcv(pair, timeframe=timeframe)

# convert to dataframe
df1 = parse_ticker_dataframe(raw, timeframe, pair=pair, drop_incomplete=False)

print(df1["date"].tail(1))
print(datetime.utcnow())
```

``` output
19   2019-06-08 00:00:00+00:00
2019-06-09 12:30:27.873327
```

The output will show the last entry from the Exchange as well as the current UTC date.
If the day shows the same day, then the last candle can be assumed as incomplete and should be dropped (leave the setting `"ohlcv_partial_candle"` from the exchange-class untouched / True). Otherwise, set `"ohlcv_partial_candle"` to `False` to not drop Candles (shown in the example above).

## Creating a release

This part of the documentation is aimed at maintainers, and shows how to create a release.

### Create release branch

``` bash
# make sure you're in develop branch
git checkout develop

# create new branch
git checkout -b new_release
```

* Edit `freqtrade/__init__.py` and add the version matching the current date (for example `2019.7` for July 2019). Minor versions can be `2019.7-1` should we need to do a second release that month.
* Commit this part
* push that branch to the remote and create a PR against the master branch

### Create changelog from git commits

!!! Note
    Make sure that both master and develop are up-todate!.

``` bash
# Needs to be done before merging / pulling that branch.
git log --oneline --no-decorate --no-merges master..develop
```

### Create github release / tag

Once the PR against master is merged (best right after merging):

* Use the button "Draft a new release" in the Github UI (subsection releases)
* Use the version-number specified as tag. 
* Use "master" as reference (this step comes after the above PR is merged).
* Use the above changelog as release comment (as codeblock)

### After-release

* Update version in develop by postfixing that with `-dev` (`2019.6 -> 2019.6-dev`).
* Create a PR against develop to update that branch.
