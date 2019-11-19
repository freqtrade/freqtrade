# Development Help

This page is intended for developers of FreqTrade, people who want to contribute to the FreqTrade codebase or documentation, or people who want to understand the source code of the application they're running.

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome. We [track issues](https://github.com/freqtrade/freqtrade/issues) on [GitHub](https://github.com) and also have a dev channel in [slack](https://join.slack.com/t/highfrequencybot/shared_invite/enQtNjU5ODcwNjI1MDU3LTU1MTgxMjkzNmYxNWE1MDEzYzQ3YmU4N2MwZjUyNjJjODRkMDVkNjg4YTAyZGYzYzlhOTZiMTE4ZjQ4YzM0OGE) where you can ask questions.

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
from tests.conftest import log_has, log_has_re

def test_method_to_test(caplog):
    method_to_test()

    assert log_has("This event happened", caplog)
    # Check regex with trailing number ...
    assert log_has_re(r"This dynamic event happened and produced \d+", caplog)

```

### Local docker usage

The fastest and easiest way to start up is to use docker-compose.develop which gives developers the ability to start the bot up with all the required dependencies, *without* needing to install any freqtrade specific dependencies on your local machine.

#### Install

* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [docker](https://docs.docker.com/install/)
* [docker-compose](https://docs.docker.com/compose/install/)

#### Starting the bot
##### Use the develop dockerfile

``` bash
rm docker-compose.yml && mv docker-compose.develop.yml docker-compose.yml
```

#### Docker Compose

##### Starting

``` bash
docker-compose up
```

![Docker compose up](https://user-images.githubusercontent.com/419355/65456322-47f63a80-de06-11e9-90c6-3c74d1bad0b8.png)

##### Rebuilding

``` bash
docker-compose build
```

##### Execing (effectively SSH into the container)

The `exec` command requires that the container already be running, if you want to start it
that can be effected by `docker-compose up` or `docker-compose run freqtrade_develop`

``` bash
docker-compose exec freqtrade_develop /bin/bash
```

![image](https://user-images.githubusercontent.com/419355/65456522-ba671a80-de06-11e9-9598-df9ca0d8dcac.png)

## Modules

### Dynamic Pairlist

You have a great idea for a new pair selection algorithm you would like to try out? Great.
Hopefully you also want to contribute this back upstream.

Whatever your motivations are - This should get you off the ground in trying to develop a new Pairlist provider.

First of all, have a look at the [VolumePairList](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/pairlist/VolumePairList.py) provider, and best copy this file with a name of your new Pairlist Provider.

This is a simple provider, which however serves as a good example on how to start developing.

Next, modify the classname of the provider (ideally align this with the Filename).

The base-class provides an instance of the exchange (`self._exchange`) the pairlist manager (`self._pairlistmanager`), as well as the main configuration (`self._config`), the pairlist dedicated configuration (`self._pairlistconfig`) and the absolute position within the list of pairlists.

```python
        self._exchange = exchange
        self._pairlistmanager = pairlistmanager
        self._config = config
        self._pairlistconfig = pairlistconfig
        self._pairlist_pos = pairlist_pos
```

Now, let's step through the methods which require actions:

#### Pairlist configuration

Configuration for PairListProvider is done in the bot configuration file in the element `"pairlist"`.
This Pairlist-object may contain configurations with additional configurations for the configured pairlist.
By convention, `"number_assets"` is used to specify the maximum number of pairs to keep in the whitelist. Please follow this to ensure a consistent user experience.

Additional elements can be configured as needed. `VolumePairList` uses `"sort_key"` to specify the sorting value - however feel free to specify whatever is necessary for your great algorithm to be successfull and dynamic.

#### short_desc

Returns a description used for Telegram messages.
This should contain the name of the Provider, as well as a short description containing the number of assets. Please follow the format `"PairlistName - top/bottom X pairs"`.

#### filter_pairlist

Override this method and run all calculations needed in this method.
This is called with each iteration of the bot - so consider implementing caching for compute/network heavy calculations.

It get's passed a pairlist (which can be the result of previous pairlists) as well as `tickers`, a pre-fetched version of `get_tickers()`.

It must return the resulting pairlist (which may then be passed into the next pairlist filter).

Validations are optional, the parent class exposes a `_verify_blacklist(pairlist)` and `_whitelist_for_active_markets(pairlist)` to do default filters. Use this if you limit your result to a certain number of pairs - so the endresult is not shorter than expected.

##### sample

``` python
    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        # Generate dynamic whitelist
        pairs = self._calculate_pairlist(pairlist, tickers)
        return pairs
```

#### _gen_pair_whitelist

This is a simple method used by `VolumePairList` - however serves as a good example.
In VolumePairList, this implements different methods of sorting, does early validation so only the expected number of pairs is returned.

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

## Updating example notebooks

To keep the jupyter notebooks aligned with the documentation, the following should be ran after updating a example notebook.

``` bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace user_data/notebooks/strategy_analysis_example.ipynb
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown user_data/notebooks/strategy_analysis_example.ipynb --stdout > docs/strategy_analysis_example.md
```

## Creating a release

This part of the documentation is aimed at maintainers, and shows how to create a release.

### Create release branch

First, pick a commit that's about one week old (to not include latest additions to releases).

``` bash
# create new branch
git checkout -b new_release <commitid>
```

Determine if crucial bugfixes have been made between this commit and the current state, and eventually cherry-pick these.

* Edit `freqtrade/__init__.py` and add the version matching the current date (for example `2019.7` for July 2019). Minor versions can be `2019.7-1` should we need to do a second release that month.
* Commit this part
* push that branch to the remote and create a PR against the master branch

### Create changelog from git commits

!!! Note
    Make sure that the master branch is uptodate!

``` bash
# Needs to be done before merging / pulling that branch.
git log --oneline --no-decorate --no-merges master..new_release
```

### Create github release / tag

Once the PR against master is merged (best right after merging):

* Use the button "Draft a new release" in the Github UI (subsection releases).
* Use the version-number specified as tag.
* Use "master" as reference (this step comes after the above PR is merged).
* Use the above changelog as release comment (as codeblock).
