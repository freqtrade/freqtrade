# Development Help

This page is intended for developers of Freqtrade, people who want to contribute to the Freqtrade codebase or documentation, or people who want to understand the source code of the application they're running.

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome. We [track issues](https://github.com/freqtrade/freqtrade/issues) on [GitHub](https://github.com) and also have a dev channel on [discord](https://discord.gg/p7nuUNVfP7) where you can ask questions.

## Documentation

Documentation is available at [https://freqtrade.io](https://www.freqtrade.io/) and needs to be provided with every new feature PR.

Special fields for the documentation (like Note boxes, ...) can be found [here](https://squidfunk.github.io/mkdocs-material/reference/admonitions/).

To test the documentation locally use the following commands.

``` bash
pip install -r docs/requirements-docs.txt
mkdocs serve
```

This will spin up a local server (usually on port 8000) so you can see if everything looks as you'd like it to.

## Developer setup

To configure a development environment, you can either use the provided [DevContainer](#devcontainer-setup), or use the `setup.sh` script and answer "y" when asked "Do you want to install dependencies for dev [y/N]? ".
Alternatively (e.g. if your system is not supported by the setup.sh script), follow the manual installation process and run `pip3 install -e .[all]`.

This will install all required tools for development, including `pytest`, `ruff`, `mypy`, and `coveralls`.

Then install the git hook scripts by running `pre-commit install`, so your changes will be verified locally before committing.
This avoids a lot of waiting for CI already, as some basic formatting checks are done locally on your machine.

Before opening a pull request, please familiarize yourself with our [Contributing Guidelines](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md).

### Devcontainer setup

The fastest and easiest way to get started is to use [VSCode](https://code.visualstudio.com/) with the Remote container extension.
This gives developers the ability to start the bot with all required dependencies *without* needing to install any freqtrade specific dependencies on your local machine.

#### Devcontainer dependencies

* [VSCode](https://code.visualstudio.com/)
* [docker](https://docs.docker.com/install/)
* [Remote container extension documentation](https://code.visualstudio.com/docs/remote)

For more information about the [Remote container extension](https://code.visualstudio.com/docs/remote), best consult the documentation.

### Tests

New code should be covered by basic unittests. Depending on the complexity of the feature, Reviewers may request more in-depth unittests.
If necessary, the Freqtrade team can assist and give guidance with writing good tests (however please don't expect anyone to write the tests for you).

#### How to run tests

Use `pytest` in root folder to run all available testcases and confirm your local environment is setup correctly

!!! Note "feature branches"
    Tests are expected to pass on the `develop` and `stable` branches. Other branches may be work in progress with tests not working yet.

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

### Debug configuration

To debug freqtrade, we recommend VSCode with the following launch configuration (located in `.vscode/launch.json`).
Details will obviously vary between setups - but this should work to get you started.

``` json
{
    "name": "freqtrade trade",
    "type": "python",
    "request": "launch",
    "module": "freqtrade",
    "console": "integratedTerminal",
    "args": [
        "trade",
        // Optional:
        // "--userdir", "user_data",
        "--strategy", 
        "MyAwesomeStrategy",
    ]
},
```

Command line arguments can be added in the `"args"` array.
This method can also be used to debug a strategy, by setting the breakpoints within the strategy.

A similar setup can also be taken for Pycharm - using `freqtrade` as module name, and setting the command line arguments as "parameters".

!!! Note "Startup directory"
    This assumes that you have the repository checked out, and the editor is started at the repository root level (so setup.py is at the top level of your repository).

## ErrorHandling

Freqtrade Exceptions all inherit from `FreqtradeException`.
This general class of error should however not be used directly. Instead, multiple specialized sub-Exceptions exist.

Below is an outline of exception inheritance hierarchy:

```
+ FreqtradeException
|
+---+ OperationalException
|
+---+ DependencyException
|   |
|   +---+ PricingError
|   |
|   +---+ ExchangeError
|       |
|       +---+ TemporaryError
|       |
|       +---+ DDosProtection
|       |
|       +---+ InvalidOrderException
|           |
|           +---+ RetryableOrderError
|           |
|           +---+ InsufficientFundsError
|
+---+ StrategyError
```

---

## Plugins

### Pairlists

You have a great idea for a new pair selection algorithm you would like to try out? Great.
Hopefully you also want to contribute this back upstream.

Whatever your motivations are - This should get you off the ground in trying to develop a new Pairlist Handler.

First of all, have a look at the [VolumePairList](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/pairlist/VolumePairList.py) Handler, and best copy this file with a name of your new Pairlist Handler.

This is a simple Handler, which however serves as a good example on how to start developing.

Next, modify the class-name of the Handler (ideally align this with the module filename).

The base-class provides an instance of the exchange (`self._exchange`) the pairlist manager (`self._pairlistmanager`), as well as the main configuration (`self._config`), the pairlist dedicated configuration (`self._pairlistconfig`) and the absolute position within the list of pairlists.

```python
        self._exchange = exchange
        self._pairlistmanager = pairlistmanager
        self._config = config
        self._pairlistconfig = pairlistconfig
        self._pairlist_pos = pairlist_pos
```

!!! Tip
    Don't forget to register your pairlist in `constants.py` under the variable `AVAILABLE_PAIRLISTS` - otherwise it will not be selectable.

Now, let's step through the methods which require actions:

#### Pairlist configuration

Configuration for the chain of Pairlist Handlers is done in the bot configuration file in the element `"pairlists"`, an array of configuration parameters for each Pairlist Handlers in the chain.

By convention, `"number_assets"` is used to specify the maximum number of pairs to keep in the pairlist. Please follow this to ensure a consistent user experience.

Additional parameters can be configured as needed. For instance, `VolumePairList` uses `"sort_key"` to specify the sorting value - however feel free to specify whatever is necessary for your great algorithm to be successful and dynamic.

#### short_desc

Returns a description used for Telegram messages.

This should contain the name of the Pairlist Handler, as well as a short description containing the number of assets. Please follow the format `"PairlistName - top/bottom X pairs"`.

#### gen_pairlist

Override this method if the Pairlist Handler can be used as the leading Pairlist Handler in the chain, defining the initial pairlist which is then handled by all Pairlist Handlers in the chain. Examples are `StaticPairList` and `VolumePairList`.

This is called with each iteration of the bot (only if the Pairlist Handler is at the first location) - so consider implementing caching for compute/network heavy calculations.

It must return the resulting pairlist (which may then be passed into the chain of Pairlist Handlers).

Validations are optional, the parent class exposes a `_verify_blacklist(pairlist)` and `_whitelist_for_active_markets(pairlist)` to do default filtering. Use this if you limit your result to a certain number of pairs - so the end-result is not shorter than expected.

#### filter_pairlist

This method is called for each Pairlist Handler in the chain by the pairlist manager.

This is called with each iteration of the bot - so consider implementing caching for compute/network heavy calculations.

It gets passed a pairlist (which can be the result of previous pairlists) as well as `tickers`, a pre-fetched version of `get_tickers()`.

The default implementation in the base class simply calls the `_validate_pair()` method for each pair in the pairlist, but you may override it. So you should either implement the `_validate_pair()` in your Pairlist Handler or override `filter_pairlist()` to do something else.

If overridden, it must return the resulting pairlist (which may then be passed into the next Pairlist Handler in the chain).

Validations are optional, the parent class exposes a `_verify_blacklist(pairlist)` and `_whitelist_for_active_markets(pairlist)` to do default filters. Use this if you limit your result to a certain number of pairs - so the end result is not shorter than expected.

In `VolumePairList`, this implements different methods of sorting, does early validation so only the expected number of pairs is returned.

##### sample

``` python
    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        # Generate dynamic whitelist
        pairs = self._calculate_pairlist(pairlist, tickers)
        return pairs
```

### Protections

Best read the [Protection documentation](plugins.md#protections) to understand protections.
This Guide is directed towards Developers who want to develop a new protection.

No protection should use datetime directly, but use the provided `date_now` variable for date calculations. This preserves the ability to backtest protections.

!!! Tip "Writing a new Protection"
    Best copy one of the existing Protections to have a good example.
    Don't forget to register your protection in `constants.py` under the variable `AVAILABLE_PROTECTIONS` - otherwise it will not be selectable.

#### Implementation of a new protection

All Protection implementations must have `IProtection` as parent class.
For that reason, they must implement the following methods:

* `short_desc()`
* `global_stop()`
* `stop_per_pair()`.

`global_stop()` and `stop_per_pair()` must return a ProtectionReturn object, which consists of:

* lock pair - boolean
* lock until - datetime - until when should the pair be locked (will be rounded up to the next new candle)
* reason - string, used for logging and storage in the database
* lock_side - long, short or '*'.

The `until` portion should be calculated using the provided `calculate_lock_end()` method.

All Protections should use `"stop_duration"` / `"stop_duration_candles"` to define how long a a pair (or all pairs) should be locked.
The content of this is made available as `self._stop_duration` to the each Protection.

If your protection requires a look-back period, please use `"lookback_period"` / `"lockback_period_candles"` to keep all protections aligned.

#### Global vs. local stops

Protections can have 2 different ways to stop trading for a limited :

* Per pair (local)
* For all Pairs (globally)

##### Protections - per pair

Protections that implement the per pair approach must set `has_local_stop=True`.
The method `stop_per_pair()` will be called whenever a trade closed (exit order completed).

##### Protections - global protection

These Protections should do their evaluation across all pairs, and consequently will also lock all pairs from trading (called a global PairLock).
Global protection must set `has_global_stop=True` to be evaluated for global stops.
The method `global_stop()` will be called whenever a trade closed (exit order completed).

##### Protections - calculating lock end time

Protections should calculate the lock end time based on the last trade it considers.
This avoids re-locking should the lookback-period be longer than the actual lock period.

The `IProtection` parent class provides a helper method for this in `calculate_lock_end()`.

---

## Implement a new Exchange (WIP)

!!! Note
    This section is a Work in Progress and is not a complete guide on how to test a new exchange with Freqtrade.

!!! Note
    Make sure to use an up-to-date version of CCXT before running any of the below tests.
    You can get the latest version of ccxt by running `pip install -U ccxt` with activated virtual environment.
    Native docker is not supported for these tests, however the available dev-container will support all required actions and eventually necessary changes.

Most exchanges supported by CCXT should work out of the box.

To quickly test the public endpoints of an exchange, add a configuration for your exchange to `test_ccxt_compat.py` and run these tests with `pytest --longrun tests/exchange/test_ccxt_compat.py`.
Completing these tests successfully a good basis point (it's a requirement, actually), however these won't guarantee correct exchange functioning, as this only tests public endpoints, but no private endpoint (like generate order or similar).

Also try to use `freqtrade download-data` for an extended timerange (multiple months) and verify that the data downloaded correctly (no holes, the specified timerange was actually downloaded).

These are prerequisites to have an exchange listed as either Supported or Community tested (listed on the homepage).
The below are "extras", which will make an exchange better (feature-complete) - but are not absolutely necessary for either of the 2 categories.

Additional tests / steps to complete:

* Verify data provided by `fetch_ohlcv()` - and eventually adjust `ohlcv_candle_limit` for this exchange
* Check L2 orderbook limit range (API documentation) - and eventually set as necessary
* Check if balance shows correctly (*)
* Create market order (*)
* Create limit order (*)
* Complete trade (enter + exit) (*)
  * Compare result calculation between exchange and bot
  * Ensure fees are applied correctly (check the database against the exchange)

(*) Requires API keys and Balance on the exchange.

### Stoploss On Exchange

Check if the new exchange supports Stoploss on Exchange orders through their API.

Since CCXT does not provide unification for Stoploss On Exchange yet, we'll need to implement the exchange-specific parameters ourselves. Best look at `binance.py` for an example implementation of this. You'll need to dig through the documentation of the Exchange's API on how exactly this can be done. [CCXT Issues](https://github.com/ccxt/ccxt/issues) may also provide great help, since others may have implemented something similar for their projects.

### Incomplete candles

While fetching candle (OHLCV) data, we may end up getting incomplete candles (depending on the exchange).
To demonstrate this, we'll use daily candles (`"1d"`) to keep things simple.
We query the api (`ct.fetch_ohlcv()`) for the timeframe and look at the date of the last entry. If this entry changes or shows the date of a "incomplete" candle, then we should drop this since having incomplete candles is problematic because indicators assume that only complete candles are passed to them, and will generate a lot of false buy signals. By default, we're therefore removing the last candle assuming it's incomplete.

To check how the new exchange behaves, you can use the following snippet:

``` python
import ccxt
from datetime import datetime, timezone
from freqtrade.data.converter import ohlcv_to_dataframe
ct = ccxt.binance()  # Use the exchange you're testing
timeframe = "1d"
pair = "BTC/USDT"  # Make sure to use a pair that exists on that exchange!
raw = ct.fetch_ohlcv(pair, timeframe=timeframe)

# convert to dataframe
df1 = ohlcv_to_dataframe(raw, timeframe, pair=pair, drop_incomplete=False)

print(df1.tail(1))
print(datetime.now(timezone.utc))
```

``` output
                         date      open      high       low     close  volume  
499 2019-06-08 00:00:00+00:00  0.000007  0.000007  0.000007  0.000007   26264344.0  
2019-06-09 12:30:27.873327
```

The output will show the last entry from the Exchange as well as the current UTC date.
If the day shows the same day, then the last candle can be assumed as incomplete and should be dropped (leave the setting `"ohlcv_partial_candle"` from the exchange-class untouched / True). Otherwise, set `"ohlcv_partial_candle"` to `False` to not drop Candles (shown in the example above).
Another way is to run this command multiple times in a row and observe if the volume is changing (while the date remains the same).

### Update binance cached leverage tiers

Updating leveraged tiers should be done regularly - and requires an authenticated account with futures enabled.

``` python
import ccxt
import json
from pathlib import Path

exchange = ccxt.binance({
    'apiKey': '<apikey>',
    'secret': '<secret>'
    'options': {'defaultType': 'swap'}
    })
_ = exchange.load_markets()

lev_tiers = exchange.fetch_leverage_tiers()

# Assumes this is running in the root of the repository.
file = Path('freqtrade/exchange/binance_leverage_tiers.json')
json.dump(dict(sorted(lev_tiers.items())), file.open('w'), indent=2)

```

This file should then be contributed upstream, so others can benefit from this, too.

## Updating example notebooks

To keep the jupyter notebooks aligned with the documentation, the following should be ran after updating a example notebook.

``` bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace freqtrade/templates/strategy_analysis_example.ipynb
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown freqtrade/templates/strategy_analysis_example.ipynb --stdout > docs/strategy_analysis_example.md
```

## Continuous integration

This documents some decisions taken for the CI Pipeline.

* CI runs on all OS variants, Linux (ubuntu), macOS and Windows.
* Docker images are build for the branches `stable` and `develop`, and are built as multiarch builds, supporting multiple platforms via the same tag.
* Docker images containing Plot dependencies are also available as `stable_plot` and `develop_plot`.
* Docker images contain a file, `/freqtrade/freqtrade_commit` containing the commit this image is based of.
* Full docker image rebuilds are run once a week via schedule.
* Deployments run on ubuntu.
* ta-lib binaries are contained in the build_helpers directory to avoid fails related to external unavailability.
* All tests must pass for a PR to be merged to `stable` or `develop`.

## Creating a release

This part of the documentation is aimed at maintainers, and shows how to create a release.

### Create release branch

First, pick a commit that's about one week old (to not include latest additions to releases).

``` bash
# create new branch
git checkout -b new_release <commitid>
```

Determine if crucial bugfixes have been made between this commit and the current state, and eventually cherry-pick these.

* Merge the release branch (stable) into this branch.
* Edit `freqtrade/__init__.py` and add the version matching the current date (for example `2019.7` for July 2019). Minor versions can be `2019.7.1` should we need to do a second release that month. Version numbers must follow allowed versions from PEP0440 to avoid failures pushing to pypi.
* Commit this part.
* push that branch to the remote and create a PR against the stable branch.
* Update develop version to next version following the pattern `2019.8-dev`.

### Create changelog from git commits

!!! Note
    Make sure that the `stable` branch is up-to-date!

``` bash
# Needs to be done before merging / pulling that branch.
git log --oneline --no-decorate --no-merges stable..new_release
```

To keep the release-log short, best wrap the full git changelog into a collapsible details section.

```markdown
<details>
<summary>Expand full changelog</summary>

... Full git changelog

</details>
```

### FreqUI release

If FreqUI has been updated substantially, make sure to create a release before merging the release branch.
Make sure that freqUI CI on the release is finished and passed before merging the release.

### Create github release / tag

Once the PR against stable is merged (best right after merging):

* Use the button "Draft a new release" in the Github UI (subsection releases).
* Use the version-number specified as tag.
* Use "stable" as reference (this step comes after the above PR is merged).
* Use the above changelog as release comment (as codeblock)

## Releases

### pypi

!!! Note
    This process is now automated as part of Github Actions.

To create a pypi release, please run the following commands:

Additional requirement: `wheel`, `twine` (for uploading), account on pypi with proper permissions.

``` bash
python setup.py sdist bdist_wheel

# For pypi test (to check if some change to the installation did work)
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# For production:
twine upload dist/*
```

Please don't push non-releases to the productive / real pypi instance.
