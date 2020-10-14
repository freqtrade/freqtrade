# Development Help

This page is intended for developers of Freqtrade, people who want to contribute to the Freqtrade codebase or documentation, or people who want to understand the source code of the application they're running.

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome. We [track issues](https://github.com/freqtrade/freqtrade/issues) on [GitHub](https://github.com) and also have a dev channel on [discord](https://discord.gg/MA9v74M) or [slack](https://join.slack.com/t/highfrequencybot/shared_invite/zt-jaut7r4m-Y17k4x5mcQES9a9swKuxbg) where you can ask questions.

## Documentation

Documentation is available at [https://freqtrade.io](https://www.freqtrade.io/) and needs to be provided with every new feature PR.

Special fields for the documentation (like Note boxes, ...) can be found [here](https://squidfunk.github.io/mkdocs-material/extensions/admonition/).

To test the documentation locally use the following commands.

``` bash
pip install -r docs/requirements-docs.txt
mkdocs serve
```

This will spin up a local server (usually on port 8000) so you can see if everything looks as you'd like it to.

## Developer setup

To configure a development environment, you can either use the provided [DevContainer](#devcontainer-setup), or use the `setup.sh` script and answer "y" when asked "Do you want to install dependencies for dev [y/N]? ".
Alternatively (e.g. if your system is not supported by the setup.sh script), follow the manual installation process and run `pip3 install -e .[all]`.

This will install all required tools for development, including `pytest`, `flake8`, `mypy`, and `coveralls`.

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

## Modules

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

!!! Note
    You'll need to register your pairlist in `constants.py` under the variable `AVAILABLE_PAIRLISTS` - otherwise it will not be selectable.

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

## Implement a new Exchange (WIP)

!!! Note
    This section is a Work in Progress and is not a complete guide on how to test a new exchange with Freqtrade.

Most exchanges supported by CCXT should work out of the box.

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
from datetime import datetime
from freqtrade.data.converter import ohlcv_to_dataframe
ct = ccxt.binance()
timeframe = "1d"
pair = "XLM/BTC"  # Make sure to use a pair that exists on that exchange!
raw = ct.fetch_ohlcv(pair, timeframe=timeframe)

# convert to dataframe
df1 = ohlcv_to_dataframe(raw, timeframe, pair=pair, drop_incomplete=False)

print(df1.tail(1))
print(datetime.utcnow())
```

``` output
                         date      open      high       low     close  volume  
499 2019-06-08 00:00:00+00:00  0.000007  0.000007  0.000007  0.000007   26264344.0  
2019-06-09 12:30:27.873327
```

The output will show the last entry from the Exchange as well as the current UTC date.
If the day shows the same day, then the last candle can be assumed as incomplete and should be dropped (leave the setting `"ohlcv_partial_candle"` from the exchange-class untouched / True). Otherwise, set `"ohlcv_partial_candle"` to `False` to not drop Candles (shown in the example above).
Another way is to run this command multiple times in a row and observe if the volume is changing (while the date remains the same).

## Updating example notebooks

To keep the jupyter notebooks aligned with the documentation, the following should be ran after updating a example notebook.

``` bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace freqtrade/templates/strategy_analysis_example.ipynb
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown freqtrade/templates/strategy_analysis_example.ipynb --stdout > docs/strategy_analysis_example.md
```

## Continuous integration

This documents some decisions taken for the CI Pipeline.

* CI runs on all OS variants, Linux (ubuntu), macOS and Windows.
* Docker images are build for the branches `stable` and `develop`.
* Docker images containing Plot dependencies are also available as `stable_plot` and `develop_plot`.
* Raspberry PI Docker images are postfixed with `_pi` - so tags will be `:stable_pi` and `develop_pi`.
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
* Commit this part
* push that branch to the remote and create a PR against the stable branch

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
