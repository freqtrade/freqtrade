# Analyzing bot data with Jupyter notebooks  

You can analyze the results of backtests and trading history easily using Jupyter notebooks. Sample notebooks are located at `user_data/notebooks/`.  

## Pro tips  

* See [jupyter.org](https://jupyter.org/documentation) for usage instructions.
* Don't forget to start a Jupyter notebook server from within your conda or venv environment or use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels)*
* Copy the example notebook before use so your changes don't get clobbered with the next freqtrade update.

## Fine print  

Some tasks don't work especially well in notebooks. For example, anything using asynchronous execution is a problem for Jupyter. Also, freqtrade's primary entry point is the shell cli, so using pure python in a notebook bypasses arguments that provide required objects and parameters to helper functions. You may need to set those values or create expected objects manually.

## Recommended workflow  

| Task | Tool |  
  --- | ---  
Bot operations | CLI  
Repetitive tasks | Shell scripts
Data analysis & visualization | Notebook  

1. Use the CLI to
    * download historical data
    * run a backtest
    * run with real-time data
    * export results  

1. Collect these actions in shell scripts
    * save complicated commands with arguments
    * execute multi-step operations  
    * automate testing strategies and preparing data for analysis

1. Use a notebook to
    * visualize data
    * munge and plot to generate insights

## Example utility snippets  

### Change directory to root  

Jupyter notebooks execute from the notebook directory. The following snippet searches for the project root, so relative paths remain consistent.

```python
import os
from pathlib import Path

# Change directory
# Modify this cell to insure that the output shows the correct path.
# Define all paths relative to the project root shown in the cell output
project_root = "somedir/freqtrade"
i=0
try:
    os.chdirdir(project_root)
    assert Path('LICENSE').is_file()
except:
    while i<4 and (not Path('LICENSE').is_file()):
        os.chdir(Path(Path.cwd(), '../'))
        i+=1
    project_root = Path.cwd()
print(Path.cwd())
```

## Load existing objects into a Jupyter notebook

These examples assume that you have already generated data using the cli. They will allow you to drill deeper into your results, and perform analysis which otherwise would make the output very difficult to digest due to information overload.

### Load backtest results into a pandas dataframe

```python
from freqtrade.data.btanalysis import load_backtest_data

# Load backtest results
df = load_backtest_data("user_data/backtest_results/backtest-result.json")

# Show value-counts per pair
df.groupby("pair")["sell_reason"].value_counts()
```

### Load live trading results into a pandas dataframe

``` python
from freqtrade.data.btanalysis import load_trades_from_db

# Fetch trades from database
df = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Display results
df.groupby("pair")["sell_reason"].value_counts()
```

### Load multiple configuration files

This option can be useful to inspect the results of passing in multiple configs.
This will also run through the whole Configuration initialization, so the configuration is completely initialized to be passed to other methods.

``` python
import json
from freqtrade.configuration import Configuration

# Load config from multiple files
config = Configuration.from_files(["config1.json", "config2.json"])

# Show the config in memory
print(json.dumps(config['original_config'], indent=2))
```

For Interactive environments, have an additional configuration specifying `user_data_dir` and pass this in last, so you don't have to change directories while running the bot.
Best avoid relative paths, since this starts at the storage location of the jupyter notebook, unless the directory is changed.

``` json
{
    "user_data_dir": "~/.freqtrade/"
}
```

### Load exchange data to a pandas dataframe

This loads candle data to a dataframe

```python
from pathlib import Path
from freqtrade.data.history import load_pair_history

# Load data using values passed to function
ticker_interval = "5m"
data_location = Path('user_data', 'data', 'bitrex')
pair = "BTC_USDT"
candles = load_pair_history(datadir=data_location,
                            ticker_interval=ticker_interval,
                            pair=pair)

# Confirm success
print(f"Loaded len(candles) rows of data for {pair} from {data_location}")
candles.head()
```

Further Data analysis documents:

* [Strategy debugging](strategy_analysis_example.md)
* [Plotting](plotting.md)

Feel free to submit an issue or Pull Request enhancing this document if you would like to share ideas on how to best analyze the data.
