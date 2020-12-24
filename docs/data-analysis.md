# Analyzing bot data with Jupyter notebooks  

You can analyze the results of backtests and trading history easily using Jupyter notebooks. Sample notebooks are located at `user_data/notebooks/` after initializing the user directory with `freqtrade create-userdir --userdir user_data`.  

## Quick start with docker

Freqtrade provides a docker-compose file which starts up a jupyter lab server.
You can run this server using the following command: `docker-compose -f docker/docker-compose-jupyter.yml up`

This will create a dockercontainer running jupyter lab, which will be accessible using `https://127.0.0.1:8888/lab`.
Please use the link that's printed in the console after startup for simplified login.

For more information, Please visit the [Data analysis with Docker](docker_quickstart.md#data-analayis-using-docker-compose) section.

### Pro tips

* See [jupyter.org](https://jupyter.org/documentation) for usage instructions.
* Don't forget to start a Jupyter notebook server from within your conda or venv environment or use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels)*
* Copy the example notebook before use so your changes don't get overwritten with the next freqtrade update.

### Using virtual environment with system-wide Jupyter installation

Sometimes it can be desired to use a system-wide installation of Jupyter notebook, and use a jupyter kernel from the virtual environment.
This prevents you from installing the full jupyter suite multiple times per system, and provides an easy way to switch between tasks (freqtrade / other analytics tasks).

For this to work, first activate your virtual environment and run the following commands:

``` bash
# Activate virtual environment
source .env/bin/activate

pip install ipykernel
ipython kernel install --user --name=freqtrade
# Restart jupyter (lab / notebook)
# select kernel "freqtrade" in the notebook
```

!!! Note
    This section is provided for completeness, the Freqtrade Team won't provide full support for problems with this setup and will recommend to install Jupyter in the virtual environment directly, as that is the easiest way to get jupyter notebooks up and running. For help with this setup please refer to the [Project Jupyter](https://jupyter.org/) [documentation](https://jupyter.org/documentation) or [help channels](https://jupyter.org/community).

!!! Warning
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

### Further Data analysis documentation

* [Strategy debugging](strategy_analysis_example.md) - also available as Jupyter notebook (`user_data/notebooks/strategy_analysis_example.ipynb`)
* [Plotting](plotting.md)

Feel free to submit an issue or Pull Request enhancing this document if you would like to share ideas on how to best analyze the data.
