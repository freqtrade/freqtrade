# Using Freqtrade with Docker

This page explains how to run the bot with Docker. It is not meant to work out of the box. You'll still need to read through the documentation and understand how to properly configure it.

## Install Docker

Start by downloading and installing Docker CE for your platform:

* [Mac](https://docs.docker.com/docker-for-mac/install/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)
* [Linux](https://docs.docker.com/install/)

To simplify running freqtrade, [`docker-compose`](https://docs.docker.com/compose/install/) should be installed and available to follow the below [docker quick start guide](#docker-quick-start).

## Freqtrade with docker-compose

Freqtrade provides an official Docker image on [Dockerhub](https://hub.docker.com/r/freqtradeorg/freqtrade/), as well as a [docker-compose file](https://github.com/freqtrade/freqtrade/blob/stable/docker-compose.yml) ready for usage.

!!! Note
    - The following section assumes that `docker` and `docker-compose` are installed and available to the logged in user.
    - All below commands use relative directories and will have to be executed from the directory containing the `docker-compose.yml` file.

### Docker quick start

Create a new directory and place the [docker-compose file](https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml) in this directory.

``` bash
mkdir ft_userdata
cd ft_userdata/
# Download the docker-compose file from the repository
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml

# Pull the freqtrade image
docker-compose pull

# Create user directory structure
docker-compose run --rm freqtrade create-userdir --userdir user_data

# Create configuration - Requires answering interactive questions
docker-compose run --rm freqtrade new-config --config user_data/config.json
```

The above snippet creates a new directory called `ft_userdata`, downloads the latest compose file and pulls the freqtrade image.
The last 2 steps in the snippet create the directory with `user_data`, as well as (interactively) the default configuration based on your selections.

!!! Question "How to edit the bot configuration?"
    You can edit the configuration at any time, which is available as `user_data/config.json` (within the directory `ft_userdata`) when using the above configuration.

    You can also change the both Strategy and commands by editing the command section of your `docker-compose.yml` file.

#### Adding a custom strategy

1. The configuration is now available as `user_data/config.json`
2. Copy a custom strategy to the directory `user_data/strategies/`
3. Add the Strategy' class name to the `docker-compose.yml` file

The `SampleStrategy` is run by default.

!!! Danger "`SampleStrategy` is just a demo!"
    The `SampleStrategy` is there for your reference and give you ideas for your own strategy.
    Please always backtest your strategy and use dry-run for some time before risking real money!
    You will find more information about Strategy development in the [Strategy documentation](strategy-customization.md).

Once this is done, you're ready to launch the bot in trading mode (Dry-run or Live-trading, depending on your answer to the corresponding question you made above).

``` bash
docker-compose up -d
```

!!! Warning "Default configuration"
    While the configuration generated will be mostly functional, you will still need to verify that all options correspond to what you want (like Pricing, pairlist, ...) before starting the bot.

#### Accessing the UI

If you've selected to enable FreqUI in the `new-config` step, you will have freqUI available at port `localhost:8080`.

You can now access the UI by typing localhost:8080 in your browser.

??? Note "UI Access on a remote servers"
    If you're running on a VPS, you should consider using either a ssh tunnel, or setup a VPN (openVPN, wireguard) to connect to your bot.
    This will ensure that freqUI is not directly exposed to the internet, which is not recommended for security reasons (freqUI does not support https out of the box).
    Setup of these tools is not part of this tutorial, however many good tutorials can be found on the internet.
    Please also read the [API configuration with docker](rest-api.md#configuration-with-docker) section to learn more about this configuration.

#### Monitoring the bot

You can check for running instances with `docker-compose ps`.
This should list the service `freqtrade` as `running`. If that's not the case, best check the logs (see next point).

#### Docker-compose logs

Logs will be written to: `user_data/logs/freqtrade.log`.  
You can also check the latest log with the command `docker-compose logs -f`.

#### Database

The database will be located at: `user_data/tradesv3.sqlite`

#### Updating freqtrade with docker-compose

Updating freqtrade when using `docker-compose` is as simple as running the following 2 commands:

``` bash
# Download the latest image
docker-compose pull
# Restart the image
docker-compose up -d
```

This will first pull the latest image, and will then restart the container with the just pulled version.

!!! Warning "Check the Changelog"
    You should always check the changelog for breaking changes / manual interventions required and make sure the bot starts correctly after the update.

### Editing the docker-compose file

Advanced users may edit the docker-compose file further to include all possible options or arguments.

All freqtrade arguments will be available by running `docker-compose run --rm freqtrade <command> <optional arguments>`.

!!! Warning "`docker-compose` for trade commands"
    Trade commands (`freqtrade trade <...>`) should not be ran via `docker-compose run` - but should use `docker-compose up -d` instead.
    This makes sure that the container is properly started (including port forwardings) and will make sure that the container will restart after a system reboot.
    If you intend to use freqUI, please also ensure to adjust the [configuration accordingly](rest-api.md#configuration-with-docker), otherwise the UI will not be available.

!!! Note "`docker-compose run --rm`"
    Including `--rm` will remove the container after completion, and is highly recommended for all modes except trading mode (running with `freqtrade trade` command).

#### Example: Download data with docker-compose

Download backtesting data for 5 days for the pair ETH/BTC and 1h timeframe from Binance. The data will be stored in the directory `user_data/data/` on the host.

``` bash
docker-compose run --rm freqtrade download-data --pairs ETH/BTC --exchange binance --days 5 -t 1h
```

Head over to the [Data Downloading Documentation](data-download.md) for more details on downloading data.

#### Example: Backtest with docker-compose

Run backtesting in docker-containers for SampleStrategy and specified timerange of historical data, on 5m timeframe:

``` bash
docker-compose run --rm freqtrade backtesting --config user_data/config.json --strategy SampleStrategy --timerange 20190801-20191001 -i 5m
```

Head over to the [Backtesting Documentation](backtesting.md) to learn more.

### Additional dependencies with docker-compose

If your strategy requires dependencies not included in the default image - it will be necessary to build the image on your host.
For this, please create a Dockerfile containing installation steps for the additional dependencies (have a look at [docker/Dockerfile.custom](https://github.com/freqtrade/freqtrade/blob/develop/docker/Dockerfile.custom) for an example).

You'll then also need to modify the `docker-compose.yml` file and uncomment the build step, as well as rename the image to avoid naming collisions.

``` yaml
    image: freqtrade_custom
    build:
      context: .
      dockerfile: "./Dockerfile.<yourextension>"
```

You can then run `docker-compose build --pull` to build the docker image, and run it using the commands described above.

### Plotting with docker-compose

Commands `freqtrade plot-profit` and `freqtrade plot-dataframe` ([Documentation](plotting.md)) are available by changing the image to `*_plot` in your docker-compose.yml file.
You can then use these commands as follows:

``` bash
docker-compose run --rm freqtrade plot-dataframe --strategy AwesomeStrategy -p BTC/ETH --timerange=20180801-20180805
```

The output will be stored in the `user_data/plot` directory, and can be opened with any modern browser.

### Data analysis using docker compose

Freqtrade provides a docker-compose file which starts up a jupyter lab server.
You can run this server using the following command:

``` bash
docker-compose -f docker/docker-compose-jupyter.yml up
```

This will create a docker-container running jupyter lab, which will be accessible using `https://127.0.0.1:8888/lab`.
Please use the link that's printed in the console after startup for simplified login.

Since part of this image is built on your machine, it is recommended to rebuild the image from time to time to keep freqtrade (and dependencies) up-to-date.

``` bash
docker-compose -f docker/docker-compose-jupyter.yml build --no-cache
```

## Troubleshooting

### Docker on Windows

* Error: `"Timestamp for this request is outside of the recvWindow."`
  * The market api requests require a synchronized clock but the time in the docker container shifts a bit over time into the past.
    To fix this issue temporarily you need to run `wsl --shutdown` and restart docker again (a popup on windows 10 will ask you to do so).
    A permanent solution is either to host the docker container on a linux host or restart the wsl from time to time with the scheduler.

    ``` bash
    taskkill /IM "Docker Desktop.exe" /F
    wsl --shutdown
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    ```

!!! Warning
    Due to the above, we do not recommend the usage of docker on windows for production setups, but only for experimentation, datadownload and backtesting.
    Best use a linux-VPS for running freqtrade reliably.
