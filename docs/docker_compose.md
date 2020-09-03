#### Editing the docker-compose file

Advanced users may edit the docker-compose file further to include all possible options or arguments.

All possible freqtrade arguments will be available by running `docker-compose run --rm freqtrade <command> <optional arguments>`.

!!! Note "`docker-compose run --rm`"
    Including `--rm` will clean up the container after completion, and is highly recommended for all modes except trading mode (running with `freqtrade trade` command).

##### Example: Download data with docker-compose

Download backtesting data for 5 days for the pair ETH/BTC and 1h timeframe from Binance. The data will be stored in the directory `user_data/data/` on the host.

``` bash
docker-compose run --rm freqtrade download-data --pairs ETH/BTC --exchange binance --days 5 -t 1h
```

Head over to the [Data Downloading Documentation](data-download.md) for more details on downloading data.

##### Example: Backtest with docker-compose

Run backtesting in docker-containers for SampleStrategy and specified timerange of historical data, on 5m timeframe:

``` bash
docker-compose run --rm freqtrade backtesting --config user_data/config.json --strategy SampleStrategy --timerange 20190801-20191001 -i 5m
```

Head over to the [Backtesting Documentation](backtesting.md) to learn more.

#### Additional dependencies with docker-compose

If your strategy requires dependencies not included in the default image (like [technical](https://github.com/freqtrade/technical)) - it will be necessary to build the image on your host.
For this, please create a Dockerfile containing installation steps for the additional dependencies (have a look at [Dockerfile.technical](https://github.com/freqtrade/freqtrade/blob/develop/Dockerfile.technical) for an example).

You'll then also need to modify the `docker-compose.yml` file and uncomment the build step, as well as rename the image to avoid naming collisions.

``` yaml
    image: freqtrade_custom
    build:
      context: .
      dockerfile: "./Dockerfile.<yourextension>"
```

You can then run `docker-compose build` to build the docker image, and run it using the commands described above.