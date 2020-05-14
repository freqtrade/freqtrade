# Using Freqtrade with Docker

## Install Docker

Start by downloading and installing Docker CE for your platform:

* [Mac](https://docs.docker.com/docker-for-mac/install/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)
* [Linux](https://docs.docker.com/install/)

Optionally, [docker-compose](https://docs.docker.com/compose/install/) should be installed and available to follow the [docker quick start guide](#docker-quick-start).

Once you have Docker installed, simply prepare the config file (e.g. `config.json`) and run the image for `freqtrade` as explained below.

## Freqtrade with docker-compose

Freqtrade provides an official Docker image on [Dockerhub](https://hub.docker.com/r/freqtradeorg/freqtrade/), as well as a [docker-compose file](https://github.com/freqtrade/freqtrade/blob/develop/docker-compose.yml) ready for usage.

!!! Note
    The following section assumes that docker and docker-compose is installed and available to the logged in user.

!!! Note
    All below comands use relative directories and will have to be executed from the directory containing the `docker-compose.yml` file.

!!! Note "Docker on Raspberry"
    If you're running freqtrade on a Raspberry PI, you must change the image from `freqtradeorg/freqtrade:master` to `freqtradeorg/freqtrade:master_pi` or `freqtradeorg/freqtrade:develop_pi`, otherwise the image will not work.

### Docker quick start

Create a new directory and place the [docker-compose file](https://github.com/freqtrade/freqtrade/blob/develop/docker-compose.yml) in this directory.

``` bash
mkdir ft_userdata
cd ft_userdata/
# Download the docker-compose file from the repository
curl https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docker-compose.yml -o docker-compose.yml

# Pull the freqtrade image
docker-compose pull

# Create user directory structure
docker-compose run --rm freqtrade create-userdir --userdir user_data

# Create configuration - Requires answering interactive questions
docker-compose run --rm freqtrade new-config --config user_data/config.json
```

The above snippet creates a new directory called "ft_userdata", downloads the latest compose file and pulls the freqtrade image.
The last 2 steps in the snippet create the directory with user-data, as well as (interactively) the default configuration based on your selections.

!!! Note
    You can edit the configuration at any time, which is available as `user_data/config.json` (within the directory `ft_userdata`) when using the above configuration.

#### Adding your strategy

The configuration is now available as `user_data/config.json`.
You should now copy your strategy to `user_data/strategies/` - and add the Strategy class name to the `docker-compose.yml` file, replacing `SampleStrategy`. If you wish to run the bot with the SampleStrategy, just leave it as it is.

!!! Warning
    The `SampleStrategy` is there for your reference and give you ideas for your own strategy.
    Please always backtest the strategy and use dry-run for some time before risking real money!

Once this is done, you're ready to launch the bot in trading mode (Dry-run or Live-trading, depending on your answer to the corresponding question you made above).

``` bash
docker-compose up -d
```

#### Docker-compose logs

Logs will be written to `user_data/logs/freqtrade.log`. 
Alternatively, you can check the latest logs using `docker-compose logs -f`.

#### Database

The database will be in the user_data directory as well, and will be called `user_data/tradesv3.sqlite`.

#### Updating freqtrade with docker-compose

To update freqtrade when using docker-compose is as simple as running the following 2 commands:

``` bash
# Download the latest image
docker-compose pull
# Restart the image
docker-compose up -d
```

This will first pull the latest image, and will then restart the container with the just pulled version.

!!! Note
    You should always check the changelog for breaking changes / manual interventions required and make sure the bot starts correctly after the update.

#### Going from here

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

## Freqtrade with docker without docker-compose

!!! Warning
    The below documentation is provided for completeness and assumes that you are somewhat familiar with running docker containers. If you're just starting out with docker, we recommend to follow the [Freqtrade with docker-compose](#freqtrade-with-docker-compose) instructions.

### Download the official Freqtrade docker image

Pull the image from docker hub.

Branches / tags available can be checked out on [Dockerhub tags page](https://hub.docker.com/r/freqtradeorg/freqtrade/tags/).

```bash
docker pull freqtradeorg/freqtrade:develop
# Optionally tag the repository so the run-commands remain shorter
docker tag freqtradeorg/freqtrade:develop freqtrade
```

To update the image, simply run the above commands again and restart your running container.

Should you require additional libraries, please [build the image yourself](#build-your-own-docker-image).

!!! Note "Docker image update frequency"
    The official docker images with tags `master`, `develop` and `latest` are automatically rebuild once a week to keep the base image uptodate.
    In addition to that, every merge to `develop` will trigger a rebuild for `develop` and `latest`.

### Prepare the configuration files

Even though you will use docker, you'll still need some files from the github repository.

#### Clone the git repository

Linux/Mac/Windows with WSL

```bash
git clone https://github.com/freqtrade/freqtrade.git
```

Windows with docker

```bash
git clone --config core.autocrlf=input https://github.com/freqtrade/freqtrade.git
```

#### Copy `config.json.example` to `config.json`

```bash
cd freqtrade
cp -n config.json.example config.json
```

> To understand the configuration options, please refer to the [Bot Configuration](configuration.md) page.

#### Create your database file

Production

```bash
touch tradesv3.sqlite
````

Dry-Run

```bash
touch tradesv3.dryrun.sqlite
```

!!! Note
    Make sure to use the path to this file when starting the bot in docker.

### Build your own Docker image

Best start by pulling the official docker image from dockerhub as explained [here](#download-the-official-docker-image) to speed up building.

To add additional libraries to your docker image, best check out [Dockerfile.technical](https://github.com/freqtrade/freqtrade/blob/develop/Dockerfile.technical) which adds the [technical](https://github.com/freqtrade/technical) module to the image.

```bash
docker build -t freqtrade -f Dockerfile.technical .
```

If you are developing using Docker, use `Dockerfile.develop` to build a dev Docker image, which will also set up develop dependencies:

```bash
docker build -f Dockerfile.develop -t freqtrade-dev .
```

!!! Note
    For security reasons, your configuration file will not be included in the image, you will need to bind mount it. It is also advised to bind mount an SQLite database file (see the "5. Run a restartable docker image" section) to keep it between  updates.

#### Verify the Docker image

After the build process you can verify that the image was created with:

```bash
docker images
```

The output should contain the freqtrade image.

### Run the Docker image

You can run a one-off container that is immediately deleted upon exiting with the following command (`config.json` must be in the current working directory):

```bash
docker run --rm -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

!!! Warning
    In this example, the database will be created inside the docker instance and will be lost when you will refresh your image.

#### Adjust timezone

By default, the container will use UTC timezone.
Should you find this irritating please add the following to your docker commands:

##### Linux

``` bash
-v /etc/timezone:/etc/timezone:ro

# Complete command:
docker run --rm -v /etc/timezone:/etc/timezone:ro -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

##### MacOS

There is known issue in OSX Docker versions after 17.09.1, whereby `/etc/localtime` cannot be shared causing Docker to not start. A work-around for this is to start with the following cmd.

```bash
docker run --rm -e TZ=`ls -la /etc/localtime | cut -d/ -f8-9` -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

More information on this docker issue and work-around can be read [here](https://github.com/docker/for-mac/issues/2396).

### Run a restartable docker image

To run a restartable instance in the background (feel free to place your configuration and database files wherever it feels comfortable on your filesystem).

#### Move your config file and database

The following will assume that you place your configuration / database files to `~/.freqtrade`, which is a hidden directory in your home directory. Feel free to use a different directory and replace the directory in the upcomming commands.

```bash
mkdir ~/.freqtrade
mv config.json ~/.freqtrade
mv tradesv3.sqlite ~/.freqtrade
```

#### Run the docker image

```bash
docker run -d \
  --name freqtrade \
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/user_data/:/freqtrade/user_data \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  freqtrade trade --db-url sqlite:///tradesv3.sqlite --strategy MyAwesomeStrategy
```

!!! Note
    When using docker, it's best to specify `--db-url` explicitly to ensure that the database URL and the mounted database file match.

!!! Note
    All available bot command line parameters can be added to the end of the `docker run` command.

!!! Note
    You can define a [restart policy](https://docs.docker.com/config/containers/start-containers-automatically/) in docker. It can be useful in some cases to use the `--restart unless-stopped` flag (crash of freqtrade or reboot of your system).

### Monitor your Docker instance

You can use the following commands to monitor and manage your container:

```bash
docker logs freqtrade
docker logs -f freqtrade
docker restart freqtrade
docker stop freqtrade
docker start freqtrade
```

For more information on how to operate Docker, please refer to the [official Docker documentation](https://docs.docker.com/).

!!! Note
    You do not need to rebuild the image for configuration changes, it will suffice to edit `config.json` and restart the container.

### Backtest with docker

The following assumes that the download/setup of the docker image have been completed successfully.
Also, backtest-data should be available at `~/.freqtrade/user_data/`.

```bash
docker run -d \
  --name freqtrade \
  -v /etc/localtime:/etc/localtime:ro \
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  -v ~/.freqtrade/user_data/:/freqtrade/user_data/ \
  freqtrade backtesting --strategy AwsomelyProfitableStrategy
```

Head over to the [Backtesting Documentation](backtesting.md) for more details.

!!! Note
    Additional bot command line parameters can be appended after the image name (`freqtrade` in the above example).
