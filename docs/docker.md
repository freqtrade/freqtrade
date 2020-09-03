# Using Freqtrade with Docker

## Install Docker

Start by downloading and installing Docker CE for your platform:

* [Mac](https://docs.docker.com/docker-for-mac/install/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)
* [Linux](https://docs.docker.com/install/)

Optionally, [`docker-compose`](https://docs.docker.com/compose/install/) should be installed and available to follow the [docker quick start guide](#docker-quick-start).

Once you have Docker installed, simply prepare the config file (e.g. `config.json`) and run the image for `freqtrade` as explained below.

## Freqtrade with docker-compose

Freqtrade provides an official Docker image on [Dockerhub](https://hub.docker.com/r/freqtradeorg/freqtrade/), as well as a [docker-compose file](https://github.com/freqtrade/freqtrade/blob/develop/docker-compose.yml) ready for usage.

!!! Note
    - The following section assumes that `docker` and `docker-compose` are installed and available to the logged in user.
    - All below comands use relative directories and will have to be executed from the directory containing the `docker-compose.yml` file.
    

### Docker quick start

Create a new directory and place the [docker-compose file](https://github.com/freqtrade/freqtrade/blob/develop/docker-compose.yml) in this directory.

=== "PC/MAC/Linux"
    ``` bash
    mkdir ft_userdata
    cd ft_userdata/
    # Download the docker-compose file from the repository
    curl https://raw.githubusercontent.com/freqtrade/freqtrade/master/docker-compose.yml -o docker-compose.yml

    # Pull the freqtrade image
    docker-compose pull

    # Create user directory structure
    docker-compose run --rm freqtrade create-userdir --userdir user_data

    # Create configuration - Requires answering interactive questions
    docker-compose run --rm freqtrade new-config --config user_data/config.json
    ```

=== "RaspberryPi"
    ``` bash
    mkdir ft_userdata
    cd ft_userdata/
    # Download the docker-compose file from the repository
    curl https://raw.githubusercontent.com/freqtrade/freqtrade/master/docker-compose.yml -o docker-compose.yml

    # Pull the freqtrade image
    docker-compose pull

    # Create user directory structure
    docker-compose run --rm freqtrade create-userdir --userdir user_data

    # Create configuration - Requires answering interactive questions
    docker-compose run --rm freqtrade new-config --config user_data/config.json
    ```

    !!! Note "Change your docker Image"
        You should change the docker image in your config file for your Raspeberry build to work properly.
        ``` bash
        image: freqtradeorg/freqtrade:master_pi
        # image: freqtradeorg/freqtrade:develop_pi
        ```

The above snippet creates a new directory called `ft_userdata`, downloads the latest compose file and pulls the freqtrade image.
The last 2 steps in the snippet create the directory with `user_data`, as well as (interactively) the default configuration based on your selections.

!!! Question "How to edit the bot configuration?"
    You can edit the configuration at any time, which is available as `user_data/config.json` (within the directory `ft_userdata`) when using the above configuration.

#### Adding a custom strategy

1. The configuration is now available as `user_data/config.json`
2. Copy a custom strategy to the directory `user_data/strategies/`
3. add the Strategy' class name to the `docker-compose.yml` file

The `SampleStrategy` is run by default.

!!! Warning "`SampleStrategy` is just a demo!"
    The `SampleStrategy` is there for your reference and give you ideas for your own strategy.
    Please always backtest the strategy and use dry-run for some time before risking real money!

Once this is done, you're ready to launch the bot in trading mode (Dry-run or Live-trading, depending on your answer to the corresponding question you made above).

=== "Docker Compose"
    ``` bash
    docker-compose up -d
    ```

#### Docker-compose logs

Logs will be located at: `user_data/logs/freqtrade.log`. 
You can check the latest log with the command `docker-compose logs -f`.

#### Database

The database will be at: `user_data/tradesv3.sqlite`

#### Updating freqtrade with docker-compose

To update freqtrade when using `docker-compose` is as simple as running the following 2 commands:


``` bash
# Download the latest image
docker-compose pull
# Restart the image
docker-compose up -d
```

This will first pull the latest image, and will then restart the container with the just pulled version.

!!! Warning "Check the Changelog"
    You should always check the changelog for breaking changes / manual interventions required and make sure the bot starts correctly after the update.

