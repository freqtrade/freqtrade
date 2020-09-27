## Freqtrade with docker without docker-compose

!!! Warning
    The below documentation is provided for completeness and assumes that you are familiar with running docker containers. If you're just starting out with Docker, we recommend to follow the [Quickstart](docker.md) instructions.

### Download the official Freqtrade docker image

Pull the image from docker hub.

Branches / tags available can be checked out on [Dockerhub tags page](https://hub.docker.com/r/freqtradeorg/freqtrade/tags/).

```bash
docker pull freqtradeorg/freqtrade:stable
# Optionally tag the repository so the run-commands remain shorter
docker tag freqtradeorg/freqtrade:stable freqtrade
```

To update the image, simply run the above commands again and restart your running container.

Should you require additional libraries, please [build the image yourself](#build-your-own-docker-image).

!!! Note "Docker image update frequency"
    The official docker images with tags `stable`, `develop` and `latest` are automatically rebuild once a week to keep the base image up-to-date.
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

=== "Dry-Run"
    ``` bash
    touch tradesv3.dryrun.sqlite
    ```

=== "Production"
    ``` bash
    touch tradesv3.sqlite
    ```


!!! Warning "Database File Path"
    Make sure to use the path to the correct database file when starting the bot in Docker.

### Build your own Docker image

Best start by pulling the official docker image from dockerhub as explained [here](#download-the-official-docker-image) to speed up building.

To add additional libraries to your docker image, best check out [Dockerfile.technical](https://github.com/freqtrade/freqtrade/blob/develop/docker/Dockerfile.technical) which adds the [technical](https://github.com/freqtrade/technical) module to the image.

```bash
docker build -t freqtrade -f docker/Dockerfile.technical .
```

If you are developing using Docker, use `docker/Dockerfile.develop` to build a dev Docker image, which will also set up develop dependencies:

```bash
docker build -f docker/Dockerfile.develop -t freqtrade-dev .
```

!!! Warning "Include your config file manually"
    For security reasons, your configuration file will not be included in the image, you will need to bind mount it. It is also advised to bind mount an SQLite database file (see [5. Run a restartable docker image](#run-a-restartable-docker-image)") to keep it between  updates.

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
    In this example, the database will be created inside the docker instance and will be lost when you refresh your image.

#### Adjust timezone

By default, the container will use UTC timezone.
If you would like to change the timezone use the following commands:

=== "Linux"
    ``` bash
    -v /etc/timezone:/etc/timezone:ro

    # Complete command:
    docker run --rm -v /etc/timezone:/etc/timezone:ro -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
    ```

=== "MacOS"
    ```bash
    docker run --rm -e TZ=`ls -la /etc/localtime | cut -d/ -f8-9` -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
    ```

!!! Note "MacOS Issues"
    The OSX Docker versions after 17.09.1 have a known issue whereby `/etc/localtime` cannot be shared causing Docker to not start.<br>
    A work-around for this is to start with the MacOS command above
    More information on this docker issue and work-around can be read [here](https://github.com/docker/for-mac/issues/2396).

### Run a restartable docker image

To run a restartable instance in the background (feel free to place your configuration and database files wherever it feels comfortable on your filesystem).

#### 1. Move your config file and database

The following will assume that you place your configuration / database files to `~/.freqtrade`, which is a hidden directory in your home directory. Feel free to use a different directory and replace the directory in the upcomming commands.

```bash
mkdir ~/.freqtrade
mv config.json ~/.freqtrade
mv tradesv3.sqlite ~/.freqtrade
```

#### 2. Run the docker image

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
