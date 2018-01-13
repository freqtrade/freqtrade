# Installation

This page explains how to prepare your environment for running the bot.

To understand how to set up the bot please read the [Bot Configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md) page.



## Table of Contents

* [Table of Contents](#table-of-contents)
* [Automatic Installation - Docker](#automatic-installation-docker)
* [Custom Installation](#custom-installation)
	- [Requirements](#requirements)
	- [Linux - Ubuntu 16.04](#linux-ubuntu-1604)
	- [MacOS](#macos)
	- [Windows](#windows)
* [First Steps](#first-step)

<!-- /TOC -->
------

## Automatic Installation - Docker

Start by downloading Docker for your platform:

* [Mac](https://www.docker.com/products/docker#/mac)
* [Windows](https://www.docker.com/products/docker#/windows)
* [Linux](https://www.docker.com/products/docker#/linux)

Once you have Docker installed, simply create the config file (e.g. `config.json`) and then create a Docker image for `freqtrade` using the Dockerfile in this repo.


### 1. Prepare the Bot

#### 1.1. Clone the git repository

```bash
git clone https://github.com/gcarq/freqtrade.git
```

#### 1.2. (Optional) Checkout the develop branch

```bash
git checkout develop
```

#### 1.3. Go into the new directory

```bash
cd freqtrade
```

#### 1.4. Copy `config.json.example` to `config.json`

```bash
cat config.json.example >> config.json
```

> To edit the config please refer to the [Bot Configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md) page.

#### 1.5. Create your database file *(optional - the bot will create it if it is missing)*


Production
```bash
touch tradesv3.sqlite
````

Dry-Run
```bash
touch tradesv3.dryrun.sqlite
```


### 2. Build the Docker image

```bash
cd freqtrade
docker build -t freqtrade .
```

For security reasons, your configuration file will not be included in the image, you will need to bind mount it. It is also advised to bind mount an SQLite database file (see the "5. Run a restartable docker image" section) to keep it between  updates.


### 3. Verify the Docker image

After the build process you can verify that the image was created with:

```bash
docker images
```


### 4. Run the Docker image

You can run a one-off container that is immediately deleted upon exiting with the following command (`config.json` must be in the current working directory):

```bash
docker run --rm -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

In this example, the database will be created inside the docker instance and will be lost when you will refresh your image.


### 5. Run a restartable docker image

To run a restartable instance in the background (feel free to place your configuration and database files wherever it feels comfortable on your filesystem).

#### 5.1. Move your config file and database

```bash
mkdir ~/.freqtrade
mv config.json ~/.freqtrade
mv tradesv3.sqlite ~/.freqtrade
```

#### 5.2. Run the docker image

```bash
docker run -d \
  --name freqtrade \
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  freqtrade
```

If you are using `dry_run=True` it's not necessary to mount `tradesv3.sqlite`, but you can mount `tradesv3.dryrun.sqlite` if you plan to use the dry run mode with the param `--dry-run-db`.


### 6. Monitor your Docker instance

You can then use the following commands to monitor and manage your container:

```bash
docker logs freqtrade
docker logs -f freqtrade
docker restart freqtrade
docker stop freqtrade
docker start freqtrade
```

You do not need to rebuild the image for configuration changes, it will suffice to edit `config.json` and restart the container.


------


## Custom Installation

We've included/collected install instructions for Ubuntu 16.04, MacOS, and Windows. These are guidelines and your success may vary with other distros.

### Requirements

Click each one for install guide:

* [Python 3.6.x](http://docs.python-guide.org/en/latest/starting/installation/), note the bot was not tested on Python >= 3.7.x
* [pip](https://pip.pypa.io/en/stable/installing/)
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (Recommended)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html)


### Linux - Ubuntu 16.04

#### 1. Install Python 3.6, Git, and wget

```bash
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6 python3.6-venv build-essential autoconf libtool pkg-config make wget git
```

#### 2. Install TA-Lib

Official webpage: https://mrjbq7.github.io/ta-lib/install.html

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install
cd ..
rm -rf ./ta-lib*
```

#### 3. [Optional] Install MongoDB

Install MongoDB if you plan to optimize your strategy with Hyperopt.

```bash
sudo apt-get install mongodb-org
```

> Complete tutorial from Digital Ocean: [How to Install MongoDB on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-mongodb-on-ubuntu-16-04).

#### 4. Install FreqTrade

Clone the git repository:

```bash
git clone https://github.com/gcarq/freqtrade.git
```

Optionally checkout the develop branch:

```bash
git checkout develop
```

#### 5. Configure `freqtrade` as a `systemd` service

From the freqtrade repo... copy `freqtrade.service` to your systemd user directory (usually `~/.config/systemd/user`) and update `WorkingDirectory` and `ExecStart` to match your setup.

After that you can start the daemon with:
```bash
systemctl --user start freqtrade
```

For this to be persistent (run when user is logged out) you'll need to enable `linger` for your freqtrade user.

```bash
sudo loginctl enable-linger "$USER"
```


### MacOS

#### 1. Install Python 3.6, git, wget and ta-lib

```bash
brew install python3 git wget ta-lib
```

#### 2. [Optional] Install MongoDB

Install MongoDB if you plan to optimize your strategy with Hyperopt.

```bash
curl -O https://fastdl.mongodb.org/osx/mongodb-osx-ssl-x86_64-3.4.10.tgz
tar -zxvf mongodb-osx-ssl-x86_64-3.4.10.tgz
mkdir -p <path_freqtrade>/env/mongodb
cp -R -n mongodb-osx-x86_64-3.4.10/ <path_freqtrade>/env/mongodb
export PATH=<path_freqtrade>/env/mongodb/bin:$PATH
```

#### 3. Install FreqTrade

Clone the git repository:

```bash
git clone https://github.com/gcarq/freqtrade.git
```

Optionally checkout the develop branch:

```bash
git checkout develop
```


### Windows

We recommend that Windows users use [Docker](#docker) as this will work
much easier and smoother (also more secure).

#### 1. Install freqtrade

copy paste `config.json` to ``\path\freqtrade-develop\freqtrade`

```cmd
>cd \path\freqtrade-develop
>python -m venv .env
>cd .env\Scripts
>activate.bat
>cd \path\freqtrade-develop
>pip install -r requirements.txt
>pip install -e .
>cd freqtrade
>python main.py
```

> Thanks [Owdr](https://github.com/Owdr) for the commands. Source: [Issue #222](https://github.com/gcarq/freqtrade/issues/222)


------


## First Steps

### 1. Initialize the configuration

```bash
cd freqtrade
cp config.json.example config.json
```

> *To edit the config please refer to [Bot Configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md).*


### 2. Setup your Python virtual environment (virtualenv)

```bash
python3.6 -m venv .env
source .env/bin/activate
pip3.6 install -r requirements.txt
pip3.6 install -e .
```

### 3. Run the Bot

If this is the first time you run the bot, ensure you are running it in Dry-run `"dry_run": true,` otherwise it will start to buy and sell coins.

```bash
python3.6 ./freqtrade/main.py -c config.json
```

Now you have an environment ready, the next step is
[Bot Configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md)...
