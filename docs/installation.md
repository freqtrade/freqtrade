# Installation

This page explains how to prepare your environment for running the bot.

To understand how to set up the bot please read the [Bot Configuration](https://github.com/freqtrade/freqtrade/blob/develop/docs/configuration.md) page.

## Table of Contents

* [Table of Contents](#table-of-contents)
* [Easy Installation - Linux Script](#easy-installation---linux-script)
* [Manual installation](#manual-installation)
* [Automatic Installation - Docker](#automatic-installation---docker)
* [Custom Linux MacOS Installation](#custom-installation)
	- [Requirements](#requirements)
	- [Linux - Ubuntu 16.04](#linux---ubuntu-1604)
	- [MacOS](#macos)
	- [Setup Config and virtual env](#setup-config-and-virtual-env)
* [Windows](#windows)

<!-- /TOC -->

------

## Easy Installation - Linux Script

If you are on Debian, Ubuntu or MacOS a freqtrade provides a script to Install, Update, Configure, and Reset your bot.

```bash
$ ./setup.sh
usage:
	-i,--install    Install freqtrade from scratch
	-u,--update     Command git pull to update.
	-r,--reset      Hard reset your develop/master branch.
	-c,--config     Easy config generator (Will override your existing file).
```

### --install

This script will install everything you need to run the bot:

* Mandatory software as: `Python3`, `ta-lib`, `wget`
* Setup your virtualenv
* Configure your `config.json` file

This script is a combination of `install script` `--reset`, `--config`

### --update

Update parameter will pull the last version of your current branch and update your virtualenv.

### --reset

Reset parameter will hard reset your branch (only if you are on `master` or `develop`) and recreate your virtualenv.

### --config

Config parameter is a `config.json` configurator. This script will ask you questions to setup your bot and create your `config.json`.


## Manual installation - Linux/MacOS
The following steps are made for Linux/MacOS environment

**1. Clone the repo**
```bash
git clone git@github.com:freqtrade/freqtrade.git
git checkout develop
cd freqtrade
```
**2. Create the config file**  
Switch `"dry_run": true,`
```bash
cp config.json.example config.json
vi config.json
```
**3. Build your docker image and run it**
```bash
docker build -t freqtrade .
docker run --rm -v /etc/localtime:/etc/localtime:ro -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

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
git clone https://github.com/freqtrade/freqtrade.git
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
cp -n config.json.example config.json
```

> To edit the config please refer to the [Bot Configuration](https://github.com/freqtrade/freqtrade/blob/develop/docs/configuration.md) page.

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
docker run --rm -v /etc/localtime:/etc/localtime:ro -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

There is known issue in OSX Docker versions after 17.09.1, whereby /etc/localtime cannot be shared causing Docker to not start. A work-around for this is to start with the following cmd.

```bash
docker run --rm -e TZ=`ls -la /etc/localtime | cut -d/ -f8-9` -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

More information on this docker issue and work-around can be read [here](https://github.com/docker/for-mac/issues/2396)

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
  -v /etc/localtime:/etc/localtime:ro \
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  freqtrade --db-url sqlite:///tradesv3.sqlite
```

NOTE: db-url defaults to `sqlite:///tradesv3.sqlite` but it defaults to `sqlite://` if `dry_run=True` is being used.
To override this behaviour use a custom db-url value: i.e.: `--db-url sqlite:///tradesv3.dryrun.sqlite`

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
sudo apt-get install python3.6 python3.6-venv python3.6-dev build-essential autoconf libtool pkg-config make wget git
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
git clone https://github.com/freqtrade/freqtrade.git
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
git clone https://github.com/freqtrade/freqtrade.git
```

Optionally checkout the develop branch:

```bash
git checkout develop
```

### Setup Config and virtual env

#### 1. Initialize the configuration

```bash
cd freqtrade
cp config.json.example config.json
```

> *To edit the config please refer to [Bot Configuration](https://github.com/freqtrade/freqtrade/blob/develop/docs/configuration.md).*

#### 2. Setup your Python virtual environment (virtualenv)

```bash
python3.6 -m venv .env
source .env/bin/activate
pip3.6 install --upgrade pip
pip3.6 install -r requirements.txt
pip3.6 install -e .
```

#### 3. Run the Bot

If this is the first time you run the bot, ensure you are running it in Dry-run `"dry_run": true,` otherwise it will start to buy and sell coins.

```bash
python3.6 ./freqtrade/main.py -c config.json
```

------

## Windows

We recommend that Windows users use [Docker](#docker) as this will work much easier and smoother (also more secure).

If that is not possible, try using the Windows Linux subsystem (WSL) - for which the Ubuntu instructions should work.
If that is not available on your system, feel free to try the instructions below, which led to success for some.

### Install freqtrade manually

#### Clone the git repository

```bash
git clone https://github.com/freqtrade/freqtrade.git
```

copy paste `config.json` to ``\path\freqtrade-develop\freqtrade`

#### install ta-lib

Install ta-lib according to the [ta-lib documentation](https://github.com/mrjbq7/ta-lib#windows).

As compiling from source on windows has heavy dependencies (requires a partial visual studio installation), there is also a repository of inofficial precompiled windows Wheels [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib), which needs to be downloaded and installed using `pip install TA_Lib‑0.4.17‑cp36‑cp36m‑win32.whl` (make sure to use the version matching your python version)

```cmd
>cd \path\freqtrade-develop
>python -m venv .env
>cd .env\Scripts
>activate.bat
>cd \path\freqtrade-develop
REM optionally install ta-lib from wheel
REM >pip install TA_Lib‑0.4.17‑cp36‑cp36m‑win32.whl
>pip install -r requirements.txt
>pip install -e .
>python freqtrade\main.py
```

> Thanks [Owdr](https://github.com/Owdr) for the commands. Source: [Issue #222](https://github.com/freqtrade/freqtrade/issues/222)

Now you have an environment ready, the next step is
[Bot Configuration](https://github.com/freqtrade/freqtrade/blob/develop/docs/configuration.md)...
