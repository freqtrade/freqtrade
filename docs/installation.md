# Installation
This page explains how to prepare your environment for running the bot.

## Prerequisite
Before running your bot in production you will need to setup few
external API. In production mode, the bot required valid Bittrex API
credentials and a Telegram bot (optional but recommended).

- [Setup your exchange account](#setup-your-exchange-account)
- [Backtesting commands](#setup-your-telegram-bot)

### Setup your exchange account
*To be completed, please feel free to complete this section.*

### Setup your Telegram bot
The only things you need is a working Telegram bot and its API token.
Below we explain how to create your Telegram Bot, and how to get your
Telegram user id.

### 1. Create your Telegram bot

**1.1. Start a chat with https://telegram.me/BotFather**

**1.2. Send the message `/newbot`. ** *BotFather response:*
```
Alright, a new bot. How are we going to call it? Please choose a name for your bot.
```

**1.3. Choose the public name of your bot (e.x. `Freqtrade bot`)**
*BotFather response:*
```
Good. Now let's choose a username for your bot. It must end in `bot`. Like this, for example: TetrisBot or tetris_bot.
```
**1.4. Choose the name id of your bot (e.x "`My_own_freqtrade_bot`")**

**1.5. Father bot will return you the token (API key)**<br/>
Copy it and keep it you will use it for the config parameter `token`.
*BotFather response:*
```hl_lines="4"
Done! Congratulations on your new bot. You will find it at t.me/My_own_freqtrade_bot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.

Use this token to access the HTTP API:
521095879:AAEcEZEL7ADJ56FtG_qD0bQJSKETbXCBCi0

For a description of the Bot API, see this page: https://core.telegram.org/bots/api
```
**1.6. Don't forget to start the conversation with your bot, by clicking /START button**

### 2. Get your user id
**2.1. Talk to https://telegram.me/userinfobot**

**2.2. Get your "Id", you will use it for the config parameter
`chat_id`.**
<hr/>
## Quick start
Freqtrade provides a Linux/MacOS script to install all dependencies and help you to configure the bot.

```bash
git clone git@github.com:freqtrade/freqtrade.git
cd freqtrade
git checkout develop
./setup.sh --install
```
!!! Note
    Windows installation is explained [here](#windows).
<hr/>
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

** --install **

This script will install everything you need to run the bot:

* Mandatory software as: `Python3`, `ta-lib`, `wget`
* Setup your virtualenv
* Configure your `config.json` file

This script is a combination of `install script` `--reset`, `--config`

** --update **

Update parameter will pull the last version of your current branch and update your virtualenv.

** --reset **

Reset parameter will hard reset your branch (only if you are on `master` or `develop`) and recreate your virtualenv.

** --config **

Config parameter is a `config.json` configurator. This script will ask you questions to setup your bot and create your `config.json`.

------

## Automatic Installation - Docker

Start by downloading Docker for your platform:

* [Mac](https://www.docker.com/products/docker#/mac)
* [Windows](https://www.docker.com/products/docker#/windows)
* [Linux](https://www.docker.com/products/docker#/linux)

Once you have Docker installed, simply create the config file (e.g. `config.json`) and then create a Docker image for `freqtrade` using the Dockerfile in this repo.

### 1. Prepare the Bot

**1.1. Clone the git repository**

Linux/Mac/Windows with WSL
```bash
git clone https://github.com/freqtrade/freqtrade.git
```

Windows with docker
```bash
git clone --config core.autocrlf=input https://github.com/freqtrade/freqtrade.git
```

**1.2. (Optional) Checkout the develop branch**

```bash
git checkout develop
```

**1.3. Go into the new directory**

```bash
cd freqtrade
```

**1.4. Copy `config.json.example` to `config.json`**

```bash
cp -n config.json.example config.json
```

> To edit the config please refer to the [Bot Configuration](configuration.md) page.

**1.5. Create your database file *(optional - the bot will create it if it is missing)**

Production

```bash
touch tradesv3.sqlite
````

Dry-Run

```bash
touch tradesv3.dryrun.sqlite
```

### 2. Download or build the docker image

Either use the prebuilt image from docker hub - or build the image yourself if you would like more control on which version is used.

Branches / tags available can be checked out on [Dockerhub](https://hub.docker.com/r/freqtradeorg/freqtrade/tags/).

**2.1. Download the docker image**

Pull the image from docker hub and (optionally) change the name of the image

```bash
docker pull freqtradeorg/freqtrade:develop
# Optionally tag the repository so the run-commands remain shorter
docker tag freqtradeorg/freqtrade:develop freqtrade
```

To update the image, simply run the above commands again and restart your running container.

**2.2. Build the Docker image**

```bash
cd freqtrade
docker build -t freqtrade .
```

If you are developing using Docker, use `Dockerfile.develop` to build a dev Docker image, which will also set up develop dependencies:

```bash
docker build -f ./Dockerfile.develop -t freqtrade-dev .
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

More information on this docker issue and work-around can be read [here](https://github.com/docker/for-mac/issues/2396).

In this example, the database will be created inside the docker instance and will be lost when you will refresh your image.

### 5. Run a restartable docker image

To run a restartable instance in the background (feel free to place your configuration and database files wherever it feels comfortable on your filesystem).

**5.1. Move your config file and database**

```bash
mkdir ~/.freqtrade
mv config.json ~/.freqtrade
mv tradesv3.sqlite ~/.freqtrade
```

**5.2. Run the docker image**

```bash
docker run -d \
  --name freqtrade \
  -v /etc/localtime:/etc/localtime:ro \
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  freqtrade --db-url sqlite:///tradesv3.sqlite
```

!!! Note
    db-url defaults to `sqlite:///tradesv3.sqlite` but it defaults to `sqlite://` if `dry_run=True` is being used.
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

For more information on how to operate Docker, please refer to the [official Docker documentation](https://docs.docker.com/).

!!! Note
    You do not need to rebuild the image for configuration changes, it will suffice to edit `config.json` and restart the container.

### 7. Backtest with docker

The following assumes that the above steps (1-4) have been completed successfully.
Also, backtest-data should be available at `~/.freqtrade/user_data/`.

```bash
docker run -d \
  --name freqtrade \
  -v /etc/localtime:/etc/localtime:ro \
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  -v ~/.freqtrade/user_data/:/freqtrade/user_data/ \
  freqtrade --strategy AwsomelyProfitableStrategy backtesting
```

Head over to the [Backtesting Documentation](backtesting.md) for more details.

!!! Note
    Additional parameters can be appended after the image name (`freqtrade` in the above example).

------

## Custom Installation

We've included/collected install instructions for Ubuntu 16.04, MacOS, and Windows. These are guidelines and your success may vary with other distros.
OS Specific steps are listed first, the [Common](#common) section below is necessary for all systems.

### Requirements

Click each one for install guide:

* [Python >= 3.6.x](http://docs.python-guide.org/en/latest/starting/installation/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (Recommended)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html)

### Linux - Ubuntu 16.04

#### Install Python 3.6, Git, and wget

```bash
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6 python3.6-venv python3.6-dev build-essential autoconf libtool pkg-config make wget git
```

#### Raspberry Pi / Raspbian

Before installing FreqTrade on a Raspberry Pi running the official Raspbian Image, make sure you have at least Python 3.6 installed. The default image only provides Python 3.5. Probably the easiest way to get a recent version of python is [miniconda](https://repo.continuum.io/miniconda/).

The following assumes that miniconda3 is installed and available in your environment. Last miniconda3 installation file use python 3.4, we will update to python 3.6 on this installation.
It's recommended to use (mini)conda for this as installation/compilation of `numpy`, `scipy` and `pandas` takes a long time.
If you have installed it from (mini)conda, you can remove `numpy`, `scipy`, and `pandas` from `requirements.txt` before you install it with `pip`.

Additional package to install on your Raspbian, `libffi-dev` required by cryptography (from python-telegram-bot).

``` bash
conda config --add channels rpi
conda install python=3.6
conda create -n freqtrade python=3.6
conda activate freqtrade
conda install scipy pandas numpy

sudo apt install libffi-dev
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### MacOS

#### Install Python 3.6, git and wget

```bash
brew install python3 git wget
```

### Common

#### 1. Install TA-Lib

Official webpage: https://mrjbq7.github.io/ta-lib/install.html

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
sed -i.bak "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h
./configure --prefix=/usr/local
make
sudo make install
cd ..
rm -rf ./ta-lib*
```

!!! Note
    An already downloaded version of ta-lib is included in the repository, as the sourceforge.net source seems to have problems frequently.

#### 2. Setup your Python virtual environment (virtualenv)

!!! Note
    This step is optional but strongly recommended to keep your system organized

```bash
python3 -m venv .env
source .env/bin/activate
```

#### 3. Install FreqTrade

Clone the git repository:

```bash
git clone https://github.com/freqtrade/freqtrade.git

```

Optionally checkout the stable/master branch:

```bash
git checkout master
```

#### 4. Initialize the configuration

```bash
cd freqtrade
cp config.json.example config.json
```

> *To edit the config please refer to [Bot Configuration](configuration.md).*

#### 5. Install python dependencies

``` bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install -e .
```

#### 6. Run the Bot

If this is the first time you run the bot, ensure you are running it in Dry-run `"dry_run": true,` otherwise it will start to buy and sell coins.

```bash
python3.6 ./freqtrade/main.py -c config.json
```

*Note*: If you run the bot on a server, you should consider using [Docker](#automatic-installation---docker) a terminal multiplexer like `screen` or [`tmux`](https://en.wikipedia.org/wiki/Tmux) to avoid that the bot is stopped on logout.

#### 7. [Optional] Configure `freqtrade` as a `systemd` service

From the freqtrade repo... copy `freqtrade.service` to your systemd user directory (usually `~/.config/systemd/user`) and update `WorkingDirectory` and `ExecStart` to match your setup.

After that you can start the daemon with:

```bash
systemctl --user start freqtrade
```

For this to be persistent (run when user is logged out) you'll need to enable `linger` for your freqtrade user.

```bash
sudo loginctl enable-linger "$USER"
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

#### Install ta-lib

Install ta-lib according to the [ta-lib documentation](https://github.com/mrjbq7/ta-lib#windows).

As compiling from source on windows has heavy dependencies (requires a partial visual studio installation), there is also a repository of unofficial precompiled windows Wheels [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib), which needs to be downloaded and installed using `pip install TA_Lib‑0.4.17‑cp36‑cp36m‑win32.whl` (make sure to use the version matching your python version)

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

#### Error during installation under Windows

``` bash
error: Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools": http://landinghub.visualstudio.com/visual-cpp-build-tools
```

Unfortunately, many packages requiring compilation don't provide a pre-build wheel. It is therefore mandatory to have a C/C++ compiler installed and available for your python environment to use.

The easiest way is to download install Microsoft Visual Studio Community [here](https://visualstudio.microsoft.com/downloads/) and make sure to install "Common Tools for Visual C++" to enable building c code on Windows. Unfortunately, this is a heavy download / dependency (~4Gb) so you might want to consider WSL or docker first.

---

Now you have an environment ready, the next step is
[Bot Configuration](configuration.md).
