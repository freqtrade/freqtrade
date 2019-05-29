# Installation

This page explains how to prepare your environment for running the bot.

## Prerequisite

Before running your bot in production you will need to setup few
external API. In production mode, the bot will require valid Exchange API
credentials. We also reccomend a [Telegram bot](telegram-usage.md#setup-your-telegram-bot) (optional but recommended).

- [Setup your exchange account](#setup-your-exchange-account)

### Setup your exchange account

You will need to create API Keys (Usually you get `key` and `secret`) from the Exchange website and insert this into the appropriate fields in the configuration or when asked by the installation script.

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

Additional package to install on your Raspbian, `libffi-dev` required by cryptography (from python-telegram-bot).

``` bash
conda config --add channels rpi
conda install python=3.6
conda create -n freqtrade python=3.6
conda activate freqtrade
conda install scipy pandas numpy

sudo apt install libffi-dev
python3 -m pip install -r requirements-common.txt
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
python3.6 freqtrade -c config.json
```

*Note*: If you run the bot on a server, you should consider using [Docker](docker.md) or a terminal multiplexer like `screen` or [`tmux`](https://en.wikipedia.org/wiki/Tmux) to avoid that the bot is stopped on logout.

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

If you run the bot as a service, you can use systemd service manager as a software watchdog monitoring freqtrade bot 
state and restarting it in the case of failures. If the `internals.sd_notify` parameter is set to true in the 
configuration or the `--sd-notify` command line option is used, the bot will send keep-alive ping messages to systemd 
using the sd_notify (systemd notifications) protocol and will also tell systemd its current state (Running or Stopped) 
when it changes. 

The `freqtrade.service.watchdog` file contains an example of the service unit configuration file which uses systemd 
as the watchdog.

!!! Note 
  The sd_notify communication between the bot and the systemd service manager will not work if the bot runs in a Docker container.

------

## Windows

We recommend that Windows users use [Docker](docker.md) as this will work much easier and smoother (also more secure).

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

The easiest way is to download install Microsoft Visual Studio Community [here](https://visualstudio.microsoft.com/downloads/) and make sure to install "Common Tools for Visual C++" to enable building c code on Windows. Unfortunately, this is a heavy download / dependency (~4Gb) so you might want to consider WSL or [docker](docker.md) first.

---

Now you have an environment ready, the next step is
[Bot Configuration](configuration.md).
