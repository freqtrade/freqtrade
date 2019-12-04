# Installation

This page explains how to prepare your environment for running the bot.

## Prerequisite

### Requirements

Click each one for install guide:

* [Python >= 3.6.x](http://docs.python-guide.org/en/latest/starting/installation/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (Recommended)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html) (install instructions below)

### API keys

Before running your bot in production you will need to setup few
external API. In production mode, the bot will require valid Exchange API
credentials. We also recommend a [Telegram bot](telegram-usage.md#setup-your-telegram-bot) (optional but recommended).

### Setup your exchange account

You will need to create API Keys (Usually you get `key` and `secret`) from the Exchange website and insert this into the appropriate fields in the configuration or when asked by the installation script.

## Quick start

Freqtrade provides the Linux/MacOS Easy Installation script to install all dependencies and help you configure the bot.

!!! Note
    Windows installation is explained [here](#windows).

The easiest way to install and run Freqtrade is to clone the bot GitHub repository and then run the Easy Installation script, if it's available for your platform.

!!! Note "Version considerations"
    When cloning the repository the default working branch has the name `develop`. This branch contains all last features (can be considered as relatively stable, thanks to automated tests). The `master` branch contains the code of the last release (done usually once per month on an approximately one week old snapshot of the `develop` branch to prevent packaging bugs, so potentially it's more stable).

!!! Note
    Python3.6 or higher and the corresponding `pip` are assumed to be available. The install-script will warn you and stop if that's not the case. `git` is also needed to clone the Freqtrade repository.

This can be achieved with the following commands:

```bash
git clone git@github.com:freqtrade/freqtrade.git
cd freqtrade
git checkout master  # Optional, see (1)
./setup.sh --install
```
(1) This command switches the cloned repository to the use of the `master` branch. It's not needed if you wish to stay on the `develop` branch. You may later switch between branches at any time with the `git checkout master`/`git checkout develop` commands.

## Easy Installation Script (Linux/MacOS)

If you are on Debian, Ubuntu or MacOS Freqtrade provides the script to install, update, configure and reset the codebase of your bot.

```bash
$ ./setup.sh
usage:
	-i,--install    Install freqtrade from scratch
	-u,--update     Command git pull to update.
	-r,--reset      Hard reset your develop/master branch.
	-c,--config     Easy config generator (Will override your existing file).
```

** --install **

With this option, the script will install everything you need to run the bot:

* Mandatory software as: `ta-lib`
* Setup your virtualenv
* Configure your `config.json` file

This option is a combination of installation tasks, `--reset` and `--config`.

** --update **

This option will pull the last version of your current branch and update your virtualenv. Run the script with this option periodically to update your bot.

** --reset **

This option will hard reset your branch (only if you are on either `master` or `develop`) and recreate your virtualenv.

** --config **

Use this option to configure the `config.json` configuration file. The script will interactively ask you questions to setup your bot and create your `config.json`.

------

## Custom Installation

We've included/collected install instructions for Ubuntu 16.04, MacOS, and Windows. These are guidelines and your success may vary with other distros.
OS Specific steps are listed first, the [Common](#common) section below is necessary for all systems.

!!! Note
    Python3.6 or higher and the corresponding pip are assumed to be available.

### Linux - Ubuntu 16.04

#### Install necessary dependencies

```bash
sudo apt-get update
sudo apt-get install build-essential git
```

### Raspberry Pi / Raspbian

The following assumes the latest [Raspbian Buster lite image](https://www.raspberrypi.org/downloads/raspbian/) from at least September 2019.
This image comes with python3.7 preinstalled, making it easy to get freqtrade up and running.

Tested using a Raspberry Pi 3 with the Raspbian Buster lite image, all updates applied.

``` bash
sudo apt-get install python3-venv libatlas-base-dev
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade

bash setup.sh -i
```

!!! Note "Installation duration"
    Depending on your internet speed and the Raspberry Pi version, installation can take multiple hours to complete.

!!! Note
    The above does not install hyperopt dependencies. To install these, please use `python3 -m pip install -e .[hyperopt]`.
    We do not advise to run hyperopt on a Raspberry Pi, since this is a very resource-heavy operation, which should be done on powerful machine.

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

#### 3. Install Freqtrade

Clone the git repository:

```bash
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
```

Optionally checkout the master branch to get the latest stable release:

```bash
git checkout master
```

#### 4. Install python dependencies

``` bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

#### 5. Initialize the configuration

```bash
# Initialize the user_directory
freqtrade create-userdir --userdir user_data/

cp config.json.example config.json
```

> *To edit the config please refer to [Bot Configuration](configuration.md).*

#### 6. Run the Bot

If this is the first time you run the bot, ensure you are running it in Dry-run `"dry_run": true,` otherwise it will start to buy and sell coins.

```bash
freqtrade trade -c config.json
```

*Note*: If you run the bot on a server, you should consider using [Docker](docker.md) or a terminal multiplexer like `screen` or [`tmux`](https://en.wikipedia.org/wiki/Tmux) to avoid that the bot is stopped on logout.

#### 7. (Optional) Post-installation Tasks

On Linux, as an optional post-installation task, you may wish to setup the bot to run as a `systemd` service or configure it to send the log messages to the `syslog`/`rsyslog` or `journald` daemons. See [Advanced Logging](advanced-setup.md#advanced-logging) for details.

------

## Using Conda

Freqtrade can also be installed using Anaconda (or Miniconda).

``` bash
conda env create -f environment.yml
```

!!! Note
    This requires the [ta-lib](#1-install-ta-lib) C-library to be installed first.

## Windows

We recommend that Windows users use [Docker](docker.md) as this will work much easier and smoother (also more secure).

If that is not possible, try using the Windows Linux subsystem (WSL) - for which the Ubuntu instructions should work.
If that is not available on your system, feel free to try the instructions below, which led to success for some.

### Install freqtrade manually

!!! Note
    Make sure to use 64bit Windows and 64bit Python to avoid problems with backtesting or hyperopt due to the memory constraints 32bit applications have under Windows.

!!! Hint
    Using the [Anaconda Distribution](https://www.anaconda.com/distribution/) under Windows can greatly help with installation problems. Check out the [Conda section](#using-conda) in this document for more information.

#### Clone the git repository

```bash
git clone https://github.com/freqtrade/freqtrade.git
```

#### Install ta-lib

Install ta-lib according to the [ta-lib documentation](https://github.com/mrjbq7/ta-lib#windows).

As compiling from source on windows has heavy dependencies (requires a partial visual studio installation), there is also a repository of unofficial precompiled windows Wheels [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib), which needs to be downloaded and installed using `pip install TA_Lib‑0.4.17‑cp36‑cp36m‑win32.whl` (make sure to use the version matching your python version)

```cmd
>cd \path\freqtrade-develop
>python -m venv .env
>.env\Scripts\activate.bat
REM optionally install ta-lib from wheel
REM >pip install TA_Lib‑0.4.17‑cp36‑cp36m‑win32.whl
>pip install -r requirements.txt
>pip install -e .
>freqtrade
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
