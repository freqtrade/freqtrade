# Installation

This page explains how to prepare your environment for running the bot.

Please consider using the prebuilt [docker images](docker.md) to get started quickly while trying out freqtrade evaluating how it operates.

## Prerequisite

### Requirements

Click each one for install guide:

* [Python >= 3.6.x](http://docs.python-guide.org/en/latest/starting/installation/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html) (install instructions below)

 We also recommend a [Telegram bot](telegram-usage.md#setup-your-telegram-bot), which is optional but recommended.

!!! Warning "Up-to-date clock"
    The clock on the system running the bot must be accurate, synchronized to a NTP server frequently enough to avoid problems with communication to the exchanges.

## Quick start

Freqtrade provides the Linux/MacOS Easy Installation script to install all dependencies and help you configure the bot.

!!! Note
    Windows installation is explained [here](#windows).

The easiest way to install and run Freqtrade is to clone the bot Github repository and then run the Easy Installation script, if it's available for your platform.

!!! Note "Version considerations"
    When cloning the repository the default working branch has the name `develop`. This branch contains all last features (can be considered as relatively stable, thanks to automated tests). The `stable` branch contains the code of the last release (done usually once per month on an approximately one week old snapshot of the `develop` branch to prevent packaging bugs, so potentially it's more stable).

!!! Note
    Python3.6 or higher and the corresponding `pip` are assumed to be available. The install-script will warn you and stop if that's not the case. `git` is also needed to clone the Freqtrade repository.

This can be achieved with the following commands:

```bash
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
# git checkout stable  # Optional, see (1)
./setup.sh --install
```

(1) This command switches the cloned repository to the use of the `stable` branch. It's not needed if you wish to stay on the `develop` branch. You may later switch between branches at any time with the `git checkout stable`/`git checkout develop` commands.

## Easy Installation Script (Linux/MacOS)

If you are on Debian, Ubuntu or MacOS Freqtrade provides the script to install, update, configure and reset the codebase of your bot.

```bash
$ ./setup.sh
usage:
	-i,--install    Install freqtrade from scratch
	-u,--update     Command git pull to update.
	-r,--reset      Hard reset your develop/stable branch.
	-c,--config     Easy config generator (Will override your existing file).
```

** --install **

With this option, the script will install the bot and most dependencies:
You will need to have git and python3.6+ installed beforehand for this to work.

* Mandatory software as: `ta-lib`
* Setup your virtualenv under `.env/`

This option is a combination of installation tasks, `--reset` and `--config`.

** --update **

This option will pull the last version of your current branch and update your virtualenv. Run the script with this option periodically to update your bot.

** --reset **

This option will hard reset your branch (only if you are on either `stable` or `develop`) and recreate your virtualenv.

** --config **

DEPRECATED - use `freqtrade new-config -c config.json` instead.

### Activate your virtual environment

Each time you open a new terminal, you must run `source .env/bin/activate`.

------

## Custom Installation

We've included/collected install instructions for Ubuntu 16.04, MacOS, and Windows. These are guidelines and your success may vary with other distros.
OS Specific steps are listed first, the [Common](#common) section below is necessary for all systems.

!!! Note
    Python3.6 or higher and the corresponding pip are assumed to be available.

=== "Ubuntu 16.04"
    #### Install necessary dependencies

    ```bash
    sudo apt-get update
    sudo apt-get install build-essential git
    ```

=== "RaspberryPi/Raspbian"
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

Use the provided ta-lib installation script

```bash
sudo ./build_helpers/install_ta-lib.sh
```

!!! Note
    This will use the ta-lib tar.gz included in this repository.

##### TA-Lib manual installation

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
git checkout stable
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

# Create a new configuration file
freqtrade new-config --config config.json
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

### Anaconda

Freqtrade can also be installed using Anaconda (or Miniconda).

!!! Note
    This requires the [ta-lib](#1-install-ta-lib) C-library to be installed first. See below.

``` bash
conda env create -f environment.yml
```

-----
## Troubleshooting 

### MacOS installation error

Newer versions of MacOS may have installation failed with errors like `error: command 'g++' failed with exit status 1`.

This error will require explicit installation of the SDK Headers, which are not installed by default in this version of MacOS.
For MacOS 10.14, this can be accomplished with the below command.

``` bash
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```

If this file is inexistent, then you're probably on a different version of MacOS, so you may need to consult the internet for specific resolution details.

-----

Now you have an environment ready, the next step is
[Bot Configuration](configuration.md).