# Installation

This page explains how to prepare your environment for running the bot.


The documentation describes three ways to install freqtrade
* Manual 
* Conda 
* Docker (separate file)

Please consider using the prebuilt [docker images](docker.md) to get started quickly to try freqtrade and evaluate how it works.

------

### Information

#### Set of Notes

Freqtrade provides the Linux/MacOS `./setup.sh` script to install all dependencies and help you configure the bot.

!!! Note
    Windows installation is explained [here](#windows).

The easiest way to install and run Freqtrade is to clone the bot Github repository and then run the `./sectup.sh` script, if it's available for your platform.

!!! Note "Version considerations"
    When cloning the repository the default working branch has the name `develop`. This branch contains all last features (can be considered as relatively stable, thanks to automated tests).
    The `stable` branch contains the code of the last release (done usually once per month on an approximately one week old snapshot of the `develop` branch to prevent packaging bugs, so potentially it's more stable).

!!! Note
    Python3.7 or higher and the corresponding `pip` are assumed to be available. The install-script will warn you and stop if that's not the case. `git` is also needed to clone the Freqtrade repository.  
    Also, python headers (`python<yourversion>-dev` / `python<yourversion>-devel`) must be available for the installation to complete successfully.

!!! Warning "Up-to-date clock"
    The clock on the system running the bot must be accurate, synchronized to a NTP server frequently enough to avoid problems with communication to the exchanges.

#### Freqtrade repository

Freqtrade is an open source cryptocurrency trading bot, whose code is hosted on `github.com`

```bash
# Download `develop` branch of freqtrade repository
git clone https://github.com/freqtrade/freqtrade.git

# Enter downloaded directory
cd freqtrade

# your choice (1)
git checkout stable

# your choice (2)
git checkout develop
```

(1) This command switches the cloned repository to the use of the `stable` branch. It's not needed, if you wish to stay on the (2) `develop` branch.

You may later switch between branches at any time with the `git checkout stable`/`git checkout develop` commands.

#### Notes to /setup.sh script (Linux/MacOS)

If you are on Debian, Ubuntu or MacOS Freqtrade provides the script to install, update, configure and reset the codebase of your bot.

```bash
usage:
    # Install freqtrade from scratch
    ./setup.sh -i,--install

    # Command git pull to update.
    ./setup.sh -u,--update     

    # Hard reset your develop/stable branch.
    ./setup.sh -r,--reset      

    # Easy config generator (Will override your existing file).
    ./setup.sh -c,--config
```

    ** --install **

    With this option, the script will install the bot and most dependencies:
    You will need to have git and python3.7+ installed beforehand for this to work.

    * Mandatory software as: `ta-lib`
    * Setup your virtualenv under `.env/`

    This option is a combination of installation tasks, `--reset` and `--config`.

    ** --update **

    This option will pull the last version of your current branch and update your virtualenv. Run the script with this option periodically to update your bot.

    ** --reset **

    This option will hard reset your branch (only if you are on either `stable` or `develop`) and recreate your virtualenv.

    ** --config **

    DEPRECATED - use `freqtrade new-config -c config.json` instead.

------

## Manual Installation

#### Requirements Part A

Click each one for install guide:

* [Python >= 3.7.x](http://docs.python-guide.org/en/latest/starting/installation/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html) (install instructions below)

 We also recommend a [Telegram bot](telegram-usage.md#setup-your-telegram-bot), which is optional but recommended.

#### Requirements Part B

We've included/collected install instructions for Ubuntu, MacOS, and Windows. These are guidelines and your success may vary with other distros.
OS Specific steps are listed first, the [Common](#common) section below is necessary for all systems.

!!! Note
    Python3.7 or higher and the corresponding pip are assumed to be available.

=== "Debian/Ubuntu"
    #### Install necessary dependencies

    ```bash
    # update repository
    sudo apt-get update

    # install packages
    sudo apt install -y python3-pip \
	python3-venv \
	python3-pandas \
	python3-pip \
	git
    ```

=== "RaspberryPi/Raspbian"
    The following assumes the latest [Raspbian Buster lite image](https://www.raspberrypi.org/downloads/raspbian/).
    This image comes with python3.7 preinstalled, making it easy to get freqtrade up and running.

    Tested using a Raspberry Pi 3 with the Raspbian Buster lite image, all updates applied.


    ```bash
    sudo apt-get install python3-venv libatlas-base-dev cmake
    # Use pywheels.org to speed up installation
    sudo echo "[global]\nextra-index-url=https://www.piwheels.org/simple" > tee /etc/pip.conf

    git clone https://github.com/freqtrade/freqtrade.git
    cd freqtrade

    bash setup.sh -i
    ```

    !!! Note "Installation duration"
        Depending on your internet speed and the Raspberry Pi version, installation can take multiple hours to complete.
        Due to this, we recommend to use the prebuild docker-image for Raspberry, by following the [Docker quickstart documentation](docker_quickstart.md)

    !!! Note
        The above does not install hyperopt dependencies. To install these, please use `python3 -m pip install -e .[hyperopt]`.
        We do not advise to run hyperopt on a Raspberry Pi, since this is a very resource-heavy operation, which should be done on powerful machine.


#### Install TA-Lib

##### TA-Lib script installation

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


#### Install Freqtrade

Clone the git repository:

```bash
# download repository
git clone https://github.com/freqtrade/freqtrade.git

# enter freqtrade directory
cd freqtrade
git checkout stable

# run installation script
./setup.sh --install
```

#### Setup Python virtual environment (virtualenv)

You will run freqtrade in separated `virtual environment`

```bash
# create virtualenv in directory /freqtrade/.env
python3 -m venv .env

# run virtualenv
source .env/bin/activate
```

#### Install python dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

#### Initialize the configuration

```bash
# Initialize the user_directory
freqtrade create-userdir --userdir user_data/

# Create a new configuration file
freqtrade new-config --config config.json
```

> *To edit the config please refer to [Bot Configuration](configuration.md).*

#### Run the Bot

If this is the first time you run the bot, ensure you are running it in Dry-run `"dry_run": true,` otherwise it will start to buy and sell coins.

```bash
freqtrade trade -c config.json
```

#### Problem?

Check, if your virtual environment is activated, is you get error as below:

```bash
bash: freqtrade: command not found

# then activate your .env
source ./.env/bin/activate
```


#### (Optional) Post-installation Tasks

*Note*: If you run the bot on a server, you should consider using [Docker](docker.md) or a terminal multiplexer like `screen` or [`tmux`](https://en.wikipedia.org/wiki/Tmux) to avoid that the bot is stopped on logout.


On Linux with software suite `systemd`, as an optional post-installation task, you may wish to setup the bot to run as a `systemd service` or configure it to send the log messages to the `syslog`/`rsyslog` or `journald` daemons. See [Advanced Logging](advanced-setup.md#advanced-logging) for details.

------

## Installation with Conda (Miniconda or Anaconda)

Freqtrade can also be installed with Miniconda or Anaconda. Conda will automatically prepare and manage the extensive library-dependencies of the Freqtrade program.

##### What is Conda?

It is: (1) package, (2) dependency and (3) environment management for any programming language : https://docs.conda.io/projects/conda/en/latest/index.html

We recommend using Miniconda as it's installation footprint is smaller.

### installation

#### Install Conda

[Installing on linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent)

[Installing on windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)

Answer all questions. After installation, it is mandatory to turn your terminal OFF and ON again.

#### Freqtrade download

Download and install freqtrade.

```bash
# download freqtrade
git clone https://github.com/freqtrade/freqtrade.git

# enter downloaded directory 'freqtrade'
cd freqtrade      
```

#### Freqtrade install : Conda Environment

Prepare conda-freqtrade environment, using file `environment.yml`, which exist in main freqtrade directory

```bash
conda env create -n freqtrade-conda -f environment.yml
```

#### Enter/exit freqtrade-conda venv:

To check available environments, type

```bash
conda env list
```

Enter installed environment

```bash
# enter conda environment
conda activate freqtrade-conda

# exit - don`t do it now
conda deactivate
```urce command-line utility widely used on Linux and other Unix-flavored operating systems. It is designed to give selected, trusted users administrative control when needed.

Install last python dependencies with pip

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```


### You are ready

Do:

```bash
# Step 1 - create user folder
freqtrade create-userdir --userdir user_data

# Step 2 - create config file
freqtrade new-config --config config.json
```

You are ready to run, read [Bot Configuration](configuration.md), remember to run program as `dry_run: True` and verify that everything is working.

important shortcuts

```bash
# list installed conda environments
conda env list

# activate base environment
conda activate

# activate freqtrade-conda environment
conda activate freqtrade-conda

#deactivate any conda environments
conda deactivate                              
```

### Notes

#### Set of notes 1 - Conda settings

After opening terminal, you already will be in default `base` conda environment.
If you want, you can prevent the (base) conda environment from being activated automatically.

```bash
conda config --set auto_activate_base false
```

Channel `conda-forge` is supposingly best source of the conda updates. Switch to it

```bash    
# adding forge
conda config --env --add channels conda-forge

# make it strict
conda config --env --set channel_priority strict 	

# check status of your conda
conda info
conda config --show channels
conda config --show channel_priority
```

Further read on the topic:

https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533?gi=1db972389cd1

#### Set of Notes 2

!!! Note "Creating Conda Environment"
    The conda command `create -n` automatically installs all nested dependencies for the selected libraries, general structure of installation command is:

```bash
# choose your own packages
conda env create -n [name of the environment] [python version] [packages]

# point to file with packages
conda env create -n [name of the environment] -f [file]
```

!!! Info "New heavy packages"
    It may happen that creating a new Conda environment, populated with selected packages at the moment of creation, takes less time than installing a large, heavy dependent, GUI package, into previously set environment. Great example: Spyder

!!! Warning "pip install within conda"
    Please read the section [Market order pricing](#market-order-pricing) section when using market orders.

    The documentation of conda says that pip should NOT be used within conda, because internal problems can occur.
    However, they are rare. https://www.anaconda.com/blog/using-pip-in-a-conda-environment

    Nevertherless, that is why, the `conda-forge` channel is preferred:

    * more libraries are available (less need for `pip`)
    * `conda-forge` works better with `pip`
    * the libraries are newer


Happy trading!


-----
## Troubleshooting

### MacOS installation error

Newer versions of MacOS may have installation failed with errors like `error: command 'g++' failed with exit status 1`.

This error will require explicit installation of the SDK Headers, which are not installed by default in this version of MacOS.
For MacOS 10.14, this can be accomplished with the below command.

```bash
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```

If this file is inexistent, then you're probably on a different version of MacOS, so you may need to consult the internet for specific resolution details.

### MacOS installation error with python 3.9

When using python 3.9 on macOS, it's currently necessary to install some os-level modules to allow dependencies to compile.
The errors you'll see happen during installation and are related to the installation of `tables` or `blosc`.

You can install the necessary libraries with the following command:

```bash
brew install hdf5 c-blosc
```

After this, please run the installation (script) again.

-----

Now you have an environment ready, the next step is
[Bot Configuration](configuration.md).
