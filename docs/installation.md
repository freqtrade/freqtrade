# Install the bot
This page explains how to prepare your environment for running the bot.
To understand how to set up the bot please read the Bot
[Bot configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md)
page.

## Table of Contents
- [Docker Automatic Installation](#docker)
- [Linux or Mac manual Installation](#linux--mac)
    - [Linux - Ubuntu 16.04](#21-linux---ubuntu-1604)
    - [Linux - Other distro](#22-linux---other-distro)
    - [MacOS installation](#23-macos-installation)
    - [Advanced Linux ](#advanced-linux)
- [Windows manual Installation](#windows)

# Docker

## Easy installation
Start by downloading Docker for your platform:
- [Mac](https://www.docker.com/products/docker#/mac)
- [Windows](https://www.docker.com/products/docker#/windows)
- [Linux](https://www.docker.com/products/docker#/linux)

Once you have Docker installed, simply create the config file
(e.g. `config.json`) and then create a Docker image for `freqtrade`
using the Dockerfile in this repo.

### 1. Prepare the bot
1. Clone the git
```bash
git clone https://github.com/gcarq/freqtrade.git
```
2. (Optional) Checkout the develop branch
```bash
git checkout develop
```
3. Go into the new directory
```bash
cd freqtrade
```
4. Copy `config.sample` to `config.json`
```bash
cp config.json.example config.json
```
To edit the config please refer to the [Bot Configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md) page
5. Create your DB file (Optional, the bot will create it if it is missing)
```bash
# For Production
touch tradesv3.sqlite

# For Dry-run
touch tradesv3.dryrun.sqlite
```

### 2. Build the docker image
```bash
cd freqtrade
docker build -t freqtrade .
```

For security reasons, your configuration file will not be included in the
image, you will need to bind mount it. It is also advised to bind mount
a sqlite database file (see the "5. Run a restartable docker image"
section) to keep it between  updates.

### 3. Verify the docker image
After build process you can verify that the image was created with:
```
docker images
```

### 4. Run the docker image
You can run a one-off container that is immediately deleted upon exiting with
the following command (config.json must be in the current working directory):

```
docker run --rm -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

In this example, the database will be created inside the docker instance
and will be lost when you will refresh your image.

### 5. Run a restartable docker image
To run a restartable instance in the background (feel free to place your
configuration and database files wherever it feels comfortable on your
filesystem).

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
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  freqtrade
```
If you are using `dry_run=True` it's not necessary to mount
`tradesv3.sqlite`, but you can mount `tradesv3.dryrun.sqlite` if you
plan to use the dry run mode with the param `--dry-run-db`.


### 6. Monitor your Docker instance
You can then use the following commands to monitor and manage your container:

```bash
docker logs freqtrade
docker logs -f freqtrade
docker restart freqtrade
docker stop freqtrade
docker start freqtrade
```

You do not need to rebuild the image for configuration changes, it will
suffice to edit `config.json` and restart the container.


# Linux / MacOS
## 1. Requirements
Click each one for install guide:
- [Python 3.6.x](http://docs.python-guide.org/en/latest/starting/installation/),
note the bot was not tested on Python >= 3.7.x
- [pip](https://pip.pypa.io/en/stable/installing/)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (Recommended)
- [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html)

## 2. First install required packages
This bot require Python 3.6 and TA-LIB

### 2.1 Linux - Ubuntu 16.04

**2.1.1. Install Python 3.6, Git, and wget**
```bash
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6 python3.6-venv build-essential autoconf libtool pkg-config make wget git
```

**2.1.2. Install TA-LIB**
Official webpage: https://mrjbq7.github.io/ta-lib/install.html
```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install
cd ..
rm -rf ./ta-lib*
```

**2.1.3. [Optional] Install MongoDB**
Install MongoDB if you plan to optimize your strategy with Hyperopt.

```bash
sudo apt-get install mongodb-org
```
Complete tutorial on [Digital Ocean: How to Install MongoDB on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-mongodb-on-ubuntu-16-04)

### 2.2. Linux - Other distro
If you are on a different Linux OS you maybe have to adapt things like:

- package manager (for example yum instead of apt-get)
- package names

### 2.3. MacOS installation

**2.3.1. Install Python 3.6, git and wget**
```bash
brew install python3 git wget
```

**2.3.2. [Optional] Install MongoDB**
Install MongoDB if you plan to optimize your strategy with Hyperopt.
```bash
curl -O https://fastdl.mongodb.org/osx/mongodb-osx-ssl-x86_64-3.4.10.tgz
tar -zxvf mongodb-osx-ssl-x86_64-3.4.10.tgz
mkdir -p <path_freqtrade>/env/mongodb
cp -R -n mongodb-osx-x86_64-3.4.10/ <path_freqtrade>/env/mongodb
export PATH=<path_freqtrade>/env/mongodb/bin:$PATH
```

## 3. Clone the repo
The following steps are made for Linux/mac environment
1. Clone the git `git clone https://github.com/gcarq/freqtrade.git`
2. (Optional) Checkout the develop branch `git checkout develop`

## 4. Prepare the bot
```bash
cd freqtrade
cp config.json.example config.json
```
To edit the config please refer to [Bot Configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md)

## 5. Setup your virtual env
```bash
python3.6 -m venv .env
source .env/bin/activate
pip3.6 install -r requirements.txt
pip3.6 install -e .
```

## 6. Run the bot
If this is the first time you run the bot, ensure you are running it
in Dry-run `"dry_run": true,` otherwise it will start to buy and sell coins.

```bash
python3.6 ./freqtrade/main.py -c config.json
```

### Advanced Linux
**systemd service file**
Copy `./freqtrade.service` to your systemd user directory (usually `~/.config/systemd/user`)
and update `WorkingDirectory` and `ExecStart` to match your setup.
After that you can start the daemon with:
```bash
systemctl --user start freqtrade
```

# Windows
We do recommend Windows users to use [Docker](#docker) this will work
much easier and smoother (also safer).

```cmd
#copy paste config.json to \path\freqtrade-develop\freqtrade
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
*Thanks [Owdr](https://github.com/Owdr) for the commands. Source: [Issue #222](https://github.com/gcarq/freqtrade/issues/222)*

## Next step
Now you have an environment ready, the next step is to
[configure your bot](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md).
