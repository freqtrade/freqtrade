#!/usr/bin/env bash
#encoding=utf8

# Check which python version is installed
function check_installed_python() {
    which python3.7
    if [ $? -eq 0 ]; then
        echo "using Python 3.7"
        PYTHON=python3.7
        return
    fi

    which python3.6
    if [ $? -eq 0 ]; then
        echo "using Python 3.6"
        PYTHON=python3.6
        return
   fi

   if [ -z ${PYTHON} ]; then
        echo "No usable python found. Please make sure to have python3.6 or python3.7 installed"
        exit 1
   fi

}

function updateenv () {
    echo "-------------------------"
    echo "Update your virtual env"
    echo "-------------------------"
    source .env/bin/activate
    echo "pip3 install in-progress. Please wait..."
    pip3 install --quiet --upgrade pip
    pip3 install --quiet --upgrade -r requirements.txt
    pip3 install --quiet -r requirements.txt

    read -p "Do you want to install dependencies for dev [Y/N]? "
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        pip3 install --quiet -r requirements-dev.txt --upgrade
        pip3 install --quiet -r requirements-dev.txt
    else
        echo "Dev dependencies ignored."
    fi

    pip3 install --quiet -e .
    echo "pip3 install completed"
    echo
}

# Install tab lib
function install_talib () {
    cd build_helpers
    tar zxvf ta-lib-0.4.0-src.tar.gz
    cd ta-lib
    sed -i.bak "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h
    ./configure --prefix=/usr/local
    make
    sudo make install
    cd .. && rm -rf ./ta-lib/
    cd ..
}

# Install bot MacOS
function install_macos () {
    if [ ! -x "$(command -v brew)" ]
    then
        echo "-------------------------"
        echo "Install Brew"
        echo "-------------------------"
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi
    brew install python3 wget
    install_talib
    test_and_fix_python_on_mac
}

# Install bot Debian_ubuntu
function install_debian () {
    sudo add-apt-repository ppa:jonathonf/python-3.6
    sudo apt-get update
    sudo apt-get install python3.6 python3.6-venv python3.6-dev build-essential autoconf libtool pkg-config make wget git
    install_talib
}

# Upgrade the bot
function update () {
    git pull
    updateenv
}

# Reset Develop or Master branch
function reset () {
    echo "----------------------------"
    echo "Reset branch and virtual env"
    echo "----------------------------"
    if [ "1" == $(git branch -vv |grep -cE "\* develop|\* master") ]
    then
        if [ -d ".env" ]; then
          echo "- Delete your previous virtual env"
          rm -rf .env
        fi

        git fetch -a

        if [ "1" == $(git branch -vv |grep -c "* develop") ]
        then
          echo "- Hard resetting of 'develop' branch."
          git reset --hard origin/develop
        elif [ "1" == $(git branch -vv |grep -c "* master") ]
        then
          echo "- Hard resetting of 'master' branch."
          git reset --hard origin/master
        fi
    else
        echo "Reset ignored because you are not on 'master' or 'develop'."
    fi

    echo
    ${PYTHON} -m venv .env
    updateenv
}

function test_and_fix_python_on_mac() {

    if ! [ -x "$(command -v python3.6)" ]
    then
        echo "-------------------------"
        echo "Fixing Python"
        echo "-------------------------"
        echo "Python 3.6 is not linked in your system. Fixing it..."
        brew link --overwrite python
        echo
    fi
}

function config_generator () {

    echo "Starting to generate config.json"
    echo
    echo "General configuration"
    echo "-------------------------"
    default_max_trades=3
    read -p "Max open trades: (Default: $default_max_trades) " max_trades
    max_trades=${max_trades:-$default_max_trades}

    default_stake_amount=0.05
    read -p "Stake amount: (Default: $default_stake_amount) " stake_amount
    stake_amount=${stake_amount:-$default_stake_amount}

    default_stake_currency="BTC"
    read -p "Stake currency: (Default: $default_stake_currency) " stake_currency
    stake_currency=${stake_currency:-$default_stake_currency}

    default_fiat_currency="USD"
    read -p "Fiat currency: (Default: $default_fiat_currency) " fiat_currency
    fiat_currency=${fiat_currency:-$default_fiat_currency}

    echo
    echo "Exchange config generator"
    echo "------------------------"
    read -p "Exchange API key: " api_key
    read -p "Exchange API Secret: " api_secret

    echo
    echo "Telegram config generator"
    echo "-------------------------"
    read -p "Telegram Token: " token
    read -p "Telegram Chat_id: " chat_id

    sed -e "s/\"max_open_trades\": 3,/\"max_open_trades\": $max_trades,/g" \
        -e "s/\"stake_amount\": 0.05,/\"stake_amount\": $stake_amount,/g" \
        -e "s/\"stake_currency\": \"BTC\",/\"stake_currency\": \"$stake_currency\",/g" \
        -e "s/\"fiat_display_currency\": \"USD\",/\"fiat_display_currency\": \"$fiat_currency\",/g" \
        -e "s/\"your_exchange_key\"/\"$api_key\"/g" \
        -e "s/\"your_exchange_secret\"/\"$api_secret\"/g" \
        -e "s/\"your_telegram_token\"/\"$token\"/g" \
        -e "s/\"your_telegram_chat_id\"/\"$chat_id\"/g" \
        -e "s/\"dry_run\": false,/\"dry_run\": true,/g" config.json.example > config.json

}

function config () {

    echo "-------------------------"
    echo "Config file generator"
    echo "-------------------------"
    if [ -f config.json ]
    then
    read -p "A config file already exist, do you want to override it [Y/N]? "
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        config_generator
    else
        echo "Configuration of config.json ignored."
    fi
    else
        config_generator
    fi

    echo
    echo "-------------------------"
    echo "Config file generated"
    echo "-------------------------"
    echo "Edit ./config.json to modify Pair and other configurations."
    echo
}

function install () {
    echo "-------------------------"
    echo "Install mandatory dependencies"
    echo "-------------------------"

    if [ "$(uname -s)" == "Darwin" ]
    then
        echo "macOS detected. Setup for this system in-progress"
        install_macos
    elif [ -x "$(command -v apt-get)" ]
    then
        echo "Debian/Ubuntu detected. Setup for this system in-progress"
        install_debian
    else
        echo "This script does not support your OS."
        echo "If you have Python3.6 or Python3.7, pip, virtualenv, ta-lib you can continue."
        echo "Wait 10 seconds to continue the next install steps or use ctrl+c to interrupt this shell."
        sleep 10
    fi
    echo
    reset
    config
    echo "-------------------------"
    echo "Run the bot"
    echo "-------------------------"
    echo "You can now use the bot by executing 'source .env/bin/activate; python freqtrade/main.py'."
}

function plot () {
echo "
-----------------------------------------
Install dependencies for Plotting scripts
-----------------------------------------
"
pip install plotly --upgrade
}

function help () {
    echo "usage:"
    echo "	-i,--install    Install freqtrade from scratch"
    echo "	-u,--update     Command git pull to update."
    echo "	-r,--reset      Hard reset your develop/master branch."
    echo "	-c,--config     Easy config generator (Will override your existing file)."
    echo "	-p,--plot       Install dependencies for Plotting scripts."
}

# Verify if 3.6 or 3.7 is installed
check_installed_python

case $* in
--install|-i)
install
;;
--config|-c)
config
;;
--update|-u)
update
;;
--reset|-r)
reset
;;
--plot|-p)
plot
;;
*)
help
;;
esac
exit 0
