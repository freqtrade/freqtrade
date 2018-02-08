#!/usr/bin/env bash
#encoding=utf8

function updateenv () {
    echo "
    -------------------------
    Update your virtual env
    -------------------------
    "
    source .env/bin/activate
    pip3.6 install --upgrade pip
    pip3 install -r requirements.txt --upgrade
    pip3 install -r requirements.txt
    pip3 install -e .
}

# Install tab lib
function install_talib () {
    curl -O -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar zxvf ta-lib-0.4.0-src.tar.gz
    cd ta-lib && ./configure --prefix=/usr && make && sudo make install
    cd .. && rm -rf ./ta-lib*
}

# Install bot MacOS
function install_macos () {
    if [ ! -x "$(command -v brew)" ]
    then
        echo "-------------------------"
        echo "Install Brew"
        echo "-------------------------"
        echo
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi
    brew install python3 wget ta-lib
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
    echo
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

    python3.6 -m venv .env
    updateenv
}

function config_generator () {

    echo "Starting to generate config.json"

    echo "-------------------------"
    echo "General configuration"
    echo "-------------------------"
    echo
    read -p "Max open trades: (Default: 3) " max_trades

    read -p "Stake amount: (Default: 0.05) " stake_amount

    read -p "Stake currency: (Default: BTC) " stake_currency

    read -p "Fiat currency: (Default: USD) " fiat_currency

    echo "------------------------"
    echo "Bittrex config generator"
    echo "------------------------"
    echo
    read -p "Exchange API key: " api_key
    read -p "Exchange API Secret: " api_secret

    echo "-------------------------"
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
        -e "s/\"your_telegram_chat_id\"/\"$chat_id\"/g"
        -e "s/\"dry_run\": false,/\"dry_run\": true,/g" config.json.example > config.json

}

function config () {
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

    echo "Edit ./config.json to modify Pair and other configurations."
}

function install () {
    echo "-------------------------"
    echo "Install mandatory dependencies"
    echo "-------------------------"
    echo

    if [ "$(uname -s)" == "Darwin" ]
    then
        echo "- You are on macOS"
        install_macos
    elif [ -x "$(command -v apt-get)" ]
    then
        echo "- You are on Debian/Ubuntu"
        install_debian
    else
        echo "This script does not support your OS."
        echo "If you have Python3.6, pip, virtualenv, ta-lib you can continue."
        echo "Wait 10 seconds to continue the next install steps or use ctrl+c to interrupt this shell."
        sleep 10
    fi
    reset
    echo "
    - Install complete.
    "
    config
    echo "You can now use the bot by executing 'source .env/bin/activate; python3 freqtrade/main.py'."
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