#!/usr/bin/env bash
#encoding=utf8

function check_installed_pip() {
   ${PYTHON} -m pip > /dev/null
   if [ $? -ne 0 ]; then
        echo "pip not found (called as '${PYTHON} -m pip'). Please make sure that pip is available for ${PYTHON}."
        exit 1
   fi
}

# Check which python version is installed
function check_installed_python() {
    if [ -n "${VIRTUAL_ENV}" ]; then
        echo "Please deactivate your virtual environment before running setup.sh."
        echo "You can do this by running 'deactivate'."
        exit 2
    fi

    which python3.8
    if [ $? -eq 0 ]; then
        echo "using Python 3.8"
        PYTHON=python3.8
        check_installed_pip
        return
    fi

    which python3.9
    if [ $? -eq 0 ]; then
        echo "using Python 3.9"
        PYTHON=python3.9
        check_installed_pip
        return
    fi

    which python3.7
    if [ $? -eq 0 ]; then
        echo "using Python 3.7"
        PYTHON=python3.7
        check_installed_pip
        return
    fi


   if [ -z ${PYTHON} ]; then
        echo "No usable python found. Please make sure to have python3.7 or newer installed"
        exit 1
   fi
}

function updateenv() {
    echo "-------------------------"
    echo "Updating your virtual env"
    echo "-------------------------"
    if [ ! -f .env/bin/activate ]; then
        echo "Something went wrong, no virtual environment found."
        exit 1
    fi
    source .env/bin/activate
    SYS_ARCH=$(uname -m)
    echo "pip install in-progress. Please wait..."
    ${PYTHON} -m pip install --upgrade pip
    read -p "Do you want to install dependencies for dev [y/N]? "
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        REQUIREMENTS=requirements-dev.txt
    else
        REQUIREMENTS=requirements.txt
    fi
    REQUIREMENTS_HYPEROPT=""
    REQUIREMENTS_PLOT=""
     read -p "Do you want to install plotting dependencies (plotly) [y/N]? "
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        REQUIREMENTS_PLOT="-r requirements-plot.txt"
    fi
    if [ "${SYS_ARCH}" == "armv7l" ]; then
        echo "Detected Raspberry, installing cython, skipping hyperopt installation."
        ${PYTHON} -m pip install --upgrade cython
    else
        # Is not Raspberry
        read -p "Do you want to install hyperopt dependencies [y/N]? "
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            REQUIREMENTS_HYPEROPT="-r requirements-hyperopt.txt"
        fi
    fi

    ${PYTHON} -m pip install --upgrade -r ${REQUIREMENTS} ${REQUIREMENTS_HYPEROPT} ${REQUIREMENTS_PLOT}
    if [ $? -ne 0 ]; then
        echo "Failed installing dependencies"
        exit 1
    fi
    ${PYTHON} -m pip install -e .
    if [ $? -ne 0 ]; then
        echo "Failed installing Freqtrade"
        exit 1
    fi
    echo "pip install completed"
    echo
}

# Install tab lib
function install_talib() {
    if [ -f /usr/local/lib/libta_lib.a ]; then
        echo "ta-lib already installed, skipping"
        return
    fi

    cd build_helpers
    tar zxvf ta-lib-0.4.0-src.tar.gz
    cd ta-lib
    sed -i.bak "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h
    ./configure --prefix=/usr/local
    make
    sudo make install
    if [ -x "$(command -v apt-get)" ]; then
        echo "Updating library path using ldconfig"
        sudo ldconfig
    fi
    cd .. && rm -rf ./ta-lib/
    cd ..
}

# Install bot MacOS
function install_macos() {
    if [ ! -x "$(command -v brew)" ]
    then
        echo "-------------------------"
        echo "Installing Brew"
        echo "-------------------------"
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi
    install_talib
    test_and_fix_python_on_mac
}

# Install bot Debian_ubuntu
function install_debian() {
    sudo apt-get update
    sudo apt-get install -y build-essential autoconf libtool pkg-config make wget git libpython3-dev
    install_talib
}

# Upgrade the bot
function update() {
    git pull
    updateenv
}

# Reset Develop or Stable branch
function reset() {
    echo "----------------------------"
    echo "Reseting branch and virtual env"
    echo "----------------------------"

    if [ "1" == $(git branch -vv |grep -cE "\* develop|\* stable") ]
    then

        read -p "Reset git branch? (This will remove all changes you made!) [y/N]? "
        if [[ $REPLY =~ ^[Yy]$ ]]; then

            git fetch -a

            if [ "1" == $(git branch -vv | grep -c "* develop") ]
            then
                echo "- Hard resetting of 'develop' branch."
                git reset --hard origin/develop
            elif [ "1" == $(git branch -vv | grep -c "* stable") ]
            then
                echo "- Hard resetting of 'stable' branch."
                git reset --hard origin/stable
            fi
        fi
    else
        echo "Reset ignored because you are not on 'stable' or 'develop'."
    fi

    if [ -d ".env" ]; then
        echo "- Deleting your previous virtual env"
        rm -rf .env
    fi
    echo
    ${PYTHON} -m venv .env
    if [ $? -ne 0 ]; then
        echo "Could not create virtual environment. Leaving now"
        exit 1
    fi
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

function config() {

    echo "-------------------------"
    echo "Please use 'freqtrade new-config -c config.json' to generate a new configuration file."
    echo "-------------------------"
}

function install() {
    echo "-------------------------"
    echo "Installing mandatory dependencies"
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
    echo "Run the bot !"
    echo "-------------------------"
    echo "You can now use the bot by executing 'source .env/bin/activate; freqtrade <subcommand>'."
    echo "You can see the list of available bot sub-commands by executing 'source .env/bin/activate; freqtrade --help'."
    echo "You verify that freqtrade is installed successfully by running 'source .env/bin/activate; freqtrade --version'."
}

function plot() {
echo "
-----------------------------------------
Installing dependencies for Plotting scripts
-----------------------------------------
"
${PYTHON} -m pip install plotly --upgrade
}

function help() {
    echo "usage:"
    echo "	-i,--install    Install freqtrade from scratch"
    echo "	-u,--update     Command git pull to update."
    echo "	-r,--reset      Hard reset your develop/stable branch."
    echo "	-c,--config     Easy config generator (Will override your existing file)."
    echo "	-p,--plot       Install dependencies for Plotting scripts."
}

# Verify if 3.7 or 3.8 is installed
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
