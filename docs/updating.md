# How to update

To update your freqtrade installation, please use one of the below methods, corresponding to your installation method.

## docker-compose

!!! Note "Legacy installations using the `master` image"
    We're switching from master to stable for the release Images - please adjust your docker-file and replace `freqtradeorg/freqtrade:master` with `freqtradeorg/freqtrade:stable`

``` bash
docker-compose pull
docker-compose up -d
```

## Installation via setup script

``` bash
./setup.sh --update
```

!!! Note
    Make sure to run this command with your virtual environment disabled!

## Plain native installation

Please ensure that you're also updating dependencies - otherwise things might break without you noticing.

``` bash
git pull
pip install -U -r requirements.txt
```
