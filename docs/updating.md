# How to update

To update your freqtrade installation, please use one of the below methods, corresponding to your installation method.

!!! Note "Tracking changes"
    Breaking changes / changed behavior will be documented in the changelog that is posted alongside every release.
    For the develop branch, please follow PR's to avoid being surprised by changes.

## docker

!!! Note "Legacy installations using the `master` image"
    We're switching from master to stable for the release Images - please adjust your docker-file and replace `freqtradeorg/freqtrade:master` with `freqtradeorg/freqtrade:stable`

``` bash
docker compose pull
docker compose up -d
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
pip install -e .

# Ensure freqUI is at the latest version
freqtrade install-ui 
```

### Problems updating

Update-problems usually come missing dependencies (you didn't follow the above instructions) - or from updated dependencies, which fail to install (for example TA-lib).
Please refer to the corresponding installation sections (common problems linked below)

Common problems and their solutions:

* [ta-lib update on windows](windows_installation.md#2-install-ta-lib)
