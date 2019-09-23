# Local development

The fastest and easiest way to start up is to use docker-compose.develop which gives developers the ability to start the bot up with all the required dependencies, *without* needing to install any freqtrade specific dependencies on your local machine.

# Install
## git
``` bash
sudo apt install git
```

## docker
``` bash
sudo apt install docker
```

## docker-compose
``` bash
sudo apt install docker-compose
```
# Starting the bot
## Develop dockerfile
``` bash
rm docker-compose.yml && mv docker-compose.develop.yml docker-compose.yml
```

## Docker Compose up

``` bash
docker-compose up
```
