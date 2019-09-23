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

## Docker Compose

## Starting

``` bash
docker-compose up
```
![Docker compose up](https://user-images.githubusercontent.com/419355/65456322-47f63a80-de06-11e9-90c6-3c74d1bad0b8.png)

### Rebuilding
``` bash
docker-compose build
```

### Execing (effectively SSH into the container)

The `exec` command requires that the container already be running, if you want to start it
that can be effected by `docker-compose up` or `docker-compose run freqtrade_develop`

``` bash
docker-compose exec freqtrade_develop /bin/bash
```
![image](https://user-images.githubusercontent.com/419355/65456522-ba671a80-de06-11e9-9598-df9ca0d8dcac.png)
