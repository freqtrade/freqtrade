# Advanced Post-installation Tasks

This page explains some advanced tasks and configuration options that can be performed after the bot installation and may be uselful in some environments.

If you do not know what things mentioned here mean, you probably do not need it.

## Running multiple instances of Freqtrade

This section will show you how to run multiple bots at the same time, on the same machine.

### Things to consider

* Use different database files.
* Use different Telegram bots (requires multiple different configuration files; applies only when Telegram is enabled).
* Use different ports (applies only when Freqtrade REST API webserver is enabled).

### Different database files

In order to keep track of your trades, profits, etc., freqtrade is using a SQLite database where it stores various types of information such as the trades you performed in the past and the current position(s) you are holding at any time. This allows you to keep track of your profits, but most importantly, keep track of ongoing activity if the bot process would be restarted or would be terminated unexpectedly.

Freqtrade will, by default, use separate database files for dry-run and live bots (this assumes no database-url is given in either configuration nor via command line argument).
For live trading mode, the default database will be `tradesv3.sqlite` and for dry-run it will be `tradesv3.dryrun.sqlite`.

The optional argument to the trade command used to specify the path of these files is `--db-url`, which requires a valid SQLAlchemy url.
So when you are starting a bot with only the config and strategy arguments in dry-run mode, the following 2 commands would have the same outcome.

``` bash
freqtrade trade -c MyConfig.json -s MyStrategy
# is equivalent to
freqtrade trade -c MyConfig.json -s MyStrategy --db-url sqlite:///tradesv3.dryrun.sqlite
```

It means that if you are running the trade command in two different terminals, for example to test your strategy both for trades in USDT and in another instance for trades in BTC, you will have to run them with different databases.

If you specify the URL of a database which does not exist, freqtrade will create one with the name you specified. So to test your custom strategy with BTC and USDT stake currencies, you could use the following commands (in 2 separate terminals):

``` bash
# Terminal 1:
freqtrade trade -c MyConfigBTC.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesBTC.dryrun.sqlite
# Terminal 2:
freqtrade trade -c MyConfigUSDT.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesUSDT.dryrun.sqlite
```

Conversely, if you wish to do the same thing in production mode, you will also have to create at least one new database (in addition to the default one) and specify the path to the "live" databases, for example:

``` bash
# Terminal 1:
freqtrade trade -c MyConfigBTC.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesBTC.live.sqlite
# Terminal 2:
freqtrade trade -c MyConfigUSDT.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesUSDT.live.sqlite
```

For more information regarding usage of the sqlite databases, for example to manually enter or remove trades, please refer to the [SQL Cheatsheet](sql_cheatsheet.md).

### Multiple instances using docker

To run multiple instances of freqtrade using docker you will need to edit the docker-compose.yml file and add all the instances you want as separate services. Remember, you can separate your configuration into multiple files, so it's a good idea to think about making them modular, then if you need to edit something common to all bots, you can do that in a single config file. 
``` yml
---
version: '3'
services:
  freqtrade1:
    image: freqtradeorg/freqtrade:stable
    # image: freqtradeorg/freqtrade:develop
    # Use plotting image
    # image: freqtradeorg/freqtrade:develop_plot
    # Build step - only needed when additional dependencies are needed
    # build:
    #   context: .
    #   dockerfile: "./docker/Dockerfile.custom"
    restart: always
    container_name: freqtrade1
    volumes:
      - "./user_data:/freqtrade/user_data"
    # Expose api on port 8080 (localhost only)
    # Please read the https://www.freqtrade.io/en/latest/rest-api/ documentation
    # before enabling this.
     ports:
     - "127.0.0.1:8080:8080"
    # Default command used when running `docker compose up`
    command: >
      trade
      --logfile /freqtrade/user_data/logs/freqtrade1.log
      --db-url sqlite:////freqtrade/user_data/tradesv3_freqtrade1.sqlite
      --config /freqtrade/user_data/config.json
      --config /freqtrade/user_data/config.freqtrade1.json
      --strategy SampleStrategy
  
  freqtrade2:
    image: freqtradeorg/freqtrade:stable
    # image: freqtradeorg/freqtrade:develop
    # Use plotting image
    # image: freqtradeorg/freqtrade:develop_plot
    # Build step - only needed when additional dependencies are needed
    # build:
    #   context: .
    #   dockerfile: "./docker/Dockerfile.custom"
    restart: always
    container_name: freqtrade2
    volumes:
      - "./user_data:/freqtrade/user_data"
    # Expose api on port 8080 (localhost only)
    # Please read the https://www.freqtrade.io/en/latest/rest-api/ documentation
    # before enabling this.
    ports:
      - "127.0.0.1:8081:8080"
    # Default command used when running `docker compose up`
    command: >
      trade
      --logfile /freqtrade/user_data/logs/freqtrade2.log
      --db-url sqlite:////freqtrade/user_data/tradesv3_freqtrade2.sqlite
      --config /freqtrade/user_data/config.json
      --config /freqtrade/user_data/config.freqtrade2.json
      --strategy SampleStrategy

```

You can use whatever naming convention you want, freqtrade1 and 2 are arbitrary. Note, that you will need to use different database files, port mappings and telegram configurations for each instance, as mentioned above. 

## Use a different database system

Freqtrade is using SQLAlchemy, which supports multiple different database systems. As such, a multitude of database systems should be supported.
Freqtrade does not depend or install any additional database driver. Please refer to the [SQLAlchemy docs](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls) on installation instructions for the respective database systems.

The following systems have been tested and are known to work with freqtrade:

* sqlite (default)
* PostgreSQL
* MariaDB

!!! Warning
    By using one of the below database systems, you acknowledge that you know how to manage such a system. The freqtrade team will not provide any support with setup or maintenance (or backups) of the below database systems.

### PostgreSQL

Installation:
`pip install "psycopg[binary]"`

Usage:
`... --db-url postgresql+psycopg://<username>:<password>@localhost:5432/<database>`

Freqtrade will automatically create the tables necessary upon startup.

If you're running different instances of Freqtrade, you must either setup one database per Instance or use different users / schemas for your connections.

### MariaDB / MySQL

Freqtrade supports MariaDB by using SQLAlchemy, which supports multiple different database systems.

Installation:
`pip install pymysql`

Usage:
`... --db-url mysql+pymysql://<username>:<password>@localhost:3306/<database>`



## Configure the bot running as a systemd service

Copy the `freqtrade.service` file to your systemd user directory (usually `~/.config/systemd/user`) and update `WorkingDirectory` and `ExecStart` to match your setup.

!!! Note
    Certain systems (like Raspbian) don't load service unit files from the user directory. In this case, copy `freqtrade.service` into `/etc/systemd/user/` (requires superuser permissions).

After that you can start the daemon with:

```bash
systemctl --user start freqtrade
```

For this to be persistent (run when user is logged out) you'll need to enable `linger` for your freqtrade user.

```bash
sudo loginctl enable-linger "$USER"
```

If you run the bot as a service, you can use systemd service manager as a software watchdog monitoring freqtrade bot 
state and restarting it in the case of failures. If the `internals.sd_notify` parameter is set to true in the 
configuration or the `--sd-notify` command line option is used, the bot will send keep-alive ping messages to systemd 
using the sd_notify (systemd notifications) protocol and will also tell systemd its current state (Running, Paused or Stopped) 
when it changes. 

The `freqtrade.service.watchdog` file contains an example of the service unit configuration file which uses systemd 
as the watchdog.

!!! Note
    The sd_notify communication between the bot and the systemd service manager will not work if the bot runs in a Docker container.

## Advanced Logging

Freqtrade uses the default logging module provided by python.
Python allows for extensive [logging configuration](https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig) in this regard - way more than what can be covered here.

Default logging format (coloured terminal output) is set up by default if no `log_config` is provided in your freqtrade configuration.
Using `--logfile logfile.log` will enable the RotatingFileHandler.

If you're not content with the log format, or with the default settings provided for the RotatingFileHandler, you can customize logging to your liking by adding the `log_config` configuration to your freqtrade configuration file(s).

The default configuration looks roughly like the below, with the file handler being provided but not enabled as the `filename` is commented out.
Uncomment this line and supply a valid path/filename to enable it.

``` json hl_lines="5-7 13-16 27"
{
  "log_config": {
      "version": 1,
      "formatters": {
          "basic": {
              "format": "%(message)s"
          },
          "standard": {
              "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
          }
      },
      "handlers": {
          "console": {
              "class": "freqtrade.loggers.ft_rich_handler.FtRichHandler",
              "formatter": "basic"
          },
          "file": {
              "class": "logging.handlers.RotatingFileHandler",
              "formatter": "standard",
              // "filename": "someRandomLogFile.log",
              "maxBytes": 10485760,
              "backupCount": 10
          }
      },
      "root": {
          "handlers": [
              "console",
              // "file"
          ],
          "level": "INFO",
      }
  }
}
```

!!! Note "highlighted lines"
    Highlighted lines in the above code-block define the Rich handler and belong together.
    The formatter "standard" and "file" will belong to the FileHandler.

Each handler must use one of the defined formatters (by name), its class must be available, and must be a valid logging class.
To actually use a handler, it must be in the "handlers" section inside the "root" segment.
If this section is left out, freqtrade will provide no output (in the non-configured handler, anyway).

!!! Tip "Explicit log configuration"
    We recommend to extract the logging configuration from your main freqtrade configuration file, and provide it to your bot via [multiple configuration files](configuration.md#multiple-configuration-files) functionality. This will avoid unnecessary code duplication.

---

On many Linux systems the bot can be configured to send its log messages to `syslog` or `journald` system services. Logging to a remote `syslog` server is also available on Windows. The special values for the `--logfile` command line option can be used for this.

### Logging to syslog

To send Freqtrade log messages to a local or remote `syslog` service use the `"log_config"` setup option to configure logging.

``` json
{
  // ...
  "log_config": {
    "version": 1,
    "formatters": {
      "syslog_fmt": {
        "format": "%(name)s - %(levelname)s - %(message)s"
      }
    },
    "handlers": {
      // Other handlers? 
      "syslog": {
         "class": "logging.handlers.SysLogHandler",
          "formatter": "syslog_fmt",
          // Use one of the other options above as address instead? 
          "address": "/dev/log"
      }
    },
    "root": {
      "handlers": [
        // other handlers
        "syslog",
        
      ]
    }

  }
}
```

[Additional log-handlers](#advanced-logging) may need to be configured to for example also have log output in the console.

#### Syslog usage

Log messages are send to `syslog` with the `user` facility. So you can see them with the following commands:

* `tail -f /var/log/user`, or
* install a comprehensive graphical viewer (for instance, 'Log File Viewer' for Ubuntu).

On many systems `syslog` (`rsyslog`) fetches data from `journald` (and vice versa), so both syslog or journald can be used and the messages be viewed with both `journalctl` and a syslog viewer utility. You can combine this in any way which suites you better.

For `rsyslog` the messages from the bot can be redirected into a separate dedicated log file. To achieve this, add

```
if $programname startswith "freqtrade" then -/var/log/freqtrade.log
```

to one of the rsyslog configuration files, for example at the end of the `/etc/rsyslog.d/50-default.conf`.

For `syslog` (`rsyslog`), the reduction mode can be switched on. This will reduce the number of repeating messages. For instance, multiple bot Heartbeat messages will be reduced to a single message when nothing else happens with the bot. To achieve this, set in `/etc/rsyslog.conf`:

```
# Filter duplicated messages
$RepeatedMsgReduction on
```

#### Syslog addressing

The syslog address can be either a Unix domain socket (socket filename) or a UDP socket specification, consisting of IP address and UDP port, separated by the `:` character.

So, the following are the examples of possible addresses:

* `"address": "/dev/log"` -- log to syslog (rsyslog) using the `/dev/log` socket, suitable for most systems.
* `"address": "/var/run/syslog"` -- log to syslog (rsyslog) using the `/var/run/syslog` socket. Use this on MacOS.
* `"address": "localhost:514"` -- log to local syslog using UDP socket, if it listens on port 514.
* `"address": "<ip>:514"` -- log to remote syslog at IP address and port 514. This may be used on Windows for remote logging to an external syslog server.

??? Info "Deprecated - configure syslog via command line"
    `--logfile syslog:<syslog_address>` -- send log messages to `syslog` service using the `<syslog_address>` as the syslog address.

    The syslog address can be either a Unix domain socket (socket filename) or a UDP socket specification, consisting of IP address and UDP port, separated by the `:` character.

    So, the following are the examples of possible usages:

    * `--logfile syslog:/dev/log` -- log to syslog (rsyslog) using the `/dev/log` socket, suitable for most systems.
    * `--logfile syslog` -- same as above, the shortcut for `/dev/log`.
    * `--logfile syslog:/var/run/syslog` -- log to syslog (rsyslog) using the `/var/run/syslog` socket. Use this on MacOS.
    * `--logfile syslog:localhost:514` -- log to local syslog using UDP socket, if it listens on port 514.
    * `--logfile syslog:<ip>:514` -- log to remote syslog at IP address and port 514. This may be used on Windows for remote logging to an external syslog server.

### Logging to journald

This needs the `cysystemd` python package installed as dependency (`pip install cysystemd`), which is not available on Windows. Hence, the whole journald logging functionality is not available for a bot running on Windows.

To send Freqtrade log messages to `journald` system service, add the following configuration snippet to your configuration.

``` json
{
  // ...
  "log_config": {
    "version": 1,
    "formatters": {
      "journald_fmt": {
        "format": "%(name)s - %(levelname)s - %(message)s"
      }
    },
    "handlers": {
      // Other handlers? 
      "journald": {
         "class": "cysystemd.journal.JournaldLogHandler",
          "formatter": "journald_fmt",
      }
    },
    "root": {
      "handlers": [
        // .. 
        "journald",
        
      ]
    }

  }
}
```

[Additional log-handlers](#advanced-logging) may need to be configured to for example also have log output in the console.

Log messages are send to `journald` with the `user` facility. So you can see them with the following commands:

* `journalctl -f` -- shows Freqtrade log messages sent to `journald` along with other log messages fetched by `journald`.
* `journalctl -f -u freqtrade.service` -- this command can be used when the bot is run as a `systemd` service.

There are many other options in the `journalctl` utility to filter the messages, see manual pages for this utility.

On many systems `syslog` (`rsyslog`) fetches data from `journald` (and vice versa), so both `--logfile syslog` or `--logfile journald` can be used and the messages be viewed with both `journalctl` and a syslog viewer utility. You can combine this in any way which suites you better.

??? Info "Deprecated - configure journald via command line"
    To send Freqtrade log messages to `journald` system service use the `--logfile` command line option with the value in the following format:

    `--logfile journald` -- send log messages to `journald`.

### Log format as JSON

You can also configure the default output stream to use JSON format instead.
The "fmt_dict" attribute defines the keys for the json output - as well as the [python logging LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes).

The below configuration will change the default output to JSON. The same formatter could however also be used in combination with the `RotatingFileHandler`.
We recommend to keep one format in human readable form.

``` json
{
  // ...
  "log_config": {
    "version": 1,
    "formatters": {
       "json": {
          "()": "freqtrade.loggers.json_formatter.JsonFormatter",
          "fmt_dict": {
              "timestamp": "asctime",
              "level": "levelname",
              "logger": "name",
              "message": "message"
          }
      }
    },
    "handlers": {
      // Other handlers? 
      "jsonStream": {
          "class": "logging.StreamHandler",
          "formatter": "json"
      }
    },
    "root": {
      "handlers": [
        // .. 
        "jsonStream",
        
      ]
    }

  }
}
```
