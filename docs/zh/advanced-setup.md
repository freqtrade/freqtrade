<!-- 本文件为中文翻译版，由 AI 根据 docs/advanced-setup.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，图片与 includes 引用使用 ../ 指向英文原文档资源。 -->

# 安装后的进阶任务

本页讲解一些在机器人安装之后可以执行的进阶任务和配置选项，在某些环境中可能会有用。

如果你不知道这里提到的内容是什么意思，你可能并不需要它们。

## 运行多个 Freqtrade 实例

本节将向你展示如何在同一台机器上同时运行多个机器人。

### 需要考虑的事项

* 使用不同的数据库文件。
* 使用不同的 Telegram 机器人（需要多个不同的配置文件；仅当启用了 Telegram 时适用）。
* 使用不同的端口（仅当启用了 Freqtrade REST API webserver 时适用）。

### 不同的数据库文件

为了跟踪你的交易、利润等信息，freqtrade 使用一个 SQLite 数据库来存储各类信息，例如你过去进行的交易，以及你随时持有的当前仓位。这让你能够跟踪自己的利润，但最重要的是，在机器人进程被重启或意外终止时，能够跟踪正在进行的活动。

默认情况下，freqtrade 会为 dry-run（模拟运行）和 live（实盘）机器人使用不同的数据库文件（这假设配置和命令行参数中都没有指定 database-url）。对于实盘交易模式，默认数据库为 `tradesv3.sqlite`；对于 dry-run，默认数据库为 `tradesv3.dryrun.sqlite`。

trade 命令中用于指定这些文件路径的可选参数是 `--db-url`，它需要一个有效的 SQLAlchemy url。因此，当你仅使用配置和策略参数以 dry-run 模式启动一个机器人时，以下两条命令效果相同。

``` bash
freqtrade trade -c MyConfig.json -s MyStrategy
# 等价于
freqtrade trade -c MyConfig.json -s MyStrategy --db-url sqlite:///tradesv3.dryrun.sqlite
```

这意味着，如果你在两个不同的终端中运行 trade 命令，例如分别以 USDT 和 BTC 来测试你的策略，你将不得不使用不同的数据库来运行它们。

如果你指定的数据库 URL 不存在，freqtrade 将创建一个以你指定名称命名的数据库。因此，要用 BTC 和 USDT 两种计价货币来测试你的自定义策略，你可以使用以下命令（在 2 个独立终端中）：

``` bash
# 终端 1：
freqtrade trade -c MyConfigBTC.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesBTC.dryrun.sqlite
# 终端 2：
freqtrade trade -c MyConfigUSDT.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesUSDT.dryrun.sqlite
```

相反，如果你要在生产模式下做同样的事情，你还需要至少再创建一个新数据库（除了默认的之外），并指定指向"live"数据库的路径，例如：

``` bash
# 终端 1：
freqtrade trade -c MyConfigBTC.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesBTC.live.sqlite
# 终端 2：
freqtrade trade -c MyConfigUSDT.json -s MyCustomStrategy --db-url sqlite:///user_data/tradesUSDT.live.sqlite
```

有关使用 sqlite 数据库的更多信息，例如如何手动录入或删除交易，请参阅 [SQL Cheatsheet](sql_cheatsheet.md)。

### 使用 docker 运行多个实例

要使用 docker 运行多个 freqtrade 实例，你需要编辑 docker-compose.yml 文件，并将你想要的所有实例作为独立的服务（service）添加进去。请记住，你可以将配置拆分为多个文件，因此最好考虑让它们模块化，这样如果你需要编辑所有机器人共用的某项内容，就可以在一个单独的配置文件中完成。

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
    # Please read the https://www.freqtrade.io/en/stable/rest-api/ documentation
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
    # Please read the https://www.freqtrade.io/en/stable/rest-api/ documentation
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

你可以使用任何你想要的命名约定，freqtrade1 和 2 只是任意名称。请注意，如上所述，你需要为每个实例使用不同的数据库文件、端口映射和 Telegram 配置。

## 使用不同的数据库系统

Freqtrade 使用的是 SQLAlchemy，它支持多种不同的数据库系统。因此，应该支持众多数据库系统。Freqtrade 并不依赖或安装任何额外的数据库驱动。有关各数据库系统的安装说明，请参阅 [SQLAlchemy 文档](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls)。

以下系统已经过测试，已知可与 freqtrade 配合使用：

* sqlite（默认）
* PostgreSQL
* MariaDB

!!! Warning
    通过使用以下任一数据库系统，即表示你确认自己知道如何管理此类系统。freqtrade 团队不会为以下数据库系统的设置或维护（或备份）提供任何支持。

### PostgreSQL

安装：
`pip install "psycopg[binary]"`

使用：
`... --db-url postgresql+psycopg://<username>:<password>@localhost:5432/<database>`

Freqtrade 会在启动时自动创建所需的表。

如果你在运行不同的 freqtrade 实例，必须要么为每个实例设置一个独立的数据库，要么为你的连接使用不同的用户/模式（schema）。

### MariaDB / MySQL

Freqtrade 通过 SQLAlchemy 支持 MariaDB，而 SQLAlchemy 支持多种不同的数据库系统。

安装：
`pip install pymysql`

使用：
`... --db-url mysql+pymysql://<username>:<password>@localhost:3306/<database>`

## 将机器人配置为 systemd 服务运行

将 `freqtrade.service` 文件复制到你的 systemd 用户目录（通常是 `~/.config/systemd/user`），并更新 `WorkingDirectory` 和 `ExecStart` 以匹配你的设置。

!!! Note
    某些系统（如 Raspbian）不会从用户目录加载服务单元文件。在这种情况下，请将 `freqtrade.service` 复制到 `/etc/systemd/user/`（需要超级用户权限）。

之后，你可以用以下命令启动守护进程：

```bash
systemctl --user start freqtrade
```

为了让它在用户注销后也能持续运行（即开机/登出后自启），你需要为你的 freqtrade 用户启用 `linger`。

```bash
sudo loginctl enable-linger "$USER"
```

如果你将机器人作为服务运行，可以使用 systemd 服务管理器作为软件看门狗（watchdog）来监控 freqtrade 机器人的状态，并在发生故障时将其重启。如果在配置中将 `internals.sd_notify` 参数设为 true，或者使用了 `--sd-notify` 命令行选项，机器人就会使用 sd_notify（systemd 通知）协议向 systemd 发送保活（keep-alive）心跳消息，并且还会在状态变化（运行中 Running、已暂停 Paused 或已停止 Stopped）时通知 systemd。

`freqtrade.service.watchdog` 文件里包含一个使用 systemd 作为看门狗的服务单元配置示例。

!!! Note
    如果机器人运行在 Docker 容器中，机器人与 systemd 服务管理器之间的 sd_notify 通信将无法工作。

## 进阶日志配置

Freqtrade 使用 Python 提供的默认 logging 模块。Python 在这方面允许进行广泛的 [日志配置](https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig)——其程度远超这里所能涵盖的范围。

如果在你的 freqtrade 配置中没有提供 `log_config`，默认日志格式（带颜色的终端输出）会被默认设置好。使用 `--logfile logfile.log` 会启用 RotatingFileHandler（滚动文件处理器）。

如果你对日志格式或 RotatingFileHandler 的默认设置不满意，可以通过将 `log_config` 配置添加到你的 freqtrade 配置文件（一个或多个）中，按自己的喜好自定义日志。

默认配置大致如下，其中文件处理器已提供但并未启用（因为 `filename` 被注释掉了）。取消该行注释并提供一个有效的路径/文件名即可启用它。

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
    以上代码块中高亮的行定义了 Rich handler，并且是配套使用的。
    formatter "standard" 和 "file" 将属于 FileHandler。

每个 handler 必须使用一个已定义的 formatter（按名称），它的类必须可用，并且必须是一个有效的 logging 类。要真正使用一个 handler，它必须位于 "root" 段的 "handlers" 部分中。如果遗漏这一部分，freqtrade 将不会产生任何输出（至少在被配置的 handler 那里是如此）。

!!! Tip "显式的日志配置"
    我们建议将日志配置从你的主 freqtrade 配置文件中抽取出来，并通过 [多配置文件](configuration.md#multiple-configuration-files) 功能提供给你的机器人。这将避免不必要的代码重复。

---

在许多 Linux 系统上，机器人可以被配置为将它的日志消息发送到 `syslog` 或 `journald` 系统服务。向远程 `syslog` 服务器记录日志在 Windows 上同样可用。为此，可以使用 `--logfile` 命令行选项的特殊取值。

### 记录日志到 syslog

要向本地或远程的 `syslog` 服务发送 Freqtrade 日志消息，请使用 `"log_config"` 配置选项来配置日志。

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

[Additional log-handlers](#advanced-logging) 可能还需要进行配置，例如，以便同时在控制台也有日志输出。

#### Syslog 用法

日志消息以 `user` facility 发送到 `syslog`。因此你可以用以下命令查看它们：

* `tail -f /var/log/user`，或者
* 安装一个综合性的图形化查看器（例如，Ubuntu 的 'Log File Viewer'）。

在许多系统上，`syslog`（`rsyslog`）会从 `journald` 获取数据（反之亦然），因此 syslog 和 journald 都可以使用，并且消息既可以用 `journalctl` 也可以用 syslog 查看工具来查看。你可以以任何更适合你的方式组合使用。

对于 `rsyslog`，来自机器人的消息可以被重定向到一个独立的专用日志文件。为此，请添加

```
if $programname startswith "freqtrade" then -/var/log/freqtrade.log
```

到某个 rsyslog 配置文件中，例如添加到 `/etc/rsyslog.d/50-default.conf` 的末尾。

对于 `syslog`（`rsyslog`），可以开启精简（reduction）模式。这将减少重复消息的数量。例如，在机器人没有其他动静时，多条心跳（Heartbeat）消息会被精简为单条消息。为此，请在 `/etc/rsyslog.conf` 中设置：

```
# Filter duplicated messages
$RepeatedMsgReduction on
```

#### Syslog 寻址

syslog 地址可以是一个 Unix 域套接字（socket 文件名），也可以是一个 UDP 套接字规范，由 IP 地址和 UDP 端口组成，以 `:` 字符分隔。

因此，以下是一些可能地址的示例：

* `"address": "/dev/log"` -- 使用 `/dev/log` 套接字记录到 syslog（rsyslog），适用于大多数系统。
* `"address": "/var/run/syslog"` -- 使用 `/var/run/syslog` 套接字记录到 syslog（rsyslog）。在 MacOS 上使用此项。
* `"address": "localhost:514"` -- 如果 syslog 监听在 514 端口，则使用 UDP 套接字记录到本地 syslog。
* `"address": "<ip>:514"` -- 记录到 IP 地址和端口 514 的远程 syslog。在 Windows 上可用于向外部 syslog 服务器进行远程日志记录。

??? Info "已弃用 - 通过命令行配置 syslog"
    `--logfile syslog:<syslog_address>` -- 使用 `<syslog_address>` 作为 syslog 地址，将日志消息发送到 `syslog` 服务。

    syslog 地址可以是一个 Unix 域套接字（socket 文件名），也可以是一个 UDP 套接字规范，由 IP 地址和 UDP 端口组成，以 `:` 字符分隔。

    因此，以下是一些可能用法的示例：

    * `--logfile syslog:/dev/log` -- 使用 `/dev/log` 套接字记录到 syslog（rsyslog），适用于大多数系统。
    * `--logfile syslog` -- 同上，是 `/dev/log` 的简写。
    * `--logfile syslog:/var/run/syslog` -- 使用 `/var/run/syslog` 套接字记录到 syslog（rsyslog）。在 MacOS 上使用此项。
    * `--logfile syslog:localhost:514` -- 如果 syslog 监听在 514 端口，则使用 UDP 套接字记录到本地 syslog。
    * `--logfile syslog:<ip>:514` -- 记录到 IP 地址和端口 514 的远程 syslog。在 Windows 上可用于向外部 syslog 服务器进行远程日志记录。

### 记录日志到 journald

这需要安装 `cysystemd` Python 包作为依赖（`pip install cysystemd`），而它在 Windows 上不可用。因此，整个 journald 日志功能对运行在 Windows 上的机器人不可用。

要向 `journald` 系统服务发送 Freqtrade 日志消息，请将以下配置片段添加到你的配置中。

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

[Additional log-handlers](#advanced-logging) 可能还需要进行配置，例如，以便同时在控制台也有日志输出。

日志消息以 `user` facility 发送到 `journald`。因此你可以用以下命令查看它们：

* `journalctl -f` -- 显示发送到 `journald` 的 Freqtrade 日志消息，以及 `journald` 获取的其他日志消息。
* `journalctl -f -u freqtrade.service` -- 当机器人作为 `systemd` 服务运行时，可以使用此命令。

`journalctl` 工具中还有许多其他选项可用于过滤消息，请参阅该工具的帮助手册页。

在许多系统上，`syslog`（`rsyslog`）会从 `journald` 获取数据（反之亦然），因此 `--logfile syslog` 或 `--logfile journald` 都可以使用，消息既可以用 `journalctl` 也可以用 syslog 查看工具来查看。你可以以任何更适合你的方式组合使用。

??? Info "已弃用 - 通过命令行配置 journald"
    要向 `journald` 系统服务发送 Freqtrade 日志消息，请使用 `--logfile` 命令行选项，取值格式如下：

    `--logfile journald` -- 将日志消息发送到 `journald`。

### 以 JSON 格式记录日志

你也可以将默认输出流配置为使用 JSON 格式。

"fmt_dict" 属性定义了 json 输出的键——以及 [python logging LogRecord 属性](https://docs.python.org/3/library/logging.html#logrecord-attributes)。

以下配置会将默认输出更改为 JSON。不过，同样的 formatter 也可以与 `RotatingFileHandler` 组合使用。我们建议保留一种人类可读的格式。

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
