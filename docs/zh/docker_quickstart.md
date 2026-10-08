<!-- 本文件为中文翻译版，由 AI 根据 docs/docker_quickstart.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 使用 Docker 运行 Freqtrade

本页面介绍如何使用 Docker 运行交易机器人。本文档并非开箱即用的指南，你仍然需要阅读相关文档并理解如何正确配置。

## 安装 Docker

首先根据你的操作系统平台下载并安装 Docker / Docker Desktop：

* [Mac](https://docs.docker.com/docker-for-mac/install/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)
* [Linux](https://docs.docker.com/install/)

!!! Info "Docker compose 安装说明"
    Freqtrade 文档假设你使用的是 Docker Desktop（或 docker compose 插件）。  
    虽然 docker-compose 独立安装方式仍然可用，但需要将所有 `docker compose` 命令中的 `docker compose` 改为 `docker-compose`（例如 `docker compose up -d` 需改为 `docker-compose up -d`）。

??? Warning "Windows 上的 Docker"
    如果你刚刚在 Windows 系统上安装了 Docker，请务必重启系统，否则可能会遇到与 Docker 容器网络连接相关的无法解释的问题。

## 使用 Docker 运行 Freqtrade

Freqtrade 在 [Dockerhub](https://hub.docker.com/r/freqtradeorg/freqtrade/) 上提供了官方 Docker 镜像，同时也提供了可直接使用的 [docker compose 文件](https://github.com/freqtrade/freqtrade/blob/stable/docker-compose.yml)。

!!! Note
    - 以下内容假设 `docker` 已安装并可供当前登录用户使用。
    - 下面所有命令使用相对目录，必须从包含 `docker-compose.yml` 文件的目录中执行。

### Docker 快速入门

创建一个新目录并将 [docker-compose 文件](https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml) 放入该目录。

``` bash
mkdir ft_userdata
cd ft_userdata/
# 从仓库下载 docker-compose 文件
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml

# 拉取 freqtrade 镜像
docker compose pull

# 创建用户目录结构
docker compose run --rm freqtrade create-userdir --userdir user_data

# 创建配置 - 需要回答交互式问题
docker compose run --rm freqtrade new-config --config user_data/config.json
```

以上命令片段创建了一个名为 `ft_userdata` 的新目录，下载了最新的 compose 文件并拉取了 freqtrade 镜像。
最后两个步骤创建了 `user_data` 目录，并根据你的选择（以交互方式）生成了默认配置。

!!! Question "如何编辑机器人配置？"
    你可以随时编辑配置文件，使用上述配置时，配置文件位于 `user_data/config.json`（在 `ft_userdata` 目录内）。

    你还可以通过编辑 `docker-compose.yml` 文件中的命令部分来更改策略和命令。

#### 添加自定义策略

1. 配置文件现位于 `user_data/config.json`
2. 将自定义策略复制到 `user_data/strategies/` 目录
3. 将策略的类名添加到 `docker-compose.yml` 文件中

默认运行的是 `SampleStrategy`。

!!! Danger "`SampleStrategy` 仅是示例！"
    `SampleStrategy` 仅供参考，为你编写自己的策略提供思路。
    请务必对策略进行回测，并在投入真金白银之前先进行一段时间的模拟交易！
    更多关于策略开发的信息请参阅[策略文档](strategy-customization.md)。

完成以上步骤后，你就可以启动交易模式的机器人了（模拟交易或实盘交易，取决于你之前对相关问题的回答）。

``` bash
docker compose up -d
```

!!! Warning "默认配置"
    虽然生成的配置基本可用，但在启动机器人之前，你仍然需要验证所有选项是否符合你的需求（如价格、交易对列表等）。

#### 访问 UI 界面

如果你在 `new-config` 步骤中选择了启用 FreqUI，可以通过端口 `localhost:8080` 访问 freqUI。

你可以在浏览器中输入 localhost:8080 来访问 UI 界面。

??? Note "远程服务器上的 UI 访问"
    如果你在 VPS 上运行，建议使用 SSH 隧道或设置 VPN（openVPN、wireguard）来连接你的机器人。
    这将确保 freqUI 不会直接暴露在互联网上，出于安全原因不建议这样做（freqUI 默认不支持 HTTPS）。
    这些工具的安装不属于本教程的范围，但你可以在互联网上找到许多优秀的教程。
    另请阅读 [Docker 中的 API 配置](rest-api.md#configuration-with-docker) 章节以了解更多相关信息。

#### 监控机器人

你可以使用 `docker compose ps` 检查正在运行的实例。
该命令应该会将 `freqtrade` 服务列为 `running` 状态。如果不是这种情况，最好检查日志（见下一点）。

#### Docker compose 日志

日志将写入：`user_data/logs/freqtrade.log`。  
你也可以使用命令 `docker compose logs -f` 查看最新日志。

#### 数据库

数据库位于：`user_data/tradesv3.sqlite`

#### 使用 Docker 更新 freqtrade

使用 `docker` 更新 freqtrade 只需运行以下两个命令：

``` bash
# 下载最新镜像
docker compose pull
# 重启镜像
docker compose up -d
```

这将首先拉取最新镜像，然后使用刚拉取的版本重启容器。

!!! Warning "检查更新日志"
    你应始终检查更新日志中的破坏性变更 / 需要手动操作的内容，并确保更新后机器人能正常启动。

### 编辑 docker-compose 文件

高级用户可以进一步编辑 docker-compose 文件，以包含所有可能的选项或参数。

所有 freqtrade 参数都可以通过运行 `docker compose run --rm freqtrade <command> <optional arguments>` 来使用。

!!! Warning "交易命令使用 `docker compose`"
    交易命令（`freqtrade trade <...>`）不应通过 `docker compose run` 运行，而应使用 `docker compose up -d`。
    这可以确保容器正确启动（包括端口转发），并确保容器在系统重启后自动重启。
    如果你打算使用 freqUI，请确保相应调整[配置](rest-api.md#configuration-with-docker)，否则 UI 将不可用。

!!! Note "`docker compose run --rm`"
    包含 `--rm` 会在完成后删除容器，强烈建议在除交易模式（使用 `freqtrade trade` 命令运行）以外的所有模式中使用。

??? Note "不使用 docker compose 的 Docker 用法"
    "`docker compose run --rm`" 需要提供 compose 文件。
    一些不需要身份验证的 freqtrade 命令（如 `list-pairs`）可以改用 "`docker run --rm`" 来运行。  
    例如 `docker run --rm freqtradeorg/freqtrade:stable list-pairs --exchange binance --quote BTC --print-json`。  
    这对于获取交易所信息以添加到你的 `config.json` 中非常有用，不会影响正在运行的容器。

#### 示例：使用 Docker 下载数据

从 Binance 下载 ETH/BTC 交易对 5 天的 1h 时间框架回测数据。数据将存储在主机的 `user_data/data/` 目录中。

``` bash
docker compose run --rm freqtrade download-data --pairs ETH/BTC --exchange binance --days 5 -t 1h
```

更多关于数据下载的详细信息，请参阅[数据下载文档](data-download.md)。

#### 示例：使用 Docker 进行回测

在 Docker 容器中对 SampleStrategy 和指定的历史数据时间范围运行回测，使用 5m 时间框架：

``` bash
docker compose run --rm freqtrade backtesting --config user_data/config.json --strategy SampleStrategy --timerange 20190801-20191001 -i 5m
```

更多详细信息请参阅[回测文档](backtesting.md)。

### Docker 中的附加依赖

如果你的策略需要默认镜像中未包含的依赖项，则需要在主机上构建镜像。
为此，请创建一个包含附加依赖安装步骤的 Dockerfile（参见 [docker/Dockerfile.custom](https://github.com/freqtrade/freqtrade/blob/develop/docker/Dockerfile.custom) 作为示例）。

你还需要修改 `docker-compose.yml` 文件，取消注释构建步骤，并重命名镜像以避免命名冲突。

``` yaml
    image: freqtrade_custom
    build:
      context: .
      dockerfile: "./Dockerfile.<yourextension>"
```

然后你可以运行 `docker compose build --pull` 来构建 Docker 镜像，并使用上述命令运行它。

### 使用 Docker 进行绘图

命令 `freqtrade plot-profit` 和 `freqtrade plot-dataframe`（[文档](plotting.md)）可通过将 `docker-compose.yml` 文件中的镜像更改为 `*_plot` 来使用。
然后你可以按如下方式使用这些命令：

``` bash
docker compose run --rm freqtrade plot-dataframe --strategy AwesomeStrategy -p BTC/ETH --timerange=20180801-20180805
```

输出将存储在 `user_data/plot` 目录中，可以使用任何现代浏览器打开。

### 使用 Docker Compose 进行数据分析

Freqtrade 提供了一个启动 Jupyter Lab 服务器的 docker-compose 文件。
你可以使用以下命令运行该服务器：

``` bash
docker compose -f docker/docker-compose-jupyter.yml up
```

这将创建一个运行 Jupyter Lab 的 Docker 容器，可通过 `https://127.0.0.1:8888/lab` 访问。
请使用启动后在控制台中打印的链接进行简化登录。

由于此镜像的部分内容是在你的机器上构建的，建议定期重新构建镜像以保持 freqtrade（及依赖项）为最新版本。

``` bash
docker compose -f docker/docker-compose-jupyter.yml build --no-cache
```

## 故障排除

### Windows 上的 Docker

* 错误：`"Timestamp for this request is outside of the recvWindow."`  
  市场 API 请求需要同步时钟，但 Docker 容器中的时间会随时间推移逐渐偏移到过去。
  要临时修复此问题，你需要运行 `wsl --shutdown` 然后重新启动 Docker（Windows 10 会弹出提示要求你这样做）。
  永久解决方案是在 Linux 主机上托管 Docker 容器，或使用计划任务定期重启 WSL。

  ``` bash
  taskkill /IM "Docker Desktop.exe" /F
  wsl --shutdown
  start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  ```

* 无法连接到 API（Windows）  
  如果你在 Windows 上并且刚刚安装了 Docker Desktop，请务必重启系统。Docker 在不重启的情况下可能会出现网络连接问题。
  你当然还应该确保[设置](#访问-ui-界面)正确。

!!! Warning
    鉴于上述原因，我们不建议在 Windows 上将 Docker 用于生产环境，仅建议用于实验、数据下载和回测。
    最好使用 Linux VPS 来可靠地运行 freqtrade。