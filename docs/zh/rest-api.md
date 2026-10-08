# REST API

## FreqUI

FreqUI 现在有了自己专门的[文档章节](freq-ui.md) - 有关 FreqUI 的所有信息，请参考该章节。

## 配置

通过将 api_server 部分添加到你的配置中并将 `api_server.enabled` 设置为 `true` 来启用 rest API。

示例配置：

``` json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "error",
        "enable_openapi": false,
        "jwt_secret_key": "somethingRandomSomethingRandom123",
        "CORS_origins": [],
        "username": "Freqtrader",
        "password": "SuperSecret1!",
        "ws_token": "sercet_Ws_t0ken"
    },
```

!!! Danger "安全警告"
    默认情况下，配置仅监听 localhost（因此无法从其他系统访问）。我们强烈建议不要将此 API 暴露给互联网，并选择一个强且唯一的密码，因为其他人可能会潜在地控制你的机器人。

??? Note "远程服务器上的 API/UI 访问"
    如果你在 VPS 上运行，你应该考虑使用 ssh 隧道，或者设置 VPN（openVPN、wireguard）来连接到你的机器人。
    这将确保 freqUI 不会直接暴露给互联网，出于安全原因这是不被推荐的（freqUI 本身不支持 https）。
    这些工具的设置不在本教程范围内，但可以在互联网上找到许多优秀的教程。

然后你可以通过在浏览器中访问 `http://127.0.0.1:8080/api/v1/ping` 来检查 API 是否正常运行。
这应该返回响应：

``` output
{"status":"pong"}
```

所有其他端点都返回敏感信息，并且需要身份验证，因此无法通过 Web 浏览器访问。

### 安全

要生成一个安全的密码，最好使用密码管理器，或使用以下代码。

``` python
import secrets
secrets.token_hex()
```

!!! Hint "JWT token"
    使用相同的方法也可以生成 JWT 密钥（`jwt_secret_key`）。

!!! Danger "密码选择"
    请确保选择一个非常强、唯一的密码来保护你的机器人免受未经授权的访问。
    同时，将 `jwt_secret_key` 更改为随机值（不需要记住它，但它将用于加密你的会话，所以它最好是唯一的！）。为了安全，该值也应为 32 个字符或更长。

### 使用 docker 配置

如果你使用 docker 运行机器人，你需要让机器人监听传入的连接。安全性随后由 docker 处理。

``` json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080,
        "username": "Freqtrader",
        "password": "SuperSecret1!",
        //...
    },
```

请确保你的 docker-compose 文件中有以下 2 行可用：

```yml
    ports:
      - "127.0.0.1:8080:8080"
```

!!! Danger "安全警告"
    通过在 docker 端口映射中使用 `"8080:8080"`（或 `"0.0.0.0:8080:8080"`），API 将对连接到服务器的所有人可用（在正确的端口下），因此其他人可能能够控制你的机器人。
    如果你在安全的环境中（如你的家庭网络）运行机器人，这可能**是**安全的，但不建议将 API 暴露给互联网。

## Rest API

### 消费 API

我们建议通过使用受支持的 `freqtrade-client` 包来消费 API（也可作为 `scripts/rest_client.py` 使用）。

可以使用 `pip install freqtrade-client` 独立于任何正在运行的 freqtrade 机器人安装此命令。

此模块被设计为轻量级，仅依赖于 `requests` 和 `python-rapidjson` 模块，跳过了 freqtrade 否则需要的所有的重型依赖。

``` bash
freqtrade-client <command> [optional parameters]
```

默认情况下，该脚本假定使用 `127.0.0.1`（localhost）和端口 `8080`，但你可以指定一个配置文件来覆盖此行为。

#### 极简客户端配置

``` json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080,
        "username": "Freqtrader",
        "password": "SuperSecret1!",
        //...
    }
}
```

``` bash
freqtrade-client --config rest_config.json <command> [optional parameters]
```

带有许多参数的命令可能需要关键字参数（为了清晰起见）——可以按如下方式提供：

``` bash
freqtrade-client --config rest_config.json forceenter BTC/USDT long enter_tag=GutFeeling
```

此方法适用于所有参数——使用 "show" 命令查看可用参数列表。

??? Note "编程使用"
    `freqtrade-client` 包（可独立于 freqtrade 安装）可以在你自己的脚本中使用，以与 freqtrade API 交互。
    为此，请使用以下内容：

    ``` python
    from freqtrade_client import FtRestClient
    

    client = FtRestClient(server_url, username, password)

    # 获取机器人的状态
    ping = client.ping()
    print(ping)

    # 将币对添加到黑名单
    client.blacklist("BTC/USDT", "ETH/USDT")
    # 通过提供列表将币对添加到黑名单
    client.blacklist(*listPairs)
    # ... 
    ```

    有关可用命令的完整列表，请参阅下面的列表。

#### Freqtrade 客户端 - 可用命令

可以使用 `help` 命令从 rest-client 脚本中列出可能的命令。

``` bash
freqtrade-client help
```

--8<-- "commands/freqtrade-client.md"


### 可用端点

如果你想通过其他路由手动调用 REST API，例如直接通过 `curl`，下表的 URL 端点和参数显示了相关信息。
下表中所有端点都需要加上 API 的基础 URL 前缀，例如 `http://127.0.0.1:8080/api/v1/` - 因此命令变为 `http://127.0.0.1:8080/api/v1/<command>`。

|  端点 | 方法 | 描述 / 参数 |
|-----------|--------|--------------------------|
| `/ping` | GET | 测试 API 就绪状态的简单命令 - 无需身份验证。
| `/start` | POST | 启动交易器（trader）。
| `/pause` | POST | 暂停交易器。根据其规则优雅地处理未平仓交易。不进入新仓位。
| `/stop` | POST | 停止交易器。
| `/stopbuy` | POST | 停止交易器开启新交易。根据其规则优雅地关闭未平仓交易。
| `/reload_config` | POST | 重新加载配置文件。
| `/trades` | GET | 列出最近的交易。每次调用限制为 500 笔交易。
| `/trade/<tradeid>` | GET | 获取特定交易。<br/>*参数：*<br/>- `tradeid` (`int`)
| `/trades/<tradeid>` | DELETE | 从数据库中移除交易。尝试关闭未平仓订单。需要在交易所手动处理此交易。<br/>*参数：*<br/>- `tradeid` (`int`)
| `/trades/<tradeid>/open-order` | DELETE | 取消此交易的未平仓订单。<br/>*参数：*<br/>- `tradeid` (`int`)
| `/trades/<tradeid>/reload` | POST | 从交易所重新加载一笔交易。仅在 live 模式下有效，并且可能潜在地帮助恢复在交易所被手动卖出的交易。<br/>*参数：*<br/>- `tradeid` (`int`)
| `/show_config` | GET | 显示当前配置的一部分以及与之相关的设置。
| `/logs` | GET | 显示最后的日志消息。
| `/status` | GET | 列出所有未平仓交易。
| `/count` | GET | 显示已使用和可用的交易数量。
| `/entries` | GET | 显示给定币对每个入场标签（enter tag）的利润统计（如果未给定币对，则为所有币对）。币对是可选的。<br/>*参数：*<br/>- `pair` (`str`)
| `/exits` | GET | 显示给定币对每个出场原因（exit reason）的利润统计（如果未给定币对，则为所有币对）。币对是可选的。<br/>*参数：*<br/>- `pair` (`str`)
| `/mix_tags` | GET | 显示给定币对每个入场标签 + 出场原因组合的的利润统计（如果未给定币对，则为所有币对）。币对是可选的。<br/>*参数：*<br/>- `pair` (`str`)
| `/locks` | GET | 显示当前被锁定的币对。
| `/locks` | POST | 锁定一个币对直到 "until"。（until 将向上取整到最近的时间周期）。Side 是可选的，为 `long` 或 `short`（默认为 `long`）。Reason 是可选的。<br/>*参数：*<br/>- `<pair>` (`str`)<br/>- `<until>` (`datetime`)<br/>- `[side]` (`str`)<br/>- `[reason]` (`str`)
| `/locks/<lockid>` | DELETE | 按 id 删除（禁用）锁定。<br/>*参数：*<br/>- `lockid` (`int`)
| `/profit` | GET | 显示已平仓交易的盈亏摘要以及有关你表现的一些统计数据。
| `/forceexit` | POST | 立即退出给定的交易（忽略 `minimum_roi`），使用给定的订单类型（"market" 或 "limit"，如果未指定则使用你的配置设置），以及选择的金额（如果未指定则为全额卖出）。如果 `tradeid` 提供为 `all`，则所有当前未平仓交易将被强制退出。<br/>*参数：*<br/>- `<tradeid>` (`int` 或 `str`)<br/>- `<ordertype>` (`str`)<br/>- `[amount]` (`float`)
| `/forceenter` | POST | 立即进入给定的币对。Side 是可选的，为 `long` 或 `short`（默认为 `long`）。价格、stake 金额、入场标签和杠杆是可选的。订单类型是可选的，为 `market` 或 `long`（默认使用配置中设置的值）。（`force_entry_enable` 必须设置为 True）<br/>*参数：*<br/>- `<pair>` (`str`)<br/>- `<side>` (`str`)<br/>- `[price]` (`float`)<br/>- `[ordertype]` (`str`)<br/>- `[stakeamount]` (`float`)<br/>- `[entry_tag]` (`str`)<br/>- `[leverage]` (`float`)
| `/performance` | GET | 显示每个已完成的交易按币对分组的性能。
| `/balance` | GET | 显示每种货币的账户余额。
| `/daily` | GET | 显示过去 n 天的每日盈亏（n 默认为 7）。<br/>*参数：*<br/>- `timescale` (`int`)
| `/weekly` | GET | 显示过去 n 天的每周盈亏（n 默认为 4）。<br/>*参数：*<br/>- `timescale` (`int`)
| `/monthly` | GET | 显示过去 n 天的每月盈亏（n 默认为 3）。<br/>*参数：*<br/>- `timescale` (`int`)
| `/stats` | GET | 显示盈亏原因的摘要以及平均持有时间。
| `/whitelist` | GET | 显示当前白名单。
| `/blacklist` | GET | 显示当前黑名单。
| `/blacklist` | POST | 将指定的币对添加到黑名单。<br/>*参数：*<br/>- `blacklist` (`str`)
| `/blacklist` | DELETE | 从黑名单中删除指定的币对列表。<br/>*参数：*<br/>- `[pair,pair]` (`list[str]`)
| `/pair_candles` | GET | 在机器人运行时返回币对 / 时间周期组合的 dataframe。
| `/pair_candles` | POST | 在机器人运行时返回币对 / 时间周期组合的 dataframe，由提供的要返回的列列表过滤。<br/>*参数：*<br/>- `<column_list>` (`list[str]`)
| `/pair_history` | GET | 返回给定 timerange 的已分析 dataframe，由给定策略分析。
| `/pair_history` | POST | 返回给定 timerange 的已分析 dataframe，由给定策略分析，由提供的要返回的列列表过滤。<br/>*参数：*<br/>- `<column_list>` (`list[str]`)
| `/plot_config` | GET | 从策略获取绘图配置（如果未配置则为空）。
| `/strategies` | GET | 列出策略目录中的策略。
| `/strategy/<strategy>` | GET | 按策略类名获取特定策略内容。<br/>*参数：*<br/>- `<strategy>` (`str`)
| `/available_pairs` | GET | 列出可用的回测数据。
| `/version` | GET | 显示版本。
| `/sysinfo` | GET | 显示有关系统负载的信息。
| `/health` | GET | 显示机器人健康状况（最后一次机器人循环）。

!!! Warning "Alpha 状态"
    上面标记为 *Alpha 状态* 或 *Beta 状态* 的端点可能会在没有通知的情况下随时更改。

### 消息 WebSocket

API Server 包含一个 websocket 端点，用于订阅来自 freqtrade 机器人的 RPC 消息。
这可用于消费来自你机器人的实时数据，例如入场/出场成交消息、白名单更改、币对的已填充指标等。

这也用于设置 Freqtrade 中的[生产者/消费者模式](producer-consumer.md)。

假设你的 rest API 设置为 `127.0.0.1` 端口 `8080`，则端点位于 `http://localhost:8080/api/v1/message/ws`。

要访问 websocket 端点，需要 `ws_token` 作为端点 URL 中的查询参数。

要生成一个安全的 `ws_token`，你可以运行以下代码：

``` python
>>> import secrets
>>> secrets.token_urlsafe(25)
'hZ-y58LXyX_HZ8O1cJzVyN6ePWrLpNQv4Q'
```

然后你会将该 token 添加到你的 `api_server` 配置中的 `ws_token` 下。像这样：

``` json
"api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080,
    "verbosity": "error",
    "enable_openapi": false,
    "jwt_secret_key": "somethingRandomSomethingRandom123",
    "CORS_origins": [],
    "username": "Freqtrader",
    "password": "SuperSecret1!",
    "ws_token": "hZ-y58LXyX_HZ8O1cJzVyN6ePWrLpNQv4Q" // <-----
},
```

现在你可以连接到端点 `http://localhost:8080/api/v1/message/ws?token=hZ-y58LXyX_HZ8O1cJzVyN6ePWrLpNQv4Q`。

!!! Danger "重复使用示例 token"
    请不要使用上面的示例 token。为了确保你的安全，请生成一个全新的 token。

#### 使用 WebSocket

一旦连接到 WebSocket，机器人将向任何已订阅它们的人广播 RPC 消息。要订阅消息列表，你必须通过 WebSocket 发送一个 JSON 请求，如下所示。 `data` 键必须是消息类型字符串的列表。

``` json
{
  "type": "subscribe",
  "data": ["whitelist", "analyzed_df"] // 字符串消息类型的列表
}
```

有关消息类型的列表，请参阅 `freqtrade/enums/rpcmessagetype.py` 中的 RPCMessageType 枚举。

现在，只要这些类型的 RPC 消息在机器人中发送，只要连接处于活动状态，你就会通过 WebSocket 收到它们。它们通常采用与请求相同的形式：

``` json
{
  "type": "analyzed_df",
  "data": {
      "key": ["NEO/BTC", "5m", "spot"],
      "df": {}, // dataframe
      "la": "2022-09-08 22:14:41.457786+00:00"
  }
}
```

#### 反向代理设置

使用 [Nginx](https://nginx.org/en/docs/) 时，要使 WebSockets 工作，需要以下配置（请注意此配置不完整，它缺少一些信息，不能按原样使用）。

请确保将 `<freqtrade_listen_ip>`（以及随后的端口）替换为与你的配置/设置匹配的 IP 和端口。

```
http {
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    #...

    server {
        #...

        location / {
            proxy_http_version 1.1;
            proxy_pass http://<freqtrade_listen_ip>:8080;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
        }
    }
}
```

要正确配置你的反向代理（安全地），请查阅其代理 websockets 的文档。

- **Traefik**：Traefik 开箱即用地支持 websockets，请参阅[文档](https://doc.traefik.io/traefik/)
- **Caddy**：Caddy v2 开箱即用地支持 websockets，请参阅[文档](https://caddyserver.com/docs/v2-upgrade#proxy)

!!! Tip "SSL 证书"
    你可以使用 certbot 等工具来设置 ssl 证书，以便通过加密连接使用上述任何反向代理访问你的机器人的 UI。
    虽然这将保护传输中的数据，但我们不建议在私有网络（VPN、SSH 隧道）之外运行 freqtrade API。

### OpenAPI 接口

要启用内置的 openAPI 接口（Swagger UI），请在 api_server 配置中指定 `"enable_openapi": true`。
这将启用 `/docs` 端点处的 Swagger UI。默认情况下，它运行在 <http://localhost:8080/docs> - 但这取决于你的设置。

### 使用 JWT token 的高级 API 用法

!!! Note
    以下内容应该在应用程序中完成（一个通过 API 获取信息的 Freqtrade REST API 客户端），并不打算定期使用。

Freqtrade 的 REST API 还提供 JWT（JSON Web Tokens）。
你可以使用以下命令登录，随后使用生成的 access_token。

``` bash
> curl -X POST --user Freqtrader http://localhost:8080/api/v1/token/login
{"access_token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk2ODEsIm5iZiI6MTU4OTExOTY4MSwianRpIjoiMmEwYmY0NWUtMjhmOS00YTUzLTlmNzItMmM5ZWVlYThkNzc2IiwiZXhwIjoxNTg5MTIwNTgxLCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.qt6MAXYIa-l556OM7arBvYJ0SDI9J8bIk3_glDujF5g","refresh_token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk2ODEsIm5iZiI6MTU4OTExOTY4MSwianRpIjoiZWQ1ZWI3YjAtYjMwMy00YzAyLTg2N2MtNWViMjIxNWQ2YTMxIiwiZXhwIjoxNTkxNzExNjgxLCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJ0eXBlIjoicmVmcmVzaCJ9.d1AT_jYICyTAjD0fiQAr52rkRqtxCjUGEMwlNuuzgNQ"}

> access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk2ODEsIm5iZiI6MTU4OTExOTY4MSwianRpIjoiMmEwYmY0NWUtMjhmOS00YTUzLTlmNzItMmM5ZWVlYThkNzc2IiwiZXhwIjoxNTg5MTIwNTgxLCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.qt6MAXYIa-l556OM7arBvYJ0SDI9J8bIk3_glDujF5g"
# 使用 access_token 进行身份验证
> curl -X GET --header "Authorization: Bearer ${access_token}" http://localhost:8080/api/v1/count

```

由于访问 token 的超时时间很短（15 分钟）——应该定期使用 `token/refresh` 请求来获取新的访问 token：

``` bash
> curl -X POST --header "Authorization: Bearer ${refresh_token}"http://localhost:8080/api/v1/token/refresh
{"access_token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk5NzQsIm5iZiI6MTU4OTExOTk3NCwianRpIjoiMDBjNTlhMWUtMjBmYS00ZTk0LTliZjAtNWQwNTg2MTdiZDIyIiwiZXhwIjoxNTg5MTIwODc0LCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.1seHlII3WprjjclY6DpRhen0rqdF4j6jbvxIhUFaSbs"}
```

--8<-- "includes/cors.md"
