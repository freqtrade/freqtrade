<!-- 本文件为中文翻译版，由 AI 根据 docs/sql_cheatsheet.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# SQL 辅助指南

本页面提供了一些关于查询 sqlite 数据库的帮助信息。

!!! Tip "其他数据库系统"
    要使用其他数据库系统（如 PostgreSQL 或 MariaDB），可以使用相同的查询语句，但需要使用对应数据库系统的客户端。[点击此处](advanced-setup.md#use-a-different-database-system)了解如何在 freqtrade 中设置不同的数据库系统。

!!! Warning
    如果您不熟悉 SQL，在数据库上执行查询时应格外小心。
    在执行任何查询之前，请务必备份数据库。

## 安装 sqlite3

sqlite3 是一个基于终端的 sqlite 应用程序。
如果您更喜欢图形界面，也可以使用可视化的数据库编辑器，如 SqliteBrowser。

### Ubuntu/Debian 安装

```bash
sudo apt-get install sqlite3
```

### 通过 docker 使用 sqlite3

freqtrade 的 docker 镜像中已包含 sqlite3，因此您无需在宿主系统上安装任何软件即可编辑数据库。

``` bash
docker compose exec freqtrade /bin/bash
sqlite3 <database-file>.sqlite
```

## 打开数据库

```bash
sqlite3
.open <filepath>
```

## 表结构

### 列出所有表

```bash
.tables
```

### 显示表结构

```bash
.schema <table_name>
```

### 获取表中所有交易记录

```sql
SELECT * FROM trades;
```

## 破坏性查询

写入数据库的查询操作。
这些查询通常不应是必需的，因为 freqtrade 会尝试自行处理所有数据库操作——或通过 API 及 Telegram 命令来暴露这些功能。

!!! Warning
    在执行以下任何查询之前，请确保已备份数据库。

!!! Danger
    同时，当机器人连接到数据库时，您**绝不**应执行任何写入查询（`update`、`insert`、`delete`）。
    这会导致数据损坏——且很可能无法恢复。

### 修复在交易所手动平仓后交易仍处于开放状态的问题

!!! Warning
    在交易所手动卖出某个交易对不会被机器人检测到，机器人仍会尝试卖出。只要有可能，应使用 /forceexit <tradeid> 来完成同样的操作。
    强烈建议在进行任何手动更改之前备份数据库文件。

!!! Note
    使用 /forceexit 后通常无需执行此操作，因为 force_exit 订单会在机器人下一次迭代时自动关闭。

```sql
UPDATE trades
SET is_open=0,
  close_date=<close_date>,
  close_rate=<close_rate>,
  close_profit = close_rate / open_rate - 1,
  close_profit_abs = (amount * <close_rate> * (1 - fee_close) - (amount * (open_rate * (1 - fee_open)))),
  exit_reason=<exit_reason>
WHERE id=<trade_ID_to_update>;
```

#### 示例

```sql
UPDATE trades
SET is_open=0,
  close_date='2020-06-20 03:08:45.103418',
  close_rate=0.19638016,
  close_profit=0.0496,
  close_profit_abs = (amount * 0.19638016 * (1 - fee_close) - (amount * (open_rate * (1 - fee_open)))),
  exit_reason='force_exit'  
WHERE id=31;
```

### 从数据库中删除交易记录

!!! Tip "使用 RPC 方法删除交易"
    建议通过 Telegram 或 REST API 使用 `/delete <tradeid>`。这是删除交易的推荐方式，因为它还会同时删除对应的订单和自定义数据，并触发机器人中必要的事件以保持一切同步。

如果您仍想直接从数据库中删除交易记录，可以使用以下查询。

!!! Danger
    某些系统（Ubuntu）在其 sqlite3 打包版本中禁用了外键约束。使用 sqlite 时——请确保在执行上述查询前通过运行 `PRAGMA foreign_keys = ON` 来启用外键约束。

```sql
DELETE FROM trades WHERE id = <tradeid>;
DELETE FROM orders WHERE ft_trade_id = <tradeid>;
DELETE FROM trade_custom_data WHERE ft_trade_id = <tradeid>;


DELETE FROM trades WHERE id = 31;
DELETE FROM orders WHERE ft_trade_id = 31;
DELETE FROM trade_custom_data WHERE ft_trade_id = 31;
```

!!! Warning
    这将从数据库中删除指定的交易记录。请确保您获取了正确的 ID，并且**绝不**在没有 `where` 子句的情况下执行此查询。