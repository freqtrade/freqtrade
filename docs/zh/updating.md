<!-- 本文件为中文翻译版，由 AI 根据 docs/updating.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 如何更新

要更新您的 freqtrade 安装，请根据您的安装方式使用以下对应方法之一。

!!! Note "跟踪变更"
    重大变更/行为变更将记录在每次发布时附带的更新日志中。
    对于 develop 分支，请关注 PR 以避免对变更感到意外。

## 为什么要更新？

保持您的机器人更新不仅能确保您获得最新的功能和改进，而且是保持机器人平稳运行的必要条件。
Freqtrade 严重依赖底层交易所的 API，如果考虑所有交易所，这些 API 变化相当频繁。
为确保持续兼容性，请确保定期更新您的机器人。

## Docker

!!! Note "使用 `master` 镜像的旧版安装"
    我们正在将发布镜像从 master 切换到 stable - 请调整您的 docker 文件，将 `freqtradeorg/freqtrade:master` 替换为 `freqtradeorg/freqtrade:stable`

``` bash
docker compose pull
docker compose up -d
```

## 通过安装脚本更新

``` bash
./setup.sh --update
```

!!! Note
    请确保在禁用虚拟环境的情况下运行此命令！

## 原生手动安装

请确保您同时更新了依赖项 - 否则可能会在您不知情的情况下出现问题。

``` bash
git pull
pip install -U -r requirements.txt
pip install -e .

# 确保 freqUI 是最新版本
freqtrade install-ui 
```

## 更新问题

更新问题通常来自缺少依赖项（您没有按照上述说明操作）- 或者来自安装失败的依赖项。
我们尽量确保主要平台都有重量级依赖项的预编译包，但有时这是不可能的。

请参考相应的安装章节（常见问题章节链接如下）。

[常见安装问题](installation.md#troubleshooting)
[常见安装问题 - Windows](installation.md#windows-installation-error)