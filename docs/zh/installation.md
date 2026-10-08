# 安装（Installation）

本页介绍如何为运行机器人准备你的环境。

freqtrade 文档描述了多种安装 freqtrade 的方法：

* [Docker 镜像](docker_quickstart.md)（独立页面）
* [脚本安装](#脚本安装)
* [手动安装](#手动安装)
* [使用 Conda 安装](#使用-conda-安装)

请考虑使用预构建的 [docker 镜像](docker_quickstart.md) 以快速开始。

!!! Note "更新"
    保持 freqtrade 更新对于[确保与交易所 API 的持续兼容性](updating.md#为什么要更新)很重要。
    有关如何更新你的安装，请参阅[更新指南](updating.md)。

!!! Note "Windows 用户"
    我们**强烈**建议 Windows 用户使用 [Docker](docker_quickstart.md)，因为这样会容易和平滑得多（也更安全）。

    如果那不可能，请尝试使用 Windows Linux 子系统（WSL） - 对于 WSL，Ubuntu/Linux 的说明将适用。
    如果你真的想在 Windows 上原生安装 freqtrade，最好使用 [`./setup.ps1` 安装脚本](#使用-setupps1-windows)。

    还请确保使用 64 位版本的 Python，因为 32 位版本有严重的内存限制，这会对你的回测/hyperopt 体验产生负面影响。

------

## 信息

安装和运行 Freqtrade 最简单的方法是克隆 bot 的 Github 仓库，然后在你的平台上运行 `./setup.sh`（Windows 为 `./setup.ps1`）脚本（如果可用）。

!!! Note "版本注意事项"
    克隆仓库时，默认工作分支名为 `develop`。该分支包含所有最新功能（可视为相对稳定，得益于自动化测试）。
    `stable` 分支包含上次发布的代码（通常每月一次，在 `develop` 分支大约一周前的快照上进行，以防止打包错误，因此可能更稳定）。

!!! Note
    假设已提供 [uv](https://docs.astral.sh/uv/)，或 Python3.11 或更高版本以及相应的 `pip`。安装脚本会警告并停止（如果不满足条件）。克隆 Freqtrade 仓库还需要 `git`。
    此外，必须提供 python 头文件（`python<你的版本>-dev` / `python<你的版本>-devel`）才能成功完成安装。

!!! Warning "时钟保持最新"
    运行 bot 的系统上的时钟必须准确，并足够频繁地与 NTP 服务器同步，以避免与交易所通信出现问题。

------

## 要求

这些要求适用于[脚本安装](#脚本安装)和[手动安装](#手动安装)。

!!! Note "ARM64 系统"
    如果你正在运行 ARM64 系统（如 MacOS M1 或 Oracle 虚拟机），请使用 [docker](docker_quickstart.md) 来运行 freqtrade。
    虽然通过一些手动操作可以进行原生安装，但目前不支持。

### 安装指南

* [Python >= 3.11](http://docs.python-guide.org/en/latest/starting/installation/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html)（推荐）

### 安装代码

我们收集/整理了 Ubuntu、MacOS 和 Windows 的安装说明。这些是指导方针，使用其他发行版时你的成功率可能会有所不同。
操作系统特定的步骤先列出，下面的公共部分是所有系统都必需的。

!!! Note
    假设已提供 Python3.11 或更高版本以及相应的 pip。

=== "Debian/Ubuntu"
    #### 安装必要的依赖

    ```bash
    # 更新仓库
    sudo apt-get update

    # 安装软件包
    sudo apt install -y python3-pip python3-venv python3-dev python3-pandas git curl
    ```

=== "MacOS"
    #### 安装必要的依赖

    如果你还没有，请安装 [Homebrew](https://brew.sh/)。

    ```bash
    # 安装软件包
    brew install gettext libomp
    ```
    !!! Note
        假设你系统上已安装 brew，`setup.sh` 脚本会为你安装这些依赖。

=== "RaspberryPi/Raspbian"
    以下假设你使用的是最新的 [Raspbian Buster lite 镜像](https://www.raspberrypi.org/downloads/raspbian/)。
    该镜像预装了 python3.11，使得启动并运行 freqtrade 变得容易。

    使用带有 Raspbian Buster lite 镜像的 Raspberry Pi 3 测试，应用了所有更新。

    ```bash
    sudo apt-get install python3-venv libatlas-base-dev cmake curl libffi-dev
    # 使用 piwheels.org 加速安装
    sudo echo "[global]\nextra-index-url=https://www.piwheels.org/simple" > tee /etc/pip.conf

    git clone https://github.com/freqtrade/freqtrade.git
    cd freqtrade

    bash setup.sh -i
    ```

    !!! Note "安装时长"
        根据你的互联网速度和 Raspberry Pi 版本，安装可能需要多个小时才能完成。
        因此，我们建议使用预构建的 docker 镜像来运行 Raspberry，按照[Docker 快速入门文档](docker_quickstart.md)。

    !!! Note
        上面没有安装 hyperopt 依赖。要安装它们，请使用 `python3 -m pip install -e .[hyperopt]`。
        我们不建议在 Raspberry Pi 上运行 hyperopt，因为这是一个非常消耗资源的操作，应该在强大的机器上完成。

------

## Freqtrade 仓库

Freqtrade 是一个开源的加密货币交易机器人，其代码托管在 `github.com` 上。

```bash
# 下载 freqtrade 仓库的 `develop` 分支
git clone https://github.com/freqtrade/freqtrade.git

# 进入下载的目录
cd freqtrade

# 你的选择 (1)：新手用户
git checkout stable

# 你的选择 (2)：高级用户
git checkout develop
```

(1) 此命令将克隆的仓库切换到使用 `stable` 分支。如果你希望停留在 (2) `develop` 分支，则不需要。

你可以随时使用 `git checkout stable`/`git checkout develop` 命令在分支之间切换。

??? Note "从 pypi 安装"
    另一种安装 Freqtrade 的方式是从 [pypi](https://pypi.org/project/freqtrade/)。缺点是此方法需要事先正确安装 ta-lib，因此目前不是推荐的安装方式。

    ``` bash
    pip install freqtrade
    ```

------

## 脚本安装

安装 Freqtrade 的第一种方式是使用提供的 Linux/MacOS `./setup.sh` 脚本，它会安装所有依赖并帮助你配置机器人。

确保你满足[要求](#要求)并下载了 [Freqtrade 仓库](#freqtrade-仓库)。

### 使用 /setup.sh -install（Linux/MacOS）

如果你使用的是 Debian、Ubuntu 或 MacOS，freqtrade 提供脚本安装 freqtrade。

```bash
# --install，从头安装 freqtrade
./setup.sh -i
```

#### /setup.sh 脚本的其他选项

你也可以使用 `./setup.sh` 更新、配置和重置你的机器人代码库。

```bash
# --update，使用 git pull 命令更新。
./setup.sh -u
# --reset，硬重置你的 develop/stable 分支。
./setup.sh -r
```

```
** --install **

使用此选项，脚本将安装机器人及大部分依赖：
你需要事先安装 git 和 python3.11+ 才能使其工作。

* 必需软件如：`ta-lib`
* 在 `.venv/` 下设置你的 virtualenv

此选项是安装任务和 `--reset` 的组合

** --update **

此选项将拉取你当前分支的最新版本并更新你的 virtualenv。定期使用此选项运行脚本以更新你的机器人。

** --reset **

此选项将硬重置你的分支（仅当你在 `stable` 或 `develop` 上时），并重新创建你的 virtualenv。
```

#### 激活你的虚拟环境

每次打开新终端时，你必须运行 `source .venv/bin/activate` 来激活你的虚拟环境。

```bash
# 激活虚拟环境
source ./.venv/bin/activate
```

### 使用 ./setup.ps1（Windows）

脚本会问你几个问题，以确定应该安装哪些部分。

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass
cd freqtrade
. .\setup.ps1
```

#### 激活你的虚拟环境（Windows）

```powershell
# 激活虚拟环境
. .\.venv\Scripts\Activate.ps1
```

[你现在准备好了](#你已准备就绪) 来运行机器人。

-----

## 手动安装

确保你满足[要求](#要求)并下载了 [Freqtrade 仓库](#freqtrade-仓库)。

### 设置 Python 虚拟环境（virtualenv）

你将在独立的 `virtual environment`（虚拟环境）中运行 freqtrade。

```bash
# 在 /freqtrade/.venv 目录创建虚拟环境
python3 -m venv .venv

# 运行虚拟环境
source .venv/bin/activate
```

### 安装 python 依赖

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
# 安装 freqtrade
python3 -m pip install -e .
```

[你现在准备好了](#你已准备就绪) 来运行机器人。

### （可选）安装后任务

!!! Note
    如果你在服务器上运行机器人，你应该考虑使用 [Docker](docker_quickstart.md) 或像 `screen` 或 [`tmux`](https://en.wikipedia.org/wiki/Tmux) 这样的终端复用器，以避免机器人在注销时被停止。

在 Linux 上使用 `systemd` 软件套件时，作为可选的安装后任务，你可能希望将机器人设置为作为 `systemd service` 运行，或将其配置为将日志消息发送到 `syslog`/`rsyslog` 或 `journald` 守护进程。详见[高级日志](advanced-setup.md#高级日志)。

------

## 使用 Conda 安装

Freqtrade 也可以使用 Miniconda 或 Anaconda 安装。我们推荐使用 Miniconda，因为它的安装占用空间更小。Conda 会自动准备并管理 Freqtrade 程序的大量库依赖。

### 什么是 Conda？

Conda 是多种编程语言的包、依赖和环境管理器：[conda docs](https://docs.conda.io/projects/conda/en/latest/index.html)

### 使用 conda 安装

#### 安装 Conda

[在 linux 上安装](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent)

[在 windows 上安装](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)

回答所有问题。安装后，必须关闭并重新打开你的终端。

#### 下载 Freqtrade

下载并安装 freqtrade。

```bash
# 下载 freqtrade
git clone https://github.com/freqtrade/freqtrade.git

# 进入下载的目录 'freqtrade'
cd freqtrade
```

#### Freqtrade 安装：Conda 环境

```bash
conda create --name freqtrade python=3.12
```

!!! Note "创建 Conda 环境"
    conda 命令 `create -n` 会自动安装所选库的所有嵌套依赖，安装命令的一般结构为：

    ```bash
    # 选择你自己的包
    conda env create -n [环境名称] [python 版本] [包]
    ```

#### 进入/退出 freqtrade 环境

要检查可用的环境，输入

```bash
conda env list
```

进入已安装的环境

```bash
# 进入 conda 环境
conda activate freqtrade

# 退出 conda 环境 - 现在不要做
conda deactivate
```

使用 pip 安装最后的 python 依赖

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

[你现在准备好了](#你已准备就绪) 来运行机器人。

### 重要的快捷方式

```bash
# 列出已安装的 conda 环境
conda env list

# 激活 base 环境
conda activate

# 激活 freqtrade 环境
conda activate freqtrade

# 停用任何 conda 环境
conda deactivate
```

### 关于 anaconda 的更多信息

!!! Info "新的重量级包"
    创建新的 Conda 环境，在创建时填入所选包，可能比将大型、重量级库或应用程序安装到之前设置的环境中花费的时间更少。

!!! Warning "在 conda 内使用 pip"
    conda 的文档说不应在 conda 内使用 pip，因为可能会出现内部问题。
    不过，这种情况很少见。[Anaconda 博客文章](https://www.anaconda.com/blog/using-pip-in-a-conda-environment)

    尽管如此，这就是为什么首选 `conda-forge` 频道：

    * 更多库可用（更少需要 `pip`）
    * `conda-forge` 与 `pip` 配合得更好
    * 库更新

祝你交易愉快！

------

## 你已准备就绪

你做到了这一点，所以你已经成功安装了 freqtrade。

### 初始化配置

```bash
# 步骤 1 - 初始化用户文件夹
freqtrade create-userdir --userdir user_data

# 步骤 2 - 创建一个新的配置文件
freqtrade new-config --config user_data/config.json
```

你已准备好运行，请阅读[机器人配置](configuration.md)，记得从 `dry_run: True` 开始，并验证一切正常工作。

要了解如何设置你的配置，请参阅[机器人配置](configuration.md)文档页面。

### 启动机器人

```bash
freqtrade trade --config user_data/config.json --strategy SampleStrategy
```

!!! Warning
    你应该通读其余的文档，回测你要使用的策略，并在启用真实货币交易之前使用 dry-run。

-----

## 故障排除

### 常见问题："command not found"（找不到命令）

如果你使用了 (1)`Script` 或 (2)`Manual` 安装，你需要在虚拟环境中运行机器人。如果你遇到如下错误，请确保 venv 已激活。

```bash
# 如果：
bash: freqtrade: command not found

# 然后激活你的虚拟环境
source ./.venv/bin/activate
```

### MacOS 安装错误

较新版本的 MacOS 可能会安装失败，并出现类似 `error: command 'g++' failed with exit status 1` 的错误。

此错误将需要显式安装 SDK Headers，此版本 MacOS 默认未安装。
对于 MacOS 10.14，可以使用以下命令完成。

```bash
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```

如果这个文件不存在，那么你可能在不同的 MacOS 版本上，因此你可能需要上网查询具体的解决方案。

### Windows 安装错误

```bash
error: Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools": http://landinghub.visualstudio.com/visual-cpp-build-tools
```

不幸的是，许多需要编译的包不提供预构建的 wheel。因此，必须安装 C/C++ 编译器并使其对你的 python 环境可用。

你可以从 [Visual Studio 网站](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 下载 Visual C++ 构建工具，并以默认配置安装“使用 C++ 的桌面开发”。不幸的是，这是一个重量级的下载 / 依赖，所以你可能想先考虑 WSL2 或 [docker compose](docker_quickstart.md)。

![Windows 安装](../assets/windows_install.png)
