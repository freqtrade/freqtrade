<!-- 本文件为中文翻译版，由 AI 根据 docs/data-analysis.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，图片与 includes 引用使用 ../ 指向英文原文档资源。 -->

# 使用 Jupyter notebook 分析机器人数据

你可以使用 Jupyter notebook 轻松分析回测结果和交易历史。在使用 `freqtrade create-userdir --userdir user_data` 初始化用户目录后，示例 notebook 位于 `user_data/notebooks/`。

## 使用 docker 快速开始

Freqtrade 提供了一个 docker-compose 文件，用于启动 jupyter lab 服务器。
你可以使用以下命令运行该服务器：`docker compose -f docker/docker-compose-jupyter.yml up`

这会创建一个运行 jupyter lab 的 docker 容器，可通过 `https://127.0.0.1:8888/lab` 访问。
请使用启动后控制台中打印的链接进行简化登录。

更多信息，请访问[使用 Docker 进行数据分析](docker_quickstart.md#data-analysis-using-docker-compose)章节。

### 专业提示

* 使用说明请参阅 [jupyter.org](https://jupyter.org/documentation)。
* 别忘了在 conda 或 venv 环境中启动 Jupyter notebook 服务器，或使用 [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels)*
* 使用之前先复制示例 notebook，以免你的改动在下次 freqtrade 更新时被覆盖。

### 在系统级 Jupyter 安装中使用虚拟环境

有时你可能希望使用系统级安装的 Jupyter notebook，并使用来自虚拟环境的 jupyter kernel。
这样可以避免在每个系统上多次安装完整的 jupyter 套件，并提供一种在任务（freqtrade / 其他分析任务）之间轻松切换的方式。

为此，首先激活你的虚拟环境并运行以下命令：

``` bash
# 激活虚拟环境
source .venv/bin/activate

pip install ipykernel
ipython kernel install --user --name=freqtrade
# 重启 jupyter (lab / notebook)
# 在 notebook 中选择 "freqtrade" 内核
```

!!! Note
    提供本节是为了内容完整性，Freqtrade 团队不会对此设置的全部问题提供支持，并建议直接在虚拟环境中安装 Jupyter，因为这是让 jupyter notebook 启动并运行最简单的方式。关于此设置的帮助，请参阅 [Project Jupyter](https://jupyter.org/) 的[文档](https://jupyter.org/documentation)或[帮助频道](https://jupyter.org/community)。

!!! Warning
    某些任务在 notebook 中运行得并不理想。例如，任何使用异步执行的内容对 Jupyter 来说都是个问题。此外，freqtrade 的主要入口是 shell cli，因此在 notebook 中使用纯 python 会绕过那些为辅助函数提供所需对象和参数的参数。你可能需要手动设置这些值或创建预期的物件。

## 推荐的工作流

| 任务 | 工具 |
| --- | --- |
| 机器人操作 | CLI |
| 重复任务 | Shell 脚本 |
| 数据分析与可视化 | Notebook |

1. 使用 CLI 来
    * 下载历史数据
    * 运行回测
    * 使用实时数据运行
    * 导出结果

1. 把这些操作收集到 shell 脚本中
    * 保存带参数的复杂命令
    * 执行多步操作
    * 自动化测试策略和准备分析数据

1. 使用 notebook 来
    * 可视化数据
    * 处理并绘图以生成洞察

## 示例实用代码段

### 切换目录到项目根目录

Jupyter notebook 从 notebook 所在目录执行。以下代码段会搜索项目根目录，以保持相对路径一致。

```python
import os
from pathlib import Path

# 切换目录
# 修改此单元格以确保输出显示正确的路径。
# 所有路径都相对于单元格输出中显示的项目根目录来定义
project_root = "somedir/freqtrade"
i=0
try:
    os.chdir(project_root)
    assert Path('LICENSE').is_file()
except:
    while i<4 and (not Path('LICENSE').is_file()):
        os.chdir(Path(Path.cwd(), '../'))
        i+=1
    project_root = Path.cwd()
print(Path.cwd())
```

### 加载多个配置文件

此选项可用于检查传入多个配置文件的结果。
这也会完整运行配置初始化流程，因此配置会被完全初始化，可以传递给其他方法。

``` python
import json
from freqtrade.configuration import Configuration

# 从多个文件加载配置
config = Configuration.from_files(["config1.json", "config2.json"])

# 显示内存中的配置
print(json.dumps(config['original_config'], indent=2))
```

对于交互式环境，可以额外准备一个指定 `user_data_dir` 的配置，并将其最后传入，这样在运行机器人时就无需切换目录。
最好避免使用相对路径，因为相对路径是从 jupyter notebook 的存储位置开始的，除非切换了目录。

``` json
{
    "user_data_dir": "~/.freqtrade/"
}
```

### 更多数据分析文档

* [策略调试](strategy_analysis_example.md) - 也以 Jupyter notebook 形式提供（`user_data/notebooks/strategy_analysis_example.ipynb`）
* [绘图](plotting.md)
* [标签分析](advanced-backtesting.md)

如果你愿意分享关于如何最好地分析数据的想法，欢迎提交 issue 或 Pull Request 来完善本文档。
