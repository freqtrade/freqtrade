<!-- 本文件为中文翻译版，由 AI 根据 docs/freqai-reinforcement-learning.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 强化学习

!!! Note "安装大小"
    强化学习依赖包含大型包，例如 `torch`，应在 `./setup.sh -i` 过程中通过回答 "y" 来明确请求安装，即对问题 "Do you also want dependencies for freqai-rl (~700mb additional space required) [y/N]?" 回答 "y"。
    偏好使用 Docker 的用户应确保使用带有 `_freqairl` 后缀的 Docker 镜像。

## 背景和术语

### 什么是强化学习？为什么 FreqAI 需要它？

强化学习包含两个重要组成部分：*智能体（agent）*和训练*环境（environment）*。在智能体训练过程中，智能体逐根K线遍历历史数据，始终从一组动作中选择一个执行：做多入场、做多出场、做空入场、做空出场、中性）。在此训练过程中，环境会跟踪这些动作的表现，并根据用户自定义的 `calculate_reward()` 函数对智能体进行奖励（我们提供了一个默认奖励函数供用户在此基础上构建，[详情请见此处](#creating-a-custom-reward-function)）。该奖励用于训练神经网络中的权重。

FreqAI 强化学习实现的第二个重要组成部分是*状态（state）*信息的使用。在每一步中，状态信息会被输入到网络中，包括当前利润、当前持仓和当前交易持续时间。这些信息在训练环境中用于训练智能体，并在模拟/实盘中用于强化智能体（此功能在回测中不可用）。*FreqAI + Freqtrade 是这种强化机制的完美组合，因为在实盘部署中这些信息可以轻松获取。*

强化学习是 FreqAI 的自然演进，因为它增加了一层分类器和回归器无法匹配的适应性和市场反应能力。然而，分类器和回归器也具有强化学习所不具备的优势，例如稳健的预测能力。训练不当的强化学习智能体可能会找到"作弊"和"取巧"的方式来最大化奖励，而实际上并未赢得任何交易。因此，强化学习比典型的分类器和回归器更加复杂，需要更高层次的理解。

### 强化学习接口

在当前框架中，我们旨在通过通用的"预测模型"文件来暴露训练环境，该文件是一个用户继承的 `BaseReinforcementLearner` 对象（例如 `freqai/prediction_models/ReinforcementLearner`）。在该用户类内部，强化学习环境可通过 `MyRLEnv` 进行访问和自定义，[如下所示](#creating-a-custom-reward-function)。

我们预计大多数用户将把精力集中在创造性地设计 `calculate_reward()` 函数上（[详情请见此处](#creating-a-custom-reward-function)），而保持环境的其他部分不变。其他用户可能完全不会修改环境，他们只会调整配置设置和 FreqAI 中已有的强大特征工程。同时，我们也允许高级用户完全创建自己的模型类。

该框架基于 stable_baselines3（torch）和 OpenAI gym 构建基础环境类。但总体而言，模型类具有良好的隔离性。因此，竞争库的添加可以轻松集成到现有框架中。对于环境，它继承自 `gym.Env`，这意味着如果要切换到不同的库，需要编写一个全新的环境。

### 重要注意事项

如上所述，智能体在一个人工交易"环境"中进行"训练"。在我们的案例中，该环境可能看起来与真实的 Freqtrade 回测环境非常相似，但它*并不是*。事实上，强化学习训练环境要简化得多。它不包含任何复杂的策略逻辑，例如 `custom_exit`、`custom_stoploss`、杠杆控制等回调函数。强化学习环境是真实市场的非常"原始"的表示，智能体可以自由学习由 `calculate_reward()` 强制执行的策略（即止损、止盈等）。因此，需要注意的是，智能体训练环境与真实世界并不完全相同。

## 运行强化学习

设置和运行强化学习模型与运行回归器或分类器相同。需要在命令行上定义相同的两个标志 `--freqaimodel` 和 `--strategy`：

```bash
freqtrade trade --freqaimodel ReinforcementLearner --strategy MyRLStrategy --config config.json
```

其中 `ReinforcementLearner` 将使用 `freqai/prediction_models/ReinforcementLearner` 中的模板化 `ReinforcementLearner`（或位于 `user_data/freqaimodels` 中的用户自定义模型）。另一方面，策略遵循与典型回归器相同的基础[特征工程](freqai-feature-engineering.md)，使用 `feature_engineering_*`。区别在于目标的创建，强化学习不需要目标。但是，FreqAI 要求在动作列中设置一个默认（中性）值：

```python
    def set_freqai_targets(self, dataframe, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/stable/freqai-feature-engineering

        :param df: strategy dataframe which will receive the targets
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        # For RL, there are no direct targets to set. This is filler (neutral)
        # until the agent sends an action.
        dataframe["&-action"] = 0
        return dataframe
```

该函数的大部分内容与典型回归器相同，但下面的函数展示了策略必须如何将原始价格数据传递给智能体，以便其在训练环境中能够访问原始 OHLCV 数据：

```python
    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # The following features are necessary for RL models
        dataframe[f"%-raw_close"] = dataframe["close"]
        dataframe[f"%-raw_open"] = dataframe["open"]
        dataframe[f"%-raw_high"] = dataframe["high"]
        dataframe[f"%-raw_low"] = dataframe["low"]
    return dataframe
```

最后，不需要显式创建"标签"——而是需要分配 `&-action` 列，该列在 `populate_entry/exit_trends()` 中访问时将包含智能体的动作。在当前示例中，中性动作设为 0。此值应与所使用的环境保持一致。FreqAI 提供了两种环境，均使用 0 作为中性动作。

当用户意识到不需要设置标签后，他们很快就会理解智能体正在做出自己的入场和出场决策。这使得策略构建变得相当简单。入场和出场信号以整数形式来自智能体——直接用于在策略中决定入场和出场：

```python
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [df["do_predict"] == 1, df["&-action"] == 1]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [df["do_predict"] == 1, df["&-action"] == 3]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-action"] == 2]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df["do_predict"] == 1, df["&-action"] == 4]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
```

需要注意的是，`&-action` 取决于用户选择使用的环境。上面的示例展示了 5 个动作，其中 0 是中性，1 是做多入场，2 是做多出场，3 是做空入场，4 是做空出场。

## 配置强化学习器

为了配置 `Reinforcement Learner`，`freqai` 配置中必须存在以下字典：

```json
        "rl_config": {
            "train_cycles": 25,
            "add_state_info": true,
            "max_trade_duration_candles": 300,
            "max_training_drawdown_pct": 0.02,
            "cpu_count": 8,
            "model_type": "PPO",
            "policy_type": "MlpPolicy",
            "model_reward_parameters": {
                "rr": 1,
                "profit_aim": 0.025
            }
        }
```

参数详情可以在[此处](freqai-parameter-table.md)找到，但一般来说，`train_cycles` 决定了智能体在人工环境中循环遍历K线数据以训练模型权重的次数。`model_type` 是一个字符串，用于选择 [stable_baselines](https://stable-baselines3.readthedocs.io/en/master/)（外部链接）中可用的模型之一。

!!! Note
    如果你想尝试 `continual_learning`，则应在主 `freqai` 配置字典中将该值设置为 `true`。这将告诉强化学习库从前一模型的最终状态继续训练新模型，而不是在每次触发重新训练时从头开始训练新模型。

!!! Note
    请记住，通用的 `model_training_parameters` 字典应包含特定 `model_type` 的所有模型超参数自定义。例如，`PPO` 参数可以在[此处](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)找到。

## 创建自定义奖励函数

!!! danger "非生产环境使用"
    警告！
    Freqtrade 源代码中提供的奖励函数是一个功能展示，旨在展示/测试尽可能多的环境控制功能。它也被设计为可以在小型计算机上快速运行。这是一个基准，*并非*用于实盘生产。请注意，你需要创建自己的 `custom_reward()` 函数或使用 Freqtrade 源代码之外的其他用户构建的模板。

当你开始修改策略和预测模型时，你将很快意识到强化学习器与回归器/分类器之间的一些重要区别。首先，策略不设置目标值（没有标签！）。相反，你需要在 `MyRLEnv` 类内部设置 `calculate_reward()` 函数（见下文）。`prediction_models/ReinforcementLearner.py` 中提供了一个默认的 `calculate_reward()` 来展示创建奖励所需的基本构建块，但这*并非*为生产环境设计。用户*必须*创建自己的自定义强化学习模型类，或使用 Freqtrade 源代码之外的预构建模型，并将其保存到 `user_data/freqaimodels`。在 `calculate_reward()` 内部，可以表达关于市场的创造性理论。例如，你可以在智能体赢得交易时给予奖励，在智能体亏损交易时给予惩罚。或者，你可能希望在智能体入场交易时给予奖励，在智能体持仓时间过长时给予惩罚。下面展示了这些奖励的计算示例：

!!! note "提示"
    最好的奖励函数是连续可微且缩放良好的函数。换句话说，对罕见事件添加单一的大幅负惩罚不是一个好主意，神经网络将无法学习该函数。相反，对常见事件添加小幅负惩罚效果更好。这将帮助智能体更快地学习。不仅如此，你还可以通过让奖励/惩罚根据某些线性/指数函数按严重程度缩放来改善奖励/惩罚的连续性。换句话说，随着交易持续时间的增加，你会逐步缩放惩罚。这比在单一时间点发生的单一大幅惩罚要好。

```python
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions


class MyCoolRLModel(ReinforcementLearner):
    """
    User created RL prediction model.

    Save this file to `freqtrade/user_data/freqaimodels`

    then use it with:

    freqtrade trade --freqaimodel MyCoolRLModel --config config.json --strategy SomeCoolStrat

    Here the users can override any of the functions
    available in the `IFreqaiModel` inheritance tree. Most importantly for RL, this
    is where the user overrides `MyRLEnv` (see below), to define custom
    `calculate_reward()` function, or to override any other parts of the environment.

    This class also allows users to override any other part of the IFreqaiModel tree.
    For example, the user can override `def fit()` or `def train()` or `def predict()`
    to take fine-tuned control over these processes.

    Another common override may be `def data_cleaning_predict()` where the user can
    take fine-tuned control over the data handling pipeline.
    """
    class MyRLEnv(Base5ActionRLEnv):
        """
        User made custom environment. This class inherits from BaseEnvironment and gym.Env.
        Users can override any functions from those parent classes. Here is an example
        of a user customized `calculate_reward()` function.

        Warning!
        This is function is a showcase of functionality designed to show as many possible
        environment control features as possible. It is also designed to run quickly
        on small computers. This is a benchmark, it is *not* for live production.
        """
        def calculate_reward(self, action: int) -> float:
            # first, penalize if the action is not valid
            if not self._is_valid(action):
                return -2
            pnl = self.get_unrealized_profit()

            factor = 100

            pair = self.pair.replace(':', '')

            # you can use feature values from dataframe
            # Assumes the shifted RSI indicator has been generated in the strategy.
            rsi_now = self.raw_features[f"%-rsi-period_10_shift-1_{pair}_"
                            f"{self.config['timeframe']}"].iloc[self._current_tick]

            # reward agent for entering trades
            if (action in (Actions.Long_enter.value, Actions.Short_enter.value)
                    and self._position == Positions.Neutral):
                if rsi_now < 40:
                    factor = 40 / rsi_now
                else:
                    factor = 1
                return 25 * factor

            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1
            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            trade_duration = self._current_tick - self._last_trade_tick
            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5
            # discourage sitting in position
            if self._position in (Positions.Short, Positions.Long) and \
            action == Actions.Neutral.value:
                return -1 * trade_duration / max_trade_duration
            # close long
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)
            # close short
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)
            return 0.
```

## 使用 Tensorboard

强化学习模型受益于训练指标的跟踪。FreqAI 已集成 Tensorboard，允许用户跟踪所有币种和所有重新训练的训练及评估性能。通过以下命令激活 Tensorboard：

```bash
tensorboard --logdir user_data/models/unique-id
```

其中 `unique-id` 是在 `freqai` 配置文件中设置的 `identifier`。此命令必须在单独的 shell 中运行，以在浏览器中查看输出，地址为 127.0.0.1:6006（6006 是 Tensorboard 使用的默认端口）。

![tensorboard](assets/tensorboard.jpg)

## 自定义日志

FreqAI 还提供了一个内置的逐情节摘要日志记录器，称为 `self.tensorboard_log`，用于向 Tensorboard 日志添加自定义信息。默认情况下，此函数在环境中每步已被调用一次以记录智能体动作。单个情节中所有步骤累积的所有值在每个情节结束时报告，随后所有指标完全重置为 0，为下一个情节做准备。

`self.tensorboard_log` 也可以在环境内部的任何位置使用，例如，可以将其添加到 `calculate_reward` 函数中，以收集有关奖励各部分被调用频率的更详细信息：

```python
    class MyRLEnv(Base5ActionRLEnv):
        """
        User made custom environment. This class inherits from BaseEnvironment and gym.Env.
        Users can override any functions from those parent classes. Here is an example
        of a user customized `calculate_reward()` function.
        """
        def calculate_reward(self, action: int) -> float:
            if not self._is_valid(action):
                self.tensorboard_log("invalid")
                return -2

```

!!! Note
    `self.tensorboard_log()` 函数设计用于仅跟踪递增对象，即训练环境中的事件、动作。如果感兴趣的事件是浮点数，可以将其作为第二个参数传递，例如 `self.tensorboard_log("float_metric1", 0.23)`。在这种情况下，指标值不会递增。

## 选择基础环境

FreqAI 提供了三种基础环境：`Base3ActionRLEnvironment`、`Base4ActionEnvironment` 和 `Base5ActionEnvironment`。顾名思义，这些环境针对可以选择 3、4 或 5 个动作的智能体进行了自定义。`Base3ActionEnvironment` 是最简单的，智能体可以选择持有、做多或做空。此环境也可用于仅做多的机器人（它自动遵循策略中的 `can_short` 标志），其中做多是入场条件，做空是出场条件。同时，在 `Base4ActionEnvironment` 中，智能体可以做多入场、做空入场、保持中性或退出持仓。最后，在 `Base5ActionEnvironment` 中，智能体具有与 Base4 相同的动作，但不是单一的退出动作，而是将做多出场和做空出场分开。环境选择带来的主要变化包括：

* `calculate_reward` 中可用的动作
* 用户策略使用的动作

所有 FreqAI 提供的环境都继承自一个与动作/位置无关的环境对象 `BaseEnvironment`，该对象包含所有共享逻辑。该架构设计为易于自定义。最简单的自定义是 `calculate_reward()`（详情请见[此处](#creating-a-custom-reward-function)）。然而，自定义可以进一步扩展到环境内部的任何函数。你可以通过在预测模型文件中的 `MyRLEnv` 内覆盖这些函数来实现。或者对于更高级的自定义，建议创建一个继承自 `BaseEnvironment` 的全新环境。

!!! Note
    只有 `Base3ActionRLEnv` 可以进行仅做多的训练/交易（设置用户策略属性 `can_short = False`）。