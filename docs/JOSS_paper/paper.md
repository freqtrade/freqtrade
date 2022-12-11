---
title: '`FreqAI`: generalizing adaptive modeling for chaotic time-series market forecasts'
tags:
  - Python
  - Machine Learning
  - adaptive modeling
  - chaotic systems
  - time-series forecasting
authors:
  - name: Robert A. Caulk Ph.D
    orcid: 0000-0001-5618-8629
    affiliation: 1, 2 
  - name: Elin Törnquist Ph.D
    orcid: 0000-0003-3289-8604
    affiliation: 1, 2
  - name: Matthias Voppichler 
    orcid: 
    affiliation: 2
  - name: Andrew R. Lawless
    orcid: 
    affiliation: 2
  - name: Ryan McMullan
    orcid: 
    affiliation: 2
  - name: Wagner Costa Santos
    orcid: 
    affiliation: 1, 2
  - name: Timothy C. Pogue
    orcid: 
    affiliation: 1, 2
  - name: Johan van der Vlugt
    orcid: 
    affiliation: 2
  - name: Stefan P. Gehring
    orcid: 
    affiliation: 2
  - name: Pascal Schmidt
    orcid: 0000-0001-9328-4345
    affiliation: 2

<!-- affiliation: "1, 2" # (Multiple affiliations must be quoted) -->
affiliations:
 - name: Emergent Methods LLC, Arvada Colorado, 80005, USA
   index: 1
 - name: Freqtrade open source project
   index: 2
date: October 2022
bibliography: paper.bib


---


# Statement of need

Forecasting chaotic time-series based systems, such as equity/cryptocurrency markets, requires a broad set of tools geared toward testing a wide range of hypotheses. Fortunately, a recent maturation of robust machine learning libraries (e.g. `scikit-learn`), has opened up a wide range of research possibilities. Scientists from a diverse range of fields can now easily prototype their studies on an abundance of established machine learning algorithms. Similarly, these user-friendly libraries enable "citizen scientists" to use their basic Python skills for data-exploration. However, leveraging these machine learning libraries on historical and live chaotic data sources can be logistically difficult and expensive. Additionally, robust data-collection, storage, and handling presents a disparate challenge. [`FreqAI`](https://www.freqtrade.io/en/latest/freqai/) aims to provide a generalized and extensible open-sourced framework geared toward live deployments of adaptive modeling for market forecasting. The `FreqAI` framework is effectively a sandbox for the rich world of open-source machine learning libraries. Inside the `FreqAI` sandbox, users find they can combine a wide variety of third-party libraries to test creative hypotheses on a free live 24/7 chaotic data source - cryptocurrency exchange data. 


# Summary

[`FreqAI`](https://www.freqtrade.io/en/latest/freqai/) evolved from a desire to test and compare a range of adaptive time-series forecasting methods on chaotic data. Cryptocurrency markets provide a unique data source since they are operational 24/7 and the data is freely available via a variety of open-sourced [exchange APIs](https://docs.ccxt.com/en/latest/manual.html#exchange-structure). Luckily, an existing open-source software, [`Freqtrade`](https://www.freqtrade.io/en/stable/), had already matured under a range of talented developers to support robust data collection/storage, as well as robust live environmental interactions for standard algorithmic trading. `Freqtrade` also provides a set of data analysis/visualization tools for the evaluation of historical performance as well as live environmental feedback. `FreqAI` builds on top of `Freqtrade` to include a user-friendly well tested interface for integrating external machine learning libraries for adaptive time-series forecasting. Beyond enabling the integration of existing libraries, `FreqAI` hosts a range of custom algorithms and methodologies aimed at improving computational and predictive performances. Thus, `FreqAI` contains a range of unique features which can be easily tested in combination with all the existing Python-accessible machine learning libraries to generate novel research on live and historical data. 

The high-level overview of the software is depicted in Figure 1.

![freqai-algo](assets/freqai_algo.jpg)
*Abstracted overview of FreqAI algorithm*

## Connecting machine learning libraries

Although the `FreqAI`  framework is designed to accommodate any Python library in the "Model training" and "Feature set engineering" portions of the software (Figure 1), it already boasts a wide range of well documented examples based on various combinations of:

* scikit-learn [@scikit-learn], Catboost [@catboost], LightGBM [@lightgbm], XGBoost [@xgboost], stable_baselines3 [@stable-baselines3], openai gym [@openai], tensorflow [@tensorflow], pytorch [@pytorch], Scipy [@scipy], Numpy [@numpy], and pandas [@pandas]. 

These mature projects contain a wide range of peer-reviewed and industry standard methods, including:

* Regression, Classification, Neural Networks, Reinforcement Learning, Support Vector Machines, Principal Component Analysis, point clustering, and much more. 

which are all leveraged in `FreqAI` for users to use as templates or extend with their own methods.
  
## Furnishing novel methods and features

Beyond the industry standard methods available through external libraries - `FreqAI` includes novel methods which are not available anywhere else in the open-source (or scientific) world. For example, `FreqAI` provides :

* a custom algorithm/methodology for adaptive modeling details [here](https://www.freqtrade.io/en/stable/freqai/#general-approach) and [here](https://www.freqtrade.io/en/stable/freqai-developers/#project-architecture)
* rapid and self-monitored feature engineering tools, details [here](https://www.freqtrade.io/en/stable/freqai-feature-engineering/#feature-engineering)
* unique model features/indicators, such as the [inlier metric](https://www.freqtrade.io/en/stable/freqai-feature-engineering/#inlier-metric)
* optimized data collection/storage algorithms, all code shown [here](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/freqai/data_drawer.py)
* safely integrated outlier detection methods, details [here](https://www.freqtrade.io/en/stable/freqai-feature-engineering/#outlier-detection)
* websocket communicated forecasts, details [here](https://www.freqtrade.io/en/stable/producer-consumer/)

Of particular interest for researchers, `FreqAI` provides the option of large scale experimentation via an optimized [websocket communications interface](https://www.freqtrade.io/en/stable/producer-consumer/).

## Optimizing the back-end

`FreqAI` aims to make it simple for users to combine all the above tools to run studies based in two distinct modules:

* backtesting studies
* live-deployments 

Both of these modules and their respective data management systems are built on top of [`Freqtrade`](https://www.freqtrade.io/en/latest/), a mature and actively developed cryptocurrency trading software. This means that `FreqAI` benefits from a wide range of tangential/disparate feature developments such as:

* FreqUI, a graphical interface for backtesting and live monitoring
* telegram control
* robust database handling
* futures/leverage trading
* dollar cost averaging
* trading strategy handling
* a variety of free data sources via [CCXT](https://docs.ccxt.com/en/latest/manual.html#exchange-structure) (FTX, Binance, Kucoin etc.)

These features derive from a strong external developer community that shares in the benefit and stability of a communal CI (Continuous Integration) system. Beyond the developer community, `FreqAI` benefits strongly from the userbase of `Freqtrade`, where most `FreqAI` beta-testers/developers originated. This symbiotic relationship between `Freqtrade` and `FreqAI` ignited a thoroughly tested [`beta`](https://github.com/freqtrade/freqtrade/pull/6832), which demanded a four month beta and [comprehensive documentation](https://www.freqtrade.io/en/latest/freqai/) containing:

* numerous example scripts
* a full parameter table
* methodological descriptions
* high-resolution diagrams/figures
* detailed parameter setting recommendations

## Providing a reproducible foundation for researchers

`FreqAI` provides an extensible, robust, framework for researchers and citizen data scientists. The `FreqAI` sandbox enables rapid conception and testing of exotic hypotheses. From a research perspective, `FreqAI` handles the multitude of logistics associated with live deployments, historical backtesting, and feature engineering. With `FreqAI`, researchers can focus on their primary interests of feature engineering and hypothesis testing rather than figuring out how to collect and handle data. Further - the well maintained and easily installed open-source framework of `FreqAI` enables reproducible scientific studies. This reproducibility component is essential to general scientific advancement in time-series forecasting for chaotic systems. 

# Technical details

Typical users configure `FreqAI` via two files:

1. A `configuration` file (`--config`) which provides access to the full parameter list available [here](https://www.freqtrade.io/en/latest/freqai/):
  * control high-level feature engineering
  * customize adaptive modeling techniques
  * set any model training parameters available in third-party libraries
  * manage adaptive modeling parameters (retrain frequency, training window size, continual learning, etc.)

2. A strategy file (`--strategy`) where users:
  * list of the base training features
  * set standard technical-analysis strategies
  * control trade entry/exit criteria

With these two files, most users can exploit a wide range of pre-existing integrations in `Catboost` and 7 other libraries with a simple command:

```
freqtrade trade --config config_freqai.example.json --strategy FreqaiExampleStrategy --freqaimodel CatboostRegressor
```

Advanced users will edit one of the existing `--freqaimodel` files, which are simply an children of the `IFreqaiModel` (details below). Within these files, advanced users can customize training procedures, prediction procedures, outlier detection methods, data preparation, data saving methods, etc. This is all configured in a way where they can customize as little or as much as they want. This flexible customization is owed to the foundational architecture in `FreqAI`, which is comprised of three distinct Python objects:

* `IFreqaiModel`
  * A singular long-lived object containing all the necessary logic to collect data, store data, process data, engineer features, run training, and inference models. 
* `FreqaiDataKitchen`
  * A short-lived object which is uniquely created for each asset/model. Beyond metadata, it also contains a variety of data processing tools. 
* `FreqaiDataDrawer`
  * Singular long-lived object containing all the historical predictions, models, and save/load methods. 

These objects interact with one another with one goal in mind - to provide a clean data set to machine learning experts/enthusiasts at the user endpoint. These power-users interact with an inherited `IFreqaiModel` that allows them to dig as deep or as shallow as they wish into the inheritence tree. Typical power-users focus their efforts on customizing training procedures and testing exotic functionalities available in third-party libraries. Thus, power-users are freed from the algorithmic weight associated with data management, and can instead focus their energy on testing creative hypotheses. Meanwhile, some users choose to override deeper functionalities within `IFreqaiModel` to help them craft unique data structures and training procedures.

The class structure and algorithmic details are depicted in the following diagram: 

![image](assets/freqai_algorithm-diagram.jpg)
*Class diagram summarizing object interactions in FreqAI*

# Online documentation

The documentation for [`FreqAI`](https://www.freqtrade.io/en/latest/freqai/) is available online at [https://www.freqtrade.io/en/latest/freqai/](https://www.freqtrade.io/en/latest/freqai/) and covers a wide range of materials:

* Quick-start with a single command and example files - (beginners)
* Introduction to the feature engineering interface and basic configurations - (intermediate users)
* Parameter table with indepth descriptions and default parameter setting recommendations - (intermediate users)
* Data analysis and post-processing - (advanced users)
* Methodological considerations complemented by high resolution figures - (advanced users)
* Instructions for integrating third party machine learning libraries into custom prediction models - (advanced users)
* Software architectural description with class diagram - (developers) 
* File structure descriptions - (developers)

The docs direct users to a variety of pre-made examples which integrate `Catboost`, `LightGBM`, `XGBoost`, `Sklearn`, `stable_baselines3`, `torch`, `tensorflow`. Meanwhile, developers will also find thorough docstrings and type hinting throughout the source code to aid in code readability and customization. 

`FreqAI` also benefits from a strong support network of users and developers on the [`Freqtrade` discord](https://discord.gg/w6nDM6cM4y) as well as on the [`FreqAI` discord](https://discord.gg/xE4RMg4QYw). Within the `FreqAI` discord, users will find a deep and easily searched knowledge base containing common errors. But more importantly, users in the `FreqAI` discord share anectdotal and quantitative observations which compare performance between various third-party libraries and methods.

# State of the field

There are two other open-source tools which are geared toward helping users build models for time-series forecasts on market based data. However, each of these tools suffer from a non-generalized frameworks that do not permit comparison of methods and libraries. Additionally, they do not permit easy live-deployments or adaptive-modeling methods. For example, two open-sourced projects called [`tensortrade`](https://tensortradex.readthedocs.io/en/latest/) [@tensortrade] and [`FinRL`](https://github.com/AI4Finance-Foundation/FinRL) [@finrl] limit users to the exploration of reinforcement learning on historical data. These softwares also do not provide robust live deployments, they do not furnish novel feature engineering algorithms, and they do not provide custom data analysis tools. `FreqAI` fills the gap. 

# On-going research

Emergent Methods, based in Arvada CO, is actively using `FreqAI` to perform large scale experiments aimed at comparing machine learning libraries in live and historical environments. Past projects include backtesting parametric sweeps, while active projects include a 3 week live deployment comparison between `CatboostRegressor`, `LightGBMRegressor`, and `XGBoostRegressor`. Results from these studies are planned for submission to scientific journals as well as more general data science blogs (e.g. Medium).

# Installing and running `FreqAI`

`FreqAI` is automatically installed with `Freqtrade` using the following commands on linux systems:

```
git clone git@github.com:freqtrade/freqtrade.git
cd freqtrade
./setup.sh -i
```

However, `FreqAI` also benefits from `Freqtrade` docker distributions, and can be run with docker by pulling the stable or develop images from `Freqtrade` distributions.

# Funding sources

[`FreqAI`](https://www.freqtrade.io/en/latest/freqai/) has had no official sponsors, and is entirely grass roots. All donations into the project (e.g. the GitHub sponsor system) are kept inside the project to help support development of open-sourced and communally beneficial features. 

# Acknowledgements

We would like to acknowledge various beta testers of `FreqAI`:

- Longlong Yu (lolongcovas)
- Richárd Józsa (richardjozsa)
- Juha Nykänen (suikula)
- Emre Suzen (aemr3)
- Salah Lamkadem (ikonx)

As well as various `Freqtrade` [developers](https://github.com/freqtrade/freqtrade/graphs/contributors) maintaining tangential, yet essential, modules.

# References
