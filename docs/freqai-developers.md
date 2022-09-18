# Development 

The architechture and functions of FreqAI are generalized to encourage users to develop their own features, functions, models, etc.

The class structure and a detailed algorithmic overview is depicted in the following diagram: 

![image](assets/freqai_algorithm-diagram.jpg)

As shown, there are three distinct objects comprising FreqAI:

* **IFreqaiModel** - A singular persistent object containing all the necessary logic to collect, store, and process data, engineer features, run training, and inference models. 
* **FreqaiDataKitchen** - A non-persistent object which is created uniquely for each unique asset/model. Beyond metadata, it also contains a variety of data processing tools. 
* **FreqaiDataDrawer** - A singular persistent object containing all the historical predictions, models, and save/load methods. 

There are a variety of built-in [prediction models](freqai-configuration.md#using-different-prediction-models) which inherit directly from `IFreqaiModel`. Each of these models have full access to all methods in `IFreqaiModel` and can therefore override any of those functions at will. However, advanced users will likely stick to overriding `fit()`, `train()`, `predict()`, and `data_cleaning_train/predict()`. 
