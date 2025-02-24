# EnFoAV

## Introduction

This is the code related to the paper 
"EnFoAV: Benchmarks and Custom Package for Energy Forecasting Considering Auxiliary Variables".
This repository mainly aims at implementing routines for probabilistic energy forecasting. However, we also provide the implementation of relevant point forecasting models.
The datasets and their forecasting results in this archive will be released after arrangement. Please refer to the notebook for reproducing the results on the GEF14 dataset. Users can easily construct different forecasting models by selecting different Feature engineering methods and preprocessing, post-processing, and training models.

## Dataset
We include several different datasets in our load forecasting archive. And there is a summary of them.
|    |  Dataset | No.of series | Length | Resolution |  Load type |          External variables         |
|:--:|:--------:|:------------:|:------:|:----------:|:----------:|:-----------------------------------:|
|  1 |  Covid19 |       1      |  31912 |   hourly   | aggregated |    airTemperature, Humidity, etc    |
|  2 |   GEF12  |      20      |  39414 |   hourly   | aggregated |            airTemperature           |
|  3 |   GEF14  |       1      |  17520 |   hourly   | aggregated |            airTemperature           |
|  4 |   GEF17  |       8      |  17544 |   hourly   | aggregated |            airTemperature           |
|  5 |    PDB   |       1      |  17520 |   hourly   | aggregated |            airTemperature           |
|  6 |  Spanish |       1      |  35064 |   hourly   | aggregated | airTemperature, seaLvlPressure, etc |
|  7 |    Hog   |      24      |  17544 |   hourly   |  building  |   airTemperature, wind speed, etc   |
|  8 |   Bull   |      41      |  17544 |   hourly   |  building  |   airTemperature, wind speed, etc   |
|  9 | Cockatoo |       1      |  17544 |   hourly   |  building  |   airTemperature, wind speed, etc   |
| 10 |    ELF   |       1      |  21792 |   hourly   | aggregated |                  No                 |
| 11 |    UCI   |      321     |  26304 |   hourly   |  building  |                  No                 |
| 12 |    REF   |       72     |  -     | 15 minute  |      -     |     irradiance, temperature, etc    |
| 13 |    Electricity Price   |       2     |  -     | hourly  |      -     |     load prediction, wind power prediction, etc    |

Among them, REF is our newly open-source renewable energy dataset, which contains several renewable energy series including onshore wind, offshore wind, and PV. We will make it open-source soon.

## Prerequisites
- Python 
- Conda

### Create a virtual environment
This is only needed when used for the first time on the machine.

```bash
conda create --name bench python=3.8.18
```

### Activate and deactivate environment
```bash
conda activate bench
conda deactivate
```

### Update your local environment

If there's a new package in the `requirements.txt` file you have to update the packages in your local env

```bash
pip install -r requirements.txt
```

### Initial packages include
  - python=3.8.18
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - plotly
  - statsmodels
  - xlrd
  - jupyterlab
  - nodejs
  - mypy
  - pytorch
## Overall framework
Our package covers the entire process of constructing forecasting models, including data preprocessing, construction of forecasting models, etc.
![contents](https://anonymous.4open.science/r/EnFoAV-3033/figure/Energy_Forecasting.jpg)
## Available forecasting models 
|    |    Models   | Paper |        Type       |                                      Description                                      |
|:--:|:-----------:|:-----:|:-----------------:|:-------------------------------------------------------------------------------------:|
|  1 |     KNNR    |  [link](https://link.springer.com/book/10.1007/978-0-387-21606-5) | Non-deep learning |                    quantile regression based on K-nearest neighbor                    |
|  2 |     RFR     |  [link](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf) | Non-deep learning |                       quantile regression based on random forest                      |
|  3 |     ERT     |  [link](https://link.springer.com/article/10.1007/s10994-006-6226-1) | Non-deep learning |                    quantile regression based on extreme random tree                   |
|  4 |     MLP    |  [link](https://ieeexplore.ieee.org/abstract/document/485891) |   deep learning   |                quantile regression based on feed-forward neural network               |
|  5 |     LSTM    |  [link](https://ieeexplore.ieee.org/abstract/document/6795963) |   deep learning   |                           quantile regression based on LSTM                           |
|  6 |     CNN     |  [link](https://ieeexplore.ieee.org/abstract/document/9451544) |   deep learning   |                            quantile regression based on CNN                           |
|  7 | Transformer |  [link](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |   deep learning   |                        quantile regression based on Transformer                       |
|  8 |    LSTNet   |  [link](https://dl.acm.org/doi/abs/10.1145/3209978.3210006) |   deep learning   |                          quantile regression based on LSTNet                          |
|  9 |   Wavenet   |  [link](https://arxiv.org/abs/1609.03499) |   deep learning   |                          quantile regression based on Wavenet                         |
| 10 |   N-BEATS   |  [link](https://arxiv.org/abs/1905.10437) |   deep learning   |                          quantile regression based on N-BEATS                         |
| 11 |   Informer   |   |   deep learning   |                          quantile regression based on Informer                         |
| 12 |   Autoformer   |  |   deep learning   |                          quantile regression based on Autoformer                         |
| 13 |   Fedformer   |   |   deep learning   |                          quantile regression based on Fedformer                         |
| 14 |   DLinear   |   |   deep learning   |                          quantile regression based on DLinear                         |
| 15 |   FiLM   |   |   deep learning   |                          quantile regression based on FiLM                         |
| 16 |   iTransformer   |   |   deep learning   |                          quantile regression based on iTransformer                         |
| 17 |   NSTransformer   |   |   deep learning   |                          quantile regression based on NSTransformer                         |
| 18 |   PatchTST   |   |   deep learning   |                          quantile regression based on PatchTST                         |
| 19 |   SegRNN   |   |   deep learning   |                          quantile regression based on SegRNN                         |
| 20 |   TimeMixer   |   |   deep learning   |                          quantile regression based on TimeMixer                         |
| 21 |   TimesNet   |   |   deep learning   |                          quantile regression based on TimesNet                         |
| 22 |   Tsmixer   |   |   deep learning   |                          quantile regression based on Tsmixer                         |
| 23 |   FreTS   |   |   deep learning   |                          quantile regression based on FreTS                         |
| 24 |   Reformer   |   |   deep learning   |                          quantile regression based on Reformer                         |
| 25 |   MICN   |   |   deep learning   |                          quantile regression based on MICN                         |
| 26 |   TimeXer   |   |   deep learning   |                          quantile regression based on TimeXer                         |
| 27 |   N-BEATSX   |   |   deep learning   |                          quantile regression based on N-BEATSX                         |
| 28 |   BiTCN   |   |   deep learning   |                          quantile regression based on BiTCN                         |
| 29 |   TFT   |   |   deep learning   |                          quantile regression based on TFT                         |
| 30 |   TiDE   |   |   deep learning   |                          quantile regression based on TiDE                         |
| 31 |   TsmixerEXT   |   |   deep learning   |                          quantile regression based on TsmixerEXT                         |
## Quick Start
To start forecasting, we first need to import some packages.
```python
import datetime as dt
import os
import pandas as pd
import evaluation.metrics as em
import feature.feature_lag_selection as fls
import feature.feature_external_selection as fes
import feature.feature_transformation as ft
import feature.time_categorical as tc
import feature.time_stationarization as ts
import models.model_init as mi
import models.benchmark_init as bi
import postprocessing.quantile as ppq
import postprocessing.value as ppv
from forecast.scenario import calculate_scenario
import random
import torch
import pickle
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import ast
```
Import the dataset, the example of the format of the dataset can be seen in [./data](https://github.com/Leo-VK/ProEnFo/tree/main/data/GFC14_load). 
```python
repeat = 3
site_id = "GEF14"
file_name = "load_with_weather.pkl"
data = pd.read_pickle(f"data/{site_id}/{file_name}")
target = 'load'
```
Define your forecasting setting, eg, forecasting quantiles, and feature engineering strategy.
err_tot, forecast_tot,true_tot = calculate_scenario(data=data,
                                                         target=target,
                                                         methods_to_train=methods_to_train,
                                                         horizon=horizon,
                                                         train_ratio=train_ratio,
                                                         feature_transformation=feature_transformation,
                                                         time_stationarization=time_stationarization,
                                                         datetime_features=datetime_features,
                                                         target_lag_selection=target_lag_selection,
                                                         target_pred_selection=target_pred_selection,
                                                         external_feature_selection=external_feature_selection,
                                                         post_processing_quantile=post_processing_quantile,
                                                         post_processing_value=post_processing_value,
                                                         evaluation_metrics=evaluation_metrics,
                                                         device=device,prob_forecasting = True)
```


## How to use the forecasting error-cost data
```python
from scipy.io import loadmat
breakpoint_new=loadmat('./simulated_data/breakpoint_new.mat')
breakpoint_new.pop("__header__")
breakpoint_new.pop("__version__")
breakpoint_new.pop("__globals__")
breakpoint_raw=list(breakpoint_new.values())
breakpoint = pytorchtools.breakpoint_generator(breakpoint_raw)[8]#take hour 9 as an example
loss_function = pytorchtools.ContinuousPiecewiseLinearFunction(breakpoint)
```

## How to add your own forecasting method into the framework
Based on Pytorch, users can simply add their own defined deep learning network to our forecasting framework.
Firstly, users need to define the initialization method for the model in [./models/model_init.py](https://anonymous.4open.science/r/EnFoAV-3033/models/model_init.py)
```python
class Predictor(MultiQuantileRegressor):
    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)
        

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=my_MQuantile_model(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device)
        return self
```
Secondly, users need to add the structure of the model in [./models/pytorch.py](https://anonymous.4open.science/r/EnFoAV-3033/models/pytorch.py)
```python
class my_MQuantile_Model(nn.Module):
    def __init__(self,configs):
        super(MYQuantile_Model, self).__init__()
        #Code here
    def forward(self, *inputs):
        #Code here
        return output
```
Finally, you can add your model to the methods_to_train.
```python
methods_to_train.append(mi.Predictor())
```

## Forecasting evaluation
We include several metrics to evaluate the forecasting performance and summarize them below. For details, users can check it in [./evaluation/metrics.py](https://anonymous.4open.science/r/EnFoAV-3033/evaluation/metrics.py)
|    |       Metrics      | Calculation method | Metric type |                                                                     Description                                                                    |
|:--:|:------------------:|:------------------:|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------:|
|  1 | CoverageError (CE) |          $$CE = \frac{1}{n} \sum_{t=1}^{n} (I(L_t \leq y_t \leq U_t) - (UB - LB))$$         |  probability  |    measures the difference between the proportion of actual observations falling within the forecasting interval and the expected coverage rate    |
|  2 | Winkler Score (WS) |        -        | probability |         evaluates whether the forecasting interval accurately captures actual observations, taking into account the width of the interval.         |
|  3 |  Pinball Loss (PL) |          $$PL=\frac{1}{n_\tau \cdot n} \sum_{t=1}^n \sum_{i=1}^{n_\tau} L_\tau\left(\hat{y}_{\tau,t}, y_t\right)$$         |  probability  |             weights the error based on whether the forecasting value falls on the side of the actual observation value (above or below)            |
|  4 |   RampScore (RS)   |          -         |  probability  |                                     measures the consistency of the slope (i.e. increasing or decreasing trend)                                    |
|  5 |  CalibrationError  |          -         |  probability  |                                      evaluates the accuracy of forecasting models in representing uncertainty                                      |
|  6 |    IntervalWidth   |          -         |  probability  |                                             calculates the width of probabilistic forecasting intervals                                            |
|  7 |  QuantileCrossing  |          -         |  probability  | gives the ratio of any two quantiles in which the predicted value of the lower quantile is greater than the predicted value of the higher quantile |
|  8 |  BoundaryCrossing  |          -         |  probability  |                                 calculates the probability that the true value falls outside the forecasting range                                 |
|  9 |      Skewness      |          -         |  probability  |                                                  describes the shape of a probability distribution                                                 |
| 10 |      Kurtosis      |          -         |  probability  |                                                  describes the shape of a probability distribution                                                 |
| 11 | QuartileDispersion |          -         |  probability  |                                                 measures the statistical dispersion of distribution                                                |
| 12 |        MAPE        |          $$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left\| \frac{y_t - F_t}{y_t} \right\| \times 100\%$$         |    point    |                                     calculates the average percentage of forecasting error for all data points                                     |
| 13 |         MAE        |          -         |    point    |                               calculates the average of the absolute value of forecasting errors for all data points                               |
| 14 |        MASE        |          $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - F_t)^2}$$         |    point    |         calculates errors by comparing the forecasting error with the average absolute first-order difference of the actual value sequence         |
| 15 |        RMSE        |          $$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} \left\| y_t - F_t \right\|$$         |    point    |                                calculates the square root of the average of the sum of squares of forecasting errors |

Based on different quantiles, we can give evaluation metrics in matrix form and visualize them to intuitively compare different forecasting methods. The following are relevant visualization examples. 

![contents](https://anonymous.4open.science/r/EnFoAV-3033/figure/visualization.jpg)

