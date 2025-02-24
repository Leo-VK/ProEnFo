import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as co
from IPython.display import display
pd.options.mode.chained_assignment = None
import os, sys
dir1 = os.path.dirname(os.path.abspath(''))
if not dir1 in sys.path: sys.path.append(dir1)
plt.rcParams["figure.figsize"] = (15,5)
plt.style.use('seaborn-v0_8-darkgrid')
import pandas as pd
from utils import randomness, pytorchtools
import feature.feature_lag_selection as fls
import feature.feature_external_selection as fes
import feature.feature_transformation as ft
import feature.time_categorical as tc
import feature.time_stationarization as ts
import models.model_init as mi
import models.benchmark_init as bi
import evaluation.metrics as em
import postprocessing.quantile as ppq
import postprocessing.value as ppv
import datetime as dt
from forecast.scenario import calculate_scenario
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")
import random

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    breakpoint_new=loadmat('./breakpoint_new.mat')
    breakpoint_new.pop("__header__")
    breakpoint_new.pop("__version__")
    breakpoint_new.pop("__globals__")
    breakpoint_raw=list(breakpoint_new.values())
    data = pd.read_pickle('./load_with_weather.pkl')

    target = ["load"]
    horizon = 1  # or int(dt_resolution.seconds/data.index.freq.n)
    train_ratio = 0.8
    lags_list = list(range(1, 25))
    preds_list = list(range(1, 24))
    feature_transformation = ft.NoTransformationStrategy()
    time_stationarization = ts.NoStationarizationStrategy()
    datetime_features = [tc.Hour(),tc.Day(),tc.Weekday(),tc.Month()]
    target_lag_selection = fls.ManualStrategy(lags=lags_list)
    target_pred_selection = fls.ManualStrategy(lags=preds_list)
    external_feature_selection = fes.LagStrategy({"airTemperature":fls.ManualStrategy(lags=lags_list)},
                                                {"airTemperature":fls.ManualStrategy(lags=preds_list)})
    evaluation_metrics = [
                        em.MAE(),
                        ]
    post_processing_quantile = ppq.QuantileSortingStrategy()
    post_processing_value = ppv.ValueClippingStrategy(0, None)
    save_results = False

    for i in range(100,103):
        setup_seed(i)
        methods_to_train = [  
                    mi.PNBEATSX(pytorchtools.ContinuousPiecewiseLinearFunction(breakpoint_raw[7]),device),
                    mi.PTSMixerExt(pytorchtools.ContinuousPiecewiseLinearFunction(breakpoint_raw[7]),device),
                    mi.PBiTCN(pytorchtools.ContinuousPiecewiseLinearFunction(breakpoint_raw[7]),device),
                    mi.PTiDE(pytorchtools.ContinuousPiecewiseLinearFunction(breakpoint_raw[7]),device),
                        mi.PTFT(pytorchtools.ContinuousPiecewiseLinearFunction(breakpoint_raw[7]),device),
                        ]

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
                                                    device=device,
                                                    prob_forecasting=False)
    
        with open(f'./err_tot_ours_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(err_tot, f)
        with open(f'./forecast_ours_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(forecast_tot, f)
        with open(f'./true_ours_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(true_tot, f)
        
        methods_to_train = [  
                  mi.PNBEATSX(device = device),
                  mi.PTSMixerExt(device = device),
                   mi.PBiTCN(device = device),
                   mi.PTiDE(device = device ),
                    mi.PTFT(device = device),
                    ]
        err_tot2, forecast_tot2,true_tot2 = calculate_scenario(data=data,
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
                                            device=device,
                                            prob_forecasting=False)
        with open(f'./err_tot_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(err_tot2, f)
        with open(f'./forecast_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(forecast_tot2, f)
        with open(f'./true_'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(true_tot2, f)