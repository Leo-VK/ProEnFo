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

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    ###################
    ### Import data ###
    ###################
    site_id = 'GFC14_load'
    file_name = "load_with_weather.pkl"
    data = pd.read_pickle(f"data/{site_id}/{file_name}")
    target = 'load'
    categories = data.columns.get_level_values(0) if target not in data.columns.get_level_values(0) else ["system"]
    quantiles = [round(k / 100, 2) for k in range(1, 100)]
    methods_to_train = [bi.BMQ(7, [round(k / 100, 2) for k in range(1, 100)]),
                        bi.BEQ([round(k / 100, 2) for k in range(1, 100)]),
                        bi.BCEP([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQCE([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQKNNR([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQRFR([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQSRFR([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQERT([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQSERT([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQFFNN([round(k / 100, 2) for k in range(1, 100)]),
                    mi.MQCNN([round(k / 100, 2) for k in range(1, 100)]),
                    mi.MQLSTM([round(k / 100, 2) for k in range(1, 100)]),
                    mi.MQLSTN([round(k / 100, 2) for k in range(1, 100)]),
                    mi.MQWaveNet([round(k / 100, 2) for k in range(1, 100)]),
                        mi.MQNBEATS([round(k / 100, 2) for k in range(1, 100)]),
                    mi.MQTransformer([round(k / 100, 2) for k in range(1, 100)])]
    horizon_list = [24]  # or int(dt_resolution.seconds/data.index.freq.n)
    train_ratio = 0.8
    feature_transformation = ft.NoTransformationStrategy()
    time_stationarization = ts.NoStationarizationStrategy()
    datetime_features = [tc.Hour(),tc.Month(),tc.Day(),tc.Weekday()]
    target_lag_selection = fls.ManualStrategy(lags=[24, 48, 72, 96, 120, 144, 168])
    external_feature_selection = fes.ZeroLagStrategy(["airTemperature"])
    evaluation_metrics = [em.ReliabilityMatrix([round(k / 100, 2) for k in range(1, 100)]),
                        em.CalibrationMatrix([round(k / 100, 2) for k in range(1, 100)]),
                        em.WinklerScoreMatrix([round(k / 100, 2) for k in range(1, 100)]),
                        em.IntervalWidthMatrix([round(k / 100, 2) for k in range(1, 100)]),
                        #   em.RampScoreMatrix(threshold=0.1, normalize=True),
                        em.PinballLossMatrix([round(k / 100, 2) for k in range(1, 100)]),
                        em.QuantileCrossingMatrix([round(k / 100, 2) for k in range(1, 100)]),
                        em.PinballLoss([round(k / 100, 2) for k in range(1, 100)]),
                        em.CalibrationError([round(k / 100, 2) for k in range(1, 100)]),
                        em.BoundaryCrossing(0, None),
                        em.QuantileCrossing([round(k / 100, 2) for k in range(1, 100)]),
                        em.Skewness(),
                        em.Kurtosis(),
                        em.QuartileDispersion()]
    post_processing_quantile = ppq.QuantileSortingStrategy()
    post_processing_value = ppv.ValueClippingStrategy(0, None)
    save_results = True
    for category in categories:
        for horizon in horizon_list:
            err_tot, forecast_tot,true = calculate_scenario(data=data if category == "system" else data[category],
                                                    target=target,
                                                    methods_to_train=methods_to_train,
                                                    horizon=horizon,
                                                    train_ratio=train_ratio,
                                                    feature_transformation=feature_transformation,
                                                    time_stationarization=time_stationarization,
                                                    datetime_features=datetime_features,
                                                    target_lag_selection=target_lag_selection,
                                                    external_feature_selection=external_feature_selection,
                                                    post_processing_quantile=post_processing_quantile,
                                                    post_processing_value=post_processing_value,
                                                    evaluation_metrics=evaluation_metrics)

            # Save model forecast and metrics
            for method in methods_to_train:
                # print(f'{method.name}')
                df_err_tot = pd.DataFrame({'value': err_tot[method.name]})
                df_forecast_tot = forecast_tot[method.name]
                # print(df_err_tot)

                if save_results:
                    # Check saving directory
                    directory = f"data/{site_id}/results/probabilistic/{category}/" \
                                f"{feature_transformation.name}_" \
                                f"{time_stationarization.name}_" \
                                f"{target_lag_selection.name}/" \
                                f"{external_feature_selection.name}/" \
                                f"{post_processing_quantile.name}/" \
                                f"{post_processing_value.name}/" \
                                f"h{horizon}/"

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    df_err_tot.to_pickle(f"{directory}/errors_{method.name}.pkl")
                    df_forecast_tot.to_pickle(f"{directory}/forecasts_{method.name}.pkl")
            true.to_pickle(f"{directory}/true.pkl")


if __name__ == "__main__":
    main()

