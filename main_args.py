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



def main(args):
    ###################
    ### Import data ###
    ###################
    site_id = args.site_id
    file_name = args.file_name
    data = pd.read_pickle(f"data/{site_id}/{file_name}")
    target = args.target
    categories = data.columns.get_level_values(0) if target not in data.columns.get_level_values(0) else ["system"]
    quantiles = args.quantiles
    methods_to_train = args.methods_to_train
    horizon_list = args.horizon_list  # or int(dt_resolution.seconds/data.index.freq.n)
    train_ratio = args.train_ratio
    feature_transformation = args.feature_transformation
    time_stationarization = args.time_stationarization
    datetime_features = args.datetime_features
    target_lag_selection = args.target_lag_selection
    external_feature_selection = args.external_feature_selection
    evaluation_metrics = args.evaluation_metrics
    post_processing_quantile = args.post_processing_quantile
    post_processing_value = args.post_processing_value
    save_results = args.save_results
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
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argparse")
    parser.add_argument("site_id", help="site_id")
    parser.add_argument("--file_name", default = "load_with_weather.pkl",help="file_name")
    parser.add_argument("target", help="target")
    parser.add_argument("--quantiles",default= [round(k / 100, 2) for k in range(1, 100)], help="forecasting quantiles")
    parser.add_argument("--horizon_list",default=[24], help="the list the horizon")
    parser.add_argument("--methods_to_train",default=[
                        bi.BMQ(7, [round(k / 100, 2) for k in range(1, 100)]),
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
                    mi.MQTransformer([round(k / 100, 2) for k in range(1, 100)])] , help="methods to train")
    parser.add_argument("--train_ratio",default=0.8, help="train_ratio")
    parser.add_argument("--feature_transformation",default=ft.NoTransformationStrategy(), help="feature_transformation")
    parser.add_argument("--time_stationarization",default=ts.NoStationarizationStrategy(), help="time_stationarization")
    parser.add_argument("--datetime_features",default = [tc.Hour(),tc.Month(),tc.Day(),tc.Weekday()], help="datetime_features")
    parser.add_argument("--target_lag_selection",default=fls.ManualStrategy(lags=[24, 48, 72, 96, 120, 144, 168]), help="target_lag_selection")
    parser.add_argument("--external_feature_selection",default=fes.ZeroLagStrategy(["airTemperature"]), help="external_feature_selection")
    parser.add_argument("--evaluation_metrics",default=
                        [em.ReliabilityMatrix([round(k / 100, 2) for k in range(1, 100)]),
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
                        em.QuartileDispersion()], help="evaluation_metrics")
    parser.add_argument("--post_processing_quantile",default=ppq.QuantileSortingStrategy(), help="post_processing_quantile")
    parser.add_argument("--post_processing_value",default=ppv.ValueClippingStrategy(0, None), help="post_processing_value")
    parser.add_argument("--save_results",default=True, help="save_results")
    args = parser.parse_args()
    main(args)

