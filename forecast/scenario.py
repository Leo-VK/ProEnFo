import datetime as dt
from typing import Union

import pandas as pd

import feature.time_categorical as tc
import feature.feature_external_selection as fes
import models.benchmark_init as bi
import models.model_init as mi
from evaluation import calculation
from evaluation.metrics import ErrorMetric
from feature.feature_lag_selection import FeatureLagSelectionStrategy
from feature.feature_transformation import FeatureTransformationStrategy
from feature.time_lag import lag_target, remove_lag_interval
from feature.time_stationarization import TimeStationarizationStrategy
from feature.transformation_chain import apply_transformations_if_requested, invert_transformations_if_requested
from forecast import batch_learning
from postprocessing.quantile import PostProcessingQuantileStrategy
from postprocessing.value import PostProcessingValueStrategy
from preprocessing.data_format import check_missing_values, check_dublicated_columns
from preprocessing.data_format import check_datetimeindex, check_data_feature_alignment
from preprocessing.data_split import split_train_test_set
from utils.timer import Timer
import torch.nn as nn

from typing import List, Tuple, Dict

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
import torch


def calculate_scenario(data: pd.DataFrame,
                       target: str,
                       methods_to_train: List[Union[bi.Benchmark, mi.QuantileRegressor, mi.MultiQuantileRegressor]],
                       horizon: int = 24,
                       train_ratio: float = 0.8,
                       feature_transformation: FeatureTransformationStrategy = ft.NoTransformationStrategy(),
                       time_stationarization: TimeStationarizationStrategy = ts.NoStationarizationStrategy(),
                       datetime_features: List[tc.TimeCategorical] = [tc.Hour(),tc.Month(),tc.Day(),tc.Weekday()],
                       target_lag_selection: FeatureLagSelectionStrategy = fls.ManualStrategy(lags=[24, 48, 72, 96, 120, 144, 168]),
                       external_feature_selection: fes.FeatureExternalSelectionStrategy = fes.NoExternalSelectionStrategy(),
                       post_processing_quantile: PostProcessingQuantileStrategy = ppq.QuantileSortingStrategy(),
                       post_processing_value: PostProcessingValueStrategy = ppv.ValueClippingStrategy(0, None),
                       evaluation_metrics: List[ErrorMetric] = [em.ReliabilityMatrix([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.CalibrationMatrix([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.WinklerScoreMatrix([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.IntervalWidthMatrix([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                    #   em.RampScoreMatrix(threshold=0.1, normalize=True),
                      em.PinballLossMatrix([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.QuantileCrossingMatrix([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.PinballLoss([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.CalibrationError([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.BoundaryCrossing(0, None),
                      em.QuantileCrossing([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                      em.Skewness(),
                      em.Kurtosis(),
                      em.QuartileDispersion()],
                       device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'),
                       prob_forecasting = True,
                       strategy='default',
                       data_name = 'Unnamed',
                       ) -> Tuple[Dict[str, Dict[str, pd.Series]], Dict[str, pd.DataFrame]]:
    """Basic probabilistic forecasting scenario"""

    # Start time measurement
    print('Run time start:', dt.datetime.now())
    script_timer = Timer()
    script_timer.start()

    # Check format
    check_datetimeindex(data)
    check_data_feature_alignment(data, target, external_feature_selection.external_names)
    check_missing_values(data)

    ###########################
    ### Feature engineering ###
    ###########################
    # Copy target data to feature matrix
    X = data[[target]]

    # Transform data
    X, data_for_selection = apply_transformations_if_requested(X, (feature_transformation, time_stationarization))

    # Lag features and target
    target_lags = target_lag_selection.select_features(data_for_selection[target], horizon)

    if target_lags:
        X = lag_target(X, target, target_lags)
    
    if external_feature_selection.name =='taosvanilla':
        # Exact tao's feature
        external_features, external_lags = external_feature_selection.select_features(data, horizon)
        # Add external feature
        X = fes.add_modified_external_features(X, external_features)
        external_features_diminsion = external_features.shape[1]

    else:
        # Add categorical features
        if datetime_features:
            X = tc.add_datetime_features_with_lag(X, datetime_features,target_lags)
            external_features_diminsion = len(datetime_features)
        # Add modified external features
        external_features, external_lags = external_feature_selection.select_features(data, horizon)
        if not external_features.empty:
            X = fes.add_modified_external_features(X, external_features)
            external_features_diminsion = external_features.shape[1]+len(datetime_features)
        
    
    # Delete lag interval
    if target_lags or not external_features.empty:
        X = remove_lag_interval(X, horizon, target_lags, external_lags)
    # Check feature matrix
    check_missing_values(X)

    # if external_feature_selection.name =='taosvanilla':
    #     if X.columns.duplicated().sum() > 0:
    #         raise ValueError("There are duplicated columns")
    # else:
    #     check_dublicated_columns(X)
    if X.columns.duplicated().sum() > 0:
        raise ValueError("There are duplicated columns")

    # Construct train and test set
    X_train, X_test, Y_train, Y_test = split_train_test_set(X, target, train_ratio)

    if prob_forecasting:
        ##############################
        ### Model Training/Testing ###
        ##############################
        strategy_name = [feature_transformation.name,time_stationarization.name,target_lag_selection.name,external_feature_selection.name,post_processing_quantile.name,post_processing_value.name]
        forecasts, performances = {}, {}
        for method in methods_to_train:
            perf_timer = Timer()
            perf_timer.start()
            if isinstance(method, mi.QuantileRegressor):
                if external_feature_selection.name =='noexternalselection':
                    external_features_diminsion = len(datetime_features)
                forecasts[method.name] = batch_learning.quantile_forecasting(X_train, Y_train, X_test, method,external_features_diminsion)
            elif isinstance(method, mi.MultiQuantileRegressor):
                if external_feature_selection.name =='noexternalselection':
                    external_features_diminsion = len(datetime_features)
                forecasts[method.name] = batch_learning.multi_quantile_forecasting(X_train, Y_train, X_test, method,external_features_diminsion = external_features_diminsion,target_lags = target_lags,strategy_name=strategy,data_name=data_name,device=device)
            elif isinstance(method, bi.Benchmark):
                forecasts[method.name] = method.model.build_benchmark(Y_train, Y_test, horizon)
            else:
                raise ValueError("Method not recognized")
            performances[method.name] = perf_timer.stop()

        ########################
        ### Model Evaluation ###
        ########################
        # Post-process
        for method in methods_to_train:
            forecasts[method.name] = invert_transformations_if_requested(forecasts[method.name],
                                                                        target,
                                                                        (feature_transformation, time_stationarization))
            forecasts[method.name] = post_processing_quantile.process_prediction(forecasts[method.name])
            forecasts[method.name] = post_processing_value.process_prediction(forecasts[method.name])
        # Calculate errors
        errors = calculation.probabilistic_evaluation(data.loc[Y_test.index, target], forecasts, evaluation_metrics)

        # Store performances
        for method in methods_to_train:
            errors[method.name]['Performance'] = performances[method.name]

        # Check run time
        print('Run time end:', dt.datetime.now())
        print('Run time duration [min]:', round((script_timer.stop()) / 60, 2))

        return errors, forecasts,data.loc[Y_test.index, target]   
    
    else:
        ##############################
        ### Model Training/Testing ###
        ##############################
        strategy_name = [feature_transformation.name,time_stationarization.name,target_lag_selection.name,external_feature_selection.name]
        forecasts, performances = {}, {}
        for method in methods_to_train:
            perf_timer = Timer()
            perf_timer.start()
            if isinstance(method, mi.PointRegressor):
                if external_feature_selection.name =='noexternalselection':
                    external_features_diminsion = len(datetime_features)
                forecasts[method.name] = batch_learning.point_forecasting(X_train, Y_train, X_test, method,external_features_diminsion = external_features_diminsion,target_lags = target_lags,strategy_name=strategy,data_name=data_name,device=device)
            else:
                raise ValueError("Method not recognized")
            performances[method.name] = perf_timer.stop()

        ########################
        ### Model Evaluation ###
        ########################
        for method in methods_to_train:
            forecasts[method.name] = invert_transformations_if_requested(forecasts[method.name],
                                                                        target,
                                                                        (feature_transformation, time_stationarization))
        # Calculate errors    
        errors = calculation.point_evaluation(data.loc[Y_test.index, target], forecasts, evaluation_metrics)
        # Store performances
        for method in methods_to_train:
            errors[method.name]['Performance'] = performances[method.name]
            # print(performances[method.name])

        # Check run time
        print('Run time end:', dt.datetime.now())
        print('Run time duration [min]:', round((script_timer.stop()) / 60, 2))

        return errors, forecasts,data.loc[Y_test.index, target]    

