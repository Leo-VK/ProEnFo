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


def calculate_scenario(data: pd.DataFrame,
                       target: str,
                       methods_to_train: list[Union[bi.Benchmark, mi.QuantileRegressor, mi.MultiQuantileRegressor]],
                       horizon: int,
                       train_ratio: float,
                       feature_transformation: FeatureTransformationStrategy,
                       time_stationarization: TimeStationarizationStrategy,
                       datetime_features: list[tc.TimeCategorical],
                       target_lag_selection: FeatureLagSelectionStrategy,
                       external_feature_selection: fes.FeatureExternalSelectionStrategy,
                       post_processing_quantile: PostProcessingQuantileStrategy,
                       post_processing_value: PostProcessingValueStrategy,
                       evaluation_metrics: list[ErrorMetric]) -> tuple[dict[str, dict[str, pd.Series]], dict[str, pd.DataFrame]]:
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

    # Add categorical features
    if datetime_features:
        X = tc.add_datetime_features(X, datetime_features)

    # Add modified external features
    external_features, external_lags = external_feature_selection.select_features(data, horizon)
    if not external_features.empty:
        X = fes.add_modified_external_features(X, external_features)

    # Delete lag interval
    if target_lags or not external_features.empty:
        X = remove_lag_interval(X, horizon, target_lags, external_lags)

    # Check feature matrix
    check_missing_values(X)
    check_dublicated_columns(X)

    # Construct train and test set
    X_train, X_test, Y_train, Y_test = split_train_test_set(X, target, train_ratio)

    ##############################
    ### Model Training/Testing ###
    ##############################
    forecasts, performances = {}, {}
    for method in methods_to_train:
        perf_timer = Timer()
        perf_timer.start()
        if isinstance(method, mi.QuantileRegressor):
            forecasts[method.name] = batch_learning.quantile_forecasting(X_train, Y_train, X_test, method)
        elif isinstance(method, mi.MultiQuantileRegressor):
            forecasts[method.name] = batch_learning.multi_quantile_forecasting(X_train, Y_train, X_test, method)
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

    return errors, forecasts
