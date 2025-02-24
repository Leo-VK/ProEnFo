import pandas as pd

from feature.feature_transformation import FeatureTransformationStrategy
from feature.time_stationarization import TimeStationarizationStrategy

from typing import Tuple


def apply_transformations_if_requested(data: pd.DataFrame,
                                       strategies: Tuple[FeatureTransformationStrategy,
                                                         TimeStationarizationStrategy]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies feature transformations and stationarizations to original features for downstream processes if requested.
    Otherwise, it will only be used for feature selection"""
    transformed_data = data.copy()
    for strategy in strategies:
        if isinstance(strategy, FeatureTransformationStrategy):
            transformed_data = strategy.transform_data(transformed_data)
            if strategy.apply_forecast:
                data = strategy.transform_data(data)
        elif isinstance(strategy, TimeStationarizationStrategy):
            transformed_data = strategy.make_stationary(transformed_data)
            if strategy.apply_forecast:
                data = strategy.make_stationary(data)
        else:
            raise ValueError("Strategy unknown!")

    return data, transformed_data


def invert_transformations_if_requested(data: pd.DataFrame,
                                        target: str,
                                        strategies: Tuple[FeatureTransformationStrategy,
                                                          TimeStationarizationStrategy]) -> pd.DataFrame:
    """Applies inverted functions of feature transformations and stationarizations to forecast"""
    transformed_data = data.to_dict('series')
    for strategy in strategies[-1::-1]:
        if strategy.apply_forecast:
            for column in transformed_data.keys():
                if isinstance(strategy, FeatureTransformationStrategy):
                    transformed_data[column] = strategy.inverse_transform_data(transformed_data[column].to_frame().rename(columns={column: target}))[target]
                elif isinstance(strategy, TimeStationarizationStrategy):
                    transformed_data[column] = strategy.invert_stationary(transformed_data[column].to_frame().rename(columns={column: target}))[target]
                else:
                    raise ValueError("Strategy unknown!")
            data = pd.concat(transformed_data, axis=1)
    return data