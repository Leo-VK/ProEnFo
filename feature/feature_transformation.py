from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox, exp10
from scipy.stats import boxcox


class FeatureTransformationStrategy(ABC):
    """Class representing a transformation function for dataframes"""

    def __init__(self, apply_forecast: bool = False):
        self.name = self.__class__.__name__.replace("Strategy", "").lower()
        self.apply_forecast = apply_forecast

    @abstractmethod
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def inverse_transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class NoTransformationStrategy(FeatureTransformationStrategy):
    """Default transformation strategy with no transformation"""

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def inverse_transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


class BoxCoxStrategy(FeatureTransformationStrategy):
    """Boxcox transformation to stabilize variance for non-normal distribution"""

    def __init__(self, apply_forecast: bool = False):
        super().__init__(apply_forecast)
        self.lmbdas: dict[str, float]
        self.shift: dict[str, float]

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        lmbdas, shifts = {}, {}
        for column in data.columns:
            shift = 1 if (data[column] == 0).sum() > 0 else 0
            transformed_column, lmbda = boxcox(data[column].add(shift))
            data.loc[:, column] = transformed_column
            lmbdas[column] = lmbda
            shifts[column] = shift
        self.lmbdas = lmbdas
        self.shifts = shifts
        return data

    def inverse_transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            transformed_column = inv_boxcox(data[column], self.lmbdas[column])
            data.loc[:, column] = transformed_column.sub(self.shifts[column])
        return data


class LogStrategy(FeatureTransformationStrategy):
    """
    Logarithm transformation to stabilize variance for non-normal distribution with
    bases log2, log10 or ln. Can also help to force the forecasts to be positive
    """

    def __init__(self, base: Literal['2', '10', 'exp'] = 'exp', apply_forecast: bool = False):
        super().__init__(apply_forecast)
        self.base = base

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.base == '2':
            return data.apply(np.log2)
        elif self.base == '10':
            return data.apply(np.log10)
        elif self.base == 'exp':
            return data.apply(np.log)
        else:
            raise ValueError("Base not supported")

    def inverse_transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.base == '2':
            return data.apply(np.exp2)
        elif self.base == '10':
            return data.apply(exp10)
        elif self.base == 'exp':
            return data.apply(np.exp)
        else:
            raise ValueError("Base not supported")


class LogitStrategy(FeatureTransformationStrategy):
    """
    Logit transformation that can help to force the forecasts to be in interval [a, ..., b]
    """

    def __init__(self,
                 left_bound: float,
                 right_bound: float,
                 base: Literal['2', '10', 'exp'] = 'exp',
                 apply_forecast: bool = False):
        super().__init__(apply_forecast)
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.base = base

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.apply(lambda x: (x - self.left_bound) / (self.right_bound - x))
        if self.base == '2':
            return data.apply(np.log2)
        elif self.base == '10':
            return data.apply(np.log10)
        elif self.base == 'exp':
            return data.apply(np.log)
        else:
            raise ValueError("Base not supported")

    def inverse_transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.base == '2':
            return data.apply(lambda x: (((self.right_bound - self.left_bound) * np.exp2(x)) / (1 - np.exp2(x))) + self.left_bound)
        elif self.base == '10':
            return data.apply(lambda x: (((self.right_bound - self.left_bound) * exp10(x)) / (1 - exp10(x))) + self.left_bound)
        elif self.base == 'exp':
            return data.apply(lambda x: (((self.right_bound - self.left_bound) * np.exp(x)) / (1 - np.exp(x))) + self.left_bound)
        else:
            raise ValueError("Base not supported")
