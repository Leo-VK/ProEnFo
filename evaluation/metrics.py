import itertools
from abc import ABC, abstractmethod
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd

from evaluation.weighting import uniform_sample_weighting, uniform_quantile_weighting
from preprocessing.quantile_format import check_prediction_interval, split_prediction_interval_symmetrically, \
    check_quantile_list
from utils.simulation import SwingingDoorCompression

from typing import List, Tuple




########################################
# Scores for Point Forecasting
########################################


class PointErrorMetric(ABC):
    """Class representing a point forecast error metric"""
    def calculate_mean_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            weights: Optional[pd.Series] = None) -> Union[float, pd.Series]:
        """Calculate mean error over time axis"""

class MAPE(PointErrorMetric):


    def calculate_instant_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        return np.abs((y_pred-y_true)/(y_true))

    def calculate_mean_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        
        return np.mean(self.calculate_instant_error(y_true,y_pred))
    

class MAE(PointErrorMetric):

    def calculate_instant_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        
        return np.abs((y_pred-y_true))
    
    def calculate_mean_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        
        return np.mean(self.calculate_instant_error(y_true,y_pred))
    


class RMSE(PointErrorMetric):
    
    def calculate_mean_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        
        return np.sqrt(np.mean(np.power(y_pred-y_true,2)))
    

class sMAPE(PointErrorMetric):

    def calculate_instant_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        
        return 2*np.abs((y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)))
    
    def calculate_mean_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        
        return np.mean(self.calculate_instant_error(y_true,y_pred))
    

class MASE(PointErrorMetric):

    def calculate_mean_error(self,
                            y_true: pd.Series,
                            y_pred: pd.DataFrame,
                            ) -> Union[float, pd.Series]:
        n = len(y_true)
        mase_numerator = np.mean(np.abs(y_pred - y_true))
        mase_denominator = np.mean(np.abs(np.array(y_true[1:]) - np.array(y_true[:-1])))
        mase = mase_numerator / mase_denominator
        return mase
        







########################################
# Scores for Probabilistic Forecasting(part)
########################################



class ErrorMetric(ABC):
    """Class representing a forecast error metric"""

    @abstractmethod
    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """Calculate error for each time step"""

    def calculate_mean_error(self,
                             y_true: pd.Series,
                             y_pred: pd.DataFrame,
                             weights: Optional[pd.Series] = None) -> Union[float, pd.Series]:
        """Calculate mean error over time axis"""
        weights = weights if weights is not None else uniform_sample_weighting(y_true)
        return self.calculate_instant_error(y_true, y_pred).multiply(weights, axis=0).div(weights.sum()).sum()

    def calculate_cumulative_error(self,
                                   y_true: pd.Series,
                                   y_pred: pd.DataFrame,
                                   weights: Optional[pd.Series] = None) -> pd.Series:
        """Calculate running mean error"""
        weights = weights if weights is not None else uniform_sample_weighting(y_true)
        return self.calculate_instant_error(y_true, y_pred).multiply(weights, axis=0).expanding(1).sum().div(
            weights.expanding(1).sum())


class AbsoluteError(ErrorMetric):
    """Absolute error"""

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        return y_pred.subtract(y_true).abs()


class AbsolutePercentageError(ErrorMetric):
    """Absolute percentage error"""

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        return y_pred.subtract(y_true).abs().divide(y_true).multiply(100)


class RootSquareError(ErrorMetric):
    """(Root) mean square error"""

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        return y_true.subtract(y_pred).pow(2)  # Be aware that we use the squared error here!

    def calculate_mean_error(self,
                             y_true: pd.Series,
                             y_pred: pd.Series,
                             weights: Optional[pd.Series] = None) -> float:
        weights = weights if weights is not None else uniform_sample_weighting(y_true)
        return np.sqrt(self.calculate_instant_error(y_true, y_pred).multiply(weights, axis=0).div(weights.sum()).sum())


class RampScore(ErrorMetric):
    """Ramp score metric
    ("Towards a standardized procedure to assess solar forecast accuracy: A new ramp and time alignment metric", L. Vallance et al.)
    ("Identifying Wind and Solar Ramping Events", A. Florita et al.)
    ("Swinging Door Trending Compression Algorithm for IoT Environments, J. Correa et al.)"""

    def __init__(self, threshold: float, fill_value: float = 0, use_datetime: bool = False, normalize: bool = False):
        self.threshold = threshold
        self.fill_value = fill_value
        self.use_datetime = use_datetime
        self.normalize = normalize

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        if self.normalize and (y_true.max() - y_true.min()) == 0:
            raise ValueError("Division by zero during scaling")
        y_true_norm = (y_true - y_true.min()) / (y_true.max() - y_true.min()) if self.normalize else y_true
        y_pred_norm = (y_pred - y_true.min()) / (y_true.max() - y_true.min()) if self.normalize else y_pred
        compressor = SwingingDoorCompression(self.threshold, self.use_datetime)
        slope_true = self._extract_slopes(compressor.compress(y_true_norm))
        slope_pred = self._extract_slopes(compressor.compress(y_pred_norm))
        return slope_true.subtract(slope_pred).abs()

    def _extract_slopes(self, series: pd.Series):
        valid_series = series[~series.isna()]
        denominator = valid_series.index.to_series().diff()
        if isinstance(denominator.index, pd.DatetimeIndex):
            if self.use_datetime:
                denominator = denominator.dt.total_seconds()
            else:
                denominator = pd.Series(np.arange(len(series)), index=series.index)[~series.isna()].diff()
        valid_slopes = valid_series.diff() / denominator
        all_slopes = valid_slopes.reindex(series.index)  # fill missing indices
        all_slopes = all_slopes.bfill()  # propagate slopes
        all_slopes.iloc[0] = np.nan  # first slope is undefined
        return all_slopes.fillna(self.fill_value)  # fill undefined slopes at beginning and end

class Consistent_quantile(ErrorMetric):
    """Extreme_quantile metric
       This only 
    """
    def __init__(self, threshold: float, fill_value: float = 0, use_datetime: bool = False, normalize: bool = False):
        self.threshold = threshold
        self.fill_value = fill_value
        self.use_datetime = use_datetime
        self.normalize = normalize

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        if self.normalize and (y_true.max() - y_true.min()) == 0:
            raise ValueError("Division by zero during scaling")
        y_true_norm = (y_true - y_true.min()) / (y_true.max() - y_true.min()) if self.normalize else y_true
        y_pred_norm = (y_pred - y_true.min()) / (y_true.max() - y_true.min()) if self.normalize else y_pred
        compressor = SwingingDoorCompression(self.threshold, self.use_datetime)
        slope_true = self._extract_slopes(compressor.compress(y_true_norm))
        slope_pred = self._extract_slopes(compressor.compress(y_pred_norm))
        return slope_true.subtract(slope_pred).abs()

    def _extract_slopes(self, series: pd.Series):
        valid_series = series[~series.isna()]
        denominator = valid_series.index.to_series().diff()
        if isinstance(denominator.index, pd.DatetimeIndex):
            if self.use_datetime:
                denominator = denominator.dt.total_seconds()
            else:
                denominator = pd.Series(np.arange(len(series)), index=series.index)[~series.isna()].diff()
        valid_slopes = valid_series.diff() / denominator
        all_slopes = valid_slopes.reindex(series.index)  # fill missing indices
        all_slopes = all_slopes.bfill()  # propagate slopes
        all_slopes.iloc[0] = np.nan  # first slope is undefined
        return all_slopes.fillna(self.fill_value)  # fill undefined slopes at beginning and end






########################################
# Scores for Probabilistic Forecasting
########################################
class PinballLoss(ErrorMetric):
    """Pinball loss, also known as check score or quantile loss. This is a proper scoring rule"""

    def __init__(self,
                 quantiles: List[float],
                 weights: Optional[pd.Series] = None):
        self.quantiles = check_quantile_list(quantiles)
        self.weights = weights if weights is not None else uniform_quantile_weighting(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        error = y_pred.sub(y_true, axis='index')
        quantile_coef = (error > 0).sub(pd.Series(self.quantiles, index=self.quantiles), axis='columns')
        return (error * quantile_coef).multiply(self.weights, axis=1).mean(axis=1)


class Skewness(ErrorMetric):
    """Skewness based on quantiles ("Measuring Skewness and Kurtosis", R. Groeneveld et al.)"""

    def __init__(self, variant: Literal["Bowley", "Kelly", "Hinkley"] = "Bowley"):
        self.calc_variants = {'Bowley': 0.7, 'Kelly': 0.9, 'Hinkley': 0.9}
        self.lower_bound = round(1 - self.calc_variants[variant], 2)
        self.upper_bound = self.calc_variants[variant]

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        nominator = y_pred[self.upper_bound] + y_pred[self.lower_bound] - 2 * y_pred[0.5]
        return nominator / (y_pred[self.upper_bound] - y_pred[self.lower_bound])


class Kurtosis(ErrorMetric):
    """Kurtosis based on quantiles ("What Is Kurtosis?: An Influence Function Approach", D. Ruppert)"""

    def __init__(self, alpha: float = 0.1, beta: float = 0.3):
        self.alpha_lower_bound = alpha
        self.alpha_upper_bound = round(1 - alpha, 2)
        self.beta_lower_bound = beta
        self.beta_upper_bound = round(1 - beta, 2)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        nominator = y_pred[self.alpha_upper_bound] - y_pred[self.alpha_lower_bound]
        return nominator / (y_pred[self.beta_upper_bound] - y_pred[self.beta_lower_bound])


class QuartileDispersion(ErrorMetric):
    """Quartile coefficient of dispersion ("Confidence interval for a coefficient of quartile variation", D. Bonett)"""

    def __init__(self):
        self.lower_bound = 0.3
        self.upper_bound = 0.7

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        return (y_pred[self.upper_bound] - y_pred[self.lower_bound]) / (
                y_pred[self.upper_bound] + y_pred[self.lower_bound])


class WinklerScore(ErrorMetric):
    """Winkler score considering sharpness and reliability for symmetric prediction interval
    ("A Decision-Theoretic Approach to Interval Estimation", R. Winkler)"""

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        check_prediction_interval([lower_bound, upper_bound])

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        y_pred_lower_bound, y_pred_upper_bound = y_pred[self.lower_bound], y_pred[self.upper_bound]
        alpha = round(1 - (self.upper_bound - self.lower_bound), 2)
        interval_widths = y_pred_upper_bound - y_pred_lower_bound
        lower_error = (y_true < y_pred_lower_bound) * (2 * (y_pred_lower_bound - y_true) / alpha)
        upper_error = (y_true > y_pred_upper_bound) * (2 * (y_true - y_pred_upper_bound) / alpha)
        return interval_widths + lower_error + upper_error


class CoverageError(ErrorMetric):
    """Interval coverage error for symmetric prediction interval"""

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        check_prediction_interval([lower_bound, upper_bound])

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        pic = self.prediction_interval_coverage(y_true, y_pred)
        nominal_coverage = round(self.upper_bound - self.lower_bound, 2)
        return pic - nominal_coverage

    def calculate_mean_error(self,
                             y_true: pd.Series,
                             y_pred: pd.DataFrame,
                             weights: Optional[pd.Series] = None) -> float:
        pic = self.prediction_interval_coverage(y_true, y_pred)
        nominal_coverage = round(self.upper_bound - self.lower_bound, 2)
        return pic.mean() - nominal_coverage

    def prediction_interval_coverage(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        y_pred_lower_bound, y_pred_upper_bound = y_pred[self.lower_bound], y_pred[self.upper_bound]
        return (y_pred_lower_bound <= y_true) * (y_true <= y_pred_upper_bound)


class IntervalWidth(ErrorMetric):
    """Prediction interval width of symmetric prediction interval
    ("Probabilistic forecasts, calibration and sharpness", Gneiting et al.)"""

    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        check_prediction_interval([lower_bound, upper_bound])

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        return (y_pred[self.upper_bound] - y_pred[self.lower_bound]).abs()


class ContinuousRankedProbabilityScore(ErrorMetric):
    """Continuous Ranked Probability Score calculated in energy form
    ("Estimation of the Continuous Ranked Probability Score with Limited Information and
    Applications to Ensemble Weather Forecasts", M. Zamo et al.)
    Note: CRPS can also be approximated by 2*Mean(PinballLoss) ("CRPS Learning", J. Berrisch et al.)"""

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        expected_xy = y_pred.sub(y_true, axis='index').abs().mean(axis=1)
        expected_xx = np.zeros_like(y_true)
        for (_, xi), (_, xj) in itertools.combinations(y_pred.items(), 2):
            expected_xx += (xi - xj).abs()
        return expected_xy - (y_pred.columns.size ** -2) * expected_xx


class QuantileCrossing(ErrorMetric):
    """Quantile crossing metric measuring the length of overcrossing quantiles"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        return self.quantile_crossing_path(y_pred).mean(axis=1)

    def quantile_crossing_path(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        distance = {}
        for (qi, yi), (qj, yj) in itertools.permutations(y_pred.items(), 2):
            distance[(qi, qj)] = yi < yj if qj < qi else yi > yj
        return pd.DataFrame.from_dict(distance)


class BoundaryCrossing(ErrorMetric):
    """Metric measuring violation of one-sided or two-sided boundary crossings"""

    def __init__(self, lower_boundary: Optional[float] = None, upper_boundary: Optional[float] = None):
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        if self.lower_boundary is not None and self.upper_boundary is not None:
            crossings = (y_pred < self.lower_boundary) + (self.upper_boundary < y_pred)
        elif self.lower_boundary is not None:
            crossings = (y_pred < self.lower_boundary)
        elif self.upper_boundary is not None:
            crossings = (self.upper_boundary < y_pred)
        else:
            raise ValueError("At least one boundary must be defined")
        return crossings.mean(axis=1)


class CalibrationError(ErrorMetric):
    """Quantile error metric to measure calibration
     ("Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification", Y. Chung et al.)"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        return y_pred.ge(y_true, axis='index').mean(axis=1)  # We use weaker notion without subtracting quantiles here

    def calculate_mean_error(self,
                             y_true: pd.Series,
                             y_pred: pd.DataFrame,
                             weights: Optional[pd.Series] = None) -> Union[float, pd.Series]:
        average = y_pred.ge(y_true, axis='index').mean(axis='index')
        average_error = (average - pd.Series(self.quantiles, index=self.quantiles))
        return average_error.abs().mean()


class RampScoreMatrix(ErrorMetric):
    """Ramp score metric across quantile forecasts"""

    def __init__(self, threshold: float, fill_value: float = 0, use_datetime: bool = False, normalize: bool = False):
        self.threshold = threshold
        self.fill_value = fill_value
        self.use_datetime = use_datetime
        self.normalize = normalize

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.DataFrame:
        metric = RampScore(self.threshold, self.fill_value, self.use_datetime, self.normalize)
        ramp_score_vec = {}
        for q, y_pred_i in y_pred.items():
            ramp_score_vec[q] = metric.calculate_instant_error(y_true, y_pred_i)
        return pd.DataFrame.from_dict(ramp_score_vec)


class ReliabilityMatrix(ErrorMetric):
    """Prediction interval coverage for symmetric prediction intervals. This can be used for a reliability diagram"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.DataFrame:
        quantile_series = pd.Series(self.quantiles).sort_values()
        lower_bounds, upper_bounds = split_prediction_interval_symmetrically(quantile_series[quantile_series < 0.5],
                                                                             quantile_series[quantile_series > 0.5])
        reliability_vec = {}
        for l, u in zip(lower_bounds, upper_bounds.sort_values(ascending=False)):
            nominal_coverage = round(u - l, 2)
            metric = CoverageError(l, u)
            reliability_vec[nominal_coverage] = metric.prediction_interval_coverage(y_true, y_pred[[l, u]])
        return pd.DataFrame.from_dict(reliability_vec)


class WinklerScoreMatrix(ErrorMetric):
    """Winkler score for symmetric prediction intervals. This can be plotted for a more comprehensive evaluation"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.DataFrame:
        quantile_series = pd.Series(self.quantiles).sort_values()
        lower_bounds, upper_bounds = split_prediction_interval_symmetrically(quantile_series[quantile_series < 0.5],
                                                                             quantile_series[quantile_series > 0.5])
        winkler_vec = {}
        for l, u in zip(lower_bounds, upper_bounds.sort_values(ascending=False)):
            nominal_coverage = round(u - l, 2)
            metric = WinklerScore(l, u)
            winkler_vec[nominal_coverage] = metric.calculate_instant_error(y_true, y_pred[[l, u]])
        return pd.DataFrame.from_dict(winkler_vec)


class IntervalWidthMatrix(ErrorMetric):
    """Interval width for symmetric prediction intervals. This can be plotted for a more comprehensive evaluation"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.DataFrame:
        quantile_series = pd.Series(self.quantiles).sort_values()
        lower_bounds, upper_bounds = split_prediction_interval_symmetrically(quantile_series[quantile_series < 0.5],
                                                                             quantile_series[quantile_series > 0.5])
        interval_vec = {}
        for l, u in zip(lower_bounds, upper_bounds.sort_values(ascending=False)):
            nominal_coverage = round(u - l, 2)
            metric = IntervalWidth(l, u)
            interval_vec[nominal_coverage] = metric.calculate_instant_error(y_true, y_pred[[l, u]])
        return pd.DataFrame.from_dict(interval_vec)


class CalibrationMatrix(ErrorMetric):
    """Calibration for given quantiles. This can be plotted for a more comprehensive evaluation"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.DataFrame:
        return y_pred.ge(y_true, axis='index')


class QuantileCrossingMatrix(ErrorMetric):
    """Average quantile crossing lengths for each quantile. This can be plotted for a more comprehensive evaluation"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.DataFrame:
        metric = QuantileCrossing(self.quantiles)
        crossing_path = metric.quantile_crossing_path(y_pred)
        crossing_path.columns = crossing_path.columns.set_names(['target', 'others'])
        return crossing_path.groupby(level='target', axis=1).mean()


class PinballLossMatrix(ErrorMetric):
    """Average Pinball loss for each quantile. This can be plotted for a more comprehensive evaluation"""

    def __init__(self, quantiles: List[float]):
        self.quantiles = check_quantile_list(quantiles)

    def calculate_instant_error(self, y_true: pd.Series, y_pred: pd.DataFrame) -> pd.Series:
        error = y_pred.sub(y_true, axis='index')
        quantile_coef = (error > 0).sub(pd.Series(self.quantiles, index=self.quantiles), axis='columns')
        return error * quantile_coef
