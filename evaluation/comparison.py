import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import acovf
from typing import Tuple

class SkillScore:
    """Skill Score for two error metric values"""
    def calculate_score(self, reference_error: float, error: float) -> float:
        return 1 - error / reference_error


class DieboldMarianoTest:
    """Diebold Marianto Test for forecast comparison.
    Adapted from (https://github.com/johntwk/Diebold-Mariano-Test/blob/master/dm_test.py)"""
    def __init__(self, horizon: int, norm: int = 1, harvey_adjusted: bool = True):
        self.norm = norm
        self.horizon = horizon
        self.harvey_adjusted = harvey_adjusted

    def calculate_score(self, reference_error: pd.Series, error: pd.Series) -> Tuple[pd.Series, float]:
        if self.norm == 1:
            differential = reference_error.abs() - error.abs()
        if self.norm == 2:
            differential = reference_error.pow(2) - error.pow(2)
        else:
            raise ValueError("Norm not supported")

        N = len(differential)
        auto_covar = acovf(differential)
        statistics = differential.mean() / ((auto_covar.iloc[0] + 2 * auto_covar.iloc[1:self.horizon].sum()) / N) ** 0.5

        if self.harvey_adjusted:
            statistics *= ((N + 1 - 2 * self.horizon + self.horizon * (self.horizon - 1) / N) / N) ** 0.5
        p_value = 2 * stats.norm.cdf(-statistics.abs(), df=N - 1)  # Two-tailed test

        return statistics, p_value
