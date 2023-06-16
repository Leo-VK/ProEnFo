import pandas as pd

from evaluation.metrics import ErrorMetric

from typing import List, Dict
import numpy as np


def probabilistic_evaluation(y_true: pd.Series,
                             forecasts: Dict[str, pd.DataFrame],
                             metrics: List[ErrorMetric]) -> Dict[str, Dict[str, pd.Series]]:
    """Calculate probabilistic metrics between true value and forecasts"""
    errors = {}
    for model in forecasts:

        # Extract prediction
        y_pred = forecasts[model]
        # Calculate error metrics
        errors[model] = {}
        for metric in metrics:
            name = metric.__class__.__name__
            errors[model][f"{name}"] = metric.calculate_mean_error(y_true, y_pred)
            errors[model][f"Instant_{name}"] = metric.calculate_instant_error(y_true, y_pred)

    return errors




def point_evaluation(y_true: pd.Series,
                             forecasts: Dict[str, pd.DataFrame],
                             metrics: List[ErrorMetric]) -> Dict[str, Dict[str, pd.Series]]:
    """Calculate point metrics between true value and forecasts"""
    errors = {}
    for model in forecasts:

        # Extract prediction
        y_pred = np.array(forecasts[model])
        y_pred = y_pred.reshape(-1)
        # Calculate error metrics
        errors[model] = {}
        for metric in metrics:
            name = metric.__class__.__name__
            errors[model][f"{name}"] = metric.calculate_mean_error(y_true, y_pred)
            if name not in ['RMSE','MASE']:
                errors[model][f"Instant_{name}"] = metric.calculate_instant_error(y_true, y_pred)

    return errors
