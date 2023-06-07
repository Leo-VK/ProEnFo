import pandas as pd

from evaluation.metrics import ErrorMetric


def probabilistic_evaluation(y_true: pd.Series,
                             forecasts: dict[str, pd.DataFrame],
                             metrics: list[ErrorMetric]) -> dict[str, dict[str, pd.Series]]:
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
            errors[model][f"Instant{name}"] = metric.calculate_instant_error(y_true, y_pred)

    return errors
