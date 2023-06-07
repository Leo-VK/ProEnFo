import pandas as pd

from models.model_init import QuantileRegressor, MultiQuantileRegressor


def quantile_forecasting(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         method: QuantileRegressor) -> pd.DataFrame:
    """Quantile forecasting workflow"""
    # Scale data if required
    if method.scaler:
        X_train_val = method.scaler.fit_transform(X_train.values)
        X_test_val = method.scaler.transform(X_test.values)
    else:
        X_train_val = X_train.values
        X_test_val = X_test.values
    y_train_val = y_train.values

    # Set params
    method.set_params(X_train.shape[1])

    # Initialize, train and test models
    preds = {}
    print(f"\n{method.name}")
    for q, quantile in enumerate(method.quantiles):
        method.model[q].fit(X_train_val, y_train_val)
        preds[quantile] = method.model[q].predict(X_test_val)
        print(f"\r q={quantile}", end="")

    return pd.DataFrame(preds, index=X_test.index)


def multi_quantile_forecasting(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_test: pd.DataFrame,
                               method: MultiQuantileRegressor) -> pd.DataFrame:
    """Multi-quantile forecasting workflow"""
    # Scale data if required
    if method.scaler:
        X_train_val = method.scaler.fit_transform(X_train.values)
        X_test_val = method.scaler.transform(X_test.values)
    else:
        X_train_val = X_train.values
        X_test_val = X_test.values
    y_train_val = y_train.values

    # Set params
    method.set_params(X_train.shape[1])

    # Initialize, train and test models
    print(method.name)
    method.model.fit(X_train_val, y_train_val)
    preds = method.model.predict(X_test_val)
    if len(preds) == len(method.quantiles):
        preds = preds.T

    return pd.DataFrame(preds, columns=method.quantiles, index=X_test.index)
