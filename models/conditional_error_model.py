from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from typing import List,Tuple,Dict


class ConditionalErrorQuantile(BaseEstimator, RegressorMixin):
    def __init__(self, model: Any = None, quantiles: Optional[List[float]] = None):
        self.model = LinearRegression() if model is None else model
        self.quantiles = quantiles
        self.quantile_levels: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        pred = self.model.predict(X)
        error = y - pred
        self.quantile_levels = np.quantile(error, self.quantiles)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.model.predict(X), (len(self.quantiles), 1)).T + self.quantile_levels


class BootstrapConditionalErrorQuantile(BaseEstimator, RegressorMixin):
    """Compute a prediction interval around the model's predictions with boostrap sampling
      Adopted from (https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/)"""
    def __init__(self,
                 model: Any = None,
                 sample_rounds: int = 3,
                 adjusted: bool = False,
                 quantiles: Optional[List[float]] = None):
        self.model = LinearRegression() if model is None else model
        self.quantiles = quantiles
        self.sample_rounds = sample_rounds  # recommended by authors is np.sqrt(n).astype(int)
        self.adjusted = adjusted
        self.X_train: np.ndarray
        self.y_train: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        nx, nt = self.X_train.shape[0], len(X)
        forecasts = []
        for i, xi in enumerate(X):
            # print(f"\r {i}/{nt}", end="")
            # Compute the prediction and the training residuals
            self.model.fit(self.X_train, self.y_train)
            preds = self.model.predict(self.X_train)
            train_residuals = self.y_train - preds

            # Compute boostrap samples and residuals
            bootstrap_preds, val_residuals = [], []
            for _ in range(self.sample_rounds):
                train_indices = np.random.choice(range(nx), size=nx, replace=True)
                val_indices = np.array([idx for idx in range(nx) if idx not in train_indices])
                self.model.fit(self.X_train[train_indices, :], self.y_train[train_indices])
                preds = self.model.predict(self.X_train[val_indices])
                val_residuals.append(self.y_train[val_indices] - preds)
                bootstrap_preds.append(self.model.predict(xi.reshape(1, -1)))
            bootstrap_preds -= np.mean(bootstrap_preds)

            # Enable comparison between error residuals
            val_residuals = np.quantile(np.concatenate(val_residuals), q=self.quantiles)
            train_residuals = np.quantile(train_residuals, q=self.quantiles)

            weight = 0.5
            if self.adjusted:
                # Compute the .632+ bootstrap estimate for the sample noise and bias
                no_information_error = np.mean(np.abs(np.random.permutation(self.y_train) - \
                                                      np.random.permutation(preds)))
                generalisation = np.abs(val_residuals.mean() - train_residuals.mean())
                no_information_val = np.abs(no_information_error - train_residuals)
                relative_overfitting_rate = np.mean(generalisation / no_information_val)
                weight = 0.632 / (1 - 0.368 * relative_overfitting_rate)

            # Construct C set and percentiles
            residuals = (1 - weight) * train_residuals + weight * val_residuals
            residuals = np.array([m + o for m in bootstrap_preds for o in residuals])
            forecast = self.model.predict(xi.reshape(1, -1)) + np.quantile(residuals, q=self.quantiles)
            forecasts.append(forecast)
        return np.array(forecasts)
