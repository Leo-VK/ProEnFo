from typing import Optional, Literal

import cvxpy as cp
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from typing import List


class QuantileOptimizationModel:
    """Class representing quantile optimization model"""
    def __init__(self, quantile: float, solver: Literal["ECOS", "OSQP", "SCS"]):
        self.quantile = quantile
        self.solver = solver
        self.weights: np.ndarray


class QuantileLinearProgram(QuantileOptimizationModel, BaseEstimator, RegressorMixin):
    """Quantile Linear Program ("Quantile Regression", R. Koenkar)"""

    def __init__(self,
                 quantile: float,
                 fit_intercept: bool = True,
                 solver: Literal["ECOS", "OSQP", "SCS"] = 'ECOS_BB'):
        super().__init__(quantile, solver)
        self.fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            X = sm.add_constant(X)
        y = y.reshape(-1, 1)
        n_measures, n_features = X.shape
        U, V = cp.Variable((n_measures, 1), nonneg=True), cp.Variable((n_measures, 1), nonneg=True)
        weights = cp.Variable((n_features, 1))

        eye = np.ones((n_measures, 1))
        constraint_error = (y - X @ weights == U - V)

        forecast_lp = cp.Problem(cp.Minimize(self.quantile * eye.T @ U + (1 - self.quantile) * eye.T @ V),
                                 [constraint_error])
        forecast_lp.solve(solver=self.solver)
        self.weights = np.squeeze(weights.value)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = sm.add_constant(X)
        return X @ self.weights


class QuantileSupportVectorRegression(QuantileOptimizationModel, BaseEstimator, RegressorMixin):
    """Quantile Support Vector Regression
    Standard variant for eps = 0 ("Nonparametric Quantile Estimation", I. Takeuchi et al.)
    Sparse variant ("Support Vector Quantile Regression Using Asymmetric e-insensitive loss function", K. Seok et al)
    Epsilon variant ("A New Asymmetric e-insensitive pinball loss function Based Support Vector Quantile Regression Model, P. Anand et al.)
    """

    def __init__(self,
                 quantile: float,
                 eps: float = 0.,
                 C: float = 1.,
                 variant: Literal["sparse", "epsilon"] = "epsilon",
                 solver: Literal["ECOS", "OSQP", "SCS"] = 'ECOS_BB'):
        super().__init__(quantile, solver)
        self.eps = eps
        self.C = C
        self.variant = variant

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.reshape(-1, 1)
        n_measures, n_features = X.shape
        U, V = cp.Variable((n_measures, 1), nonneg=True), cp.Variable((n_measures, 1), nonneg=True)
        weights = cp.Variable((n_features, 1))
        if self.variant == "sparse":
            constraint_svm_1 = (y - X @ weights <= U + 1 - self.quantile ** 2 * self.eps)
            constraint_svm_2 = (X @ weights - y <= V + self.quantile - self.quantile * self.eps)
        elif self.variant == "epsilon":
            constraint_svm_1 = (y - X @ weights <= (1 - self.quantile) * self.eps + U)
            constraint_svm_2 = (X @ weights - y <= self.quantile * self.eps + V)
        else:
            raise ValueError("Variant unknown")

        forecast_lp = cp.Problem \
            (cp.Minimize(0.5 * cp.sum_squares(weights) + self.C * cp.sum(self.quantile * U + (1 - self.quantile) * V)),
             [constraint_svm_1,
              constraint_svm_2])
        forecast_lp.solve(solver=self.solver)
        self.weights = np.squeeze(weights.value)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


class MultiQuantileOptimizationModel:
    """Class representing multi quantile optimization"""
    def __init__(self, quantiles: List[float], solver: Literal["ECOS", "OSQP", "SCS"]):
        self.quantiles = np.array(quantiles)
        self.solver = solver
        self.weights: np.ndarray


class MultiQuantileLinearProgram(MultiQuantileOptimizationModel, BaseEstimator, RegressorMixin):
    """Multi Quantile Linear Program ("Nonparametric Quantile Regression", I. Takeuchi)"""

    def __init__(self,
                 quantiles: Optional[List[float]],
                 fit_intercept: bool = True,
                 non_crossing: bool = False,
                 solver: Literal["ECOS", "OSQP", "SCS"] = 'ECOS_BB'):
        super().__init__(quantiles, solver)
        self.fit_intercept = fit_intercept
        self.non_crossing = non_crossing

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            X = sm.add_constant(X)
        n_measures, n_features = X.shape
        n_quantiles = len(self.quantiles)
        U, V = cp.Variable((n_measures, n_quantiles), nonneg=True), cp.Variable((n_measures, n_quantiles), nonneg=True)
        weights = cp.Variable((n_features, n_quantiles))

        eye = np.ones((n_measures, n_quantiles))
        constraints = [(y - X @ weights[:, i] == U[:, i] - V[:, i]) for i in range(n_quantiles)]
        if self.non_crossing:
            constraints += [(X[j, :] @ weights[:, i] - X[j, :] @ weights[:, i - 1] >= 0)
                            for i in range(1, n_quantiles) for j in range(n_measures)]

        forecast_lp = cp.Problem(
            cp.Minimize(cp.sum(cp.multiply(self.quantiles * eye, U) + cp.multiply((1 - self.quantiles) * eye, V))),
            constraints)
        forecast_lp.solve(solver=self.solver)
        self.weights = weights.value
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = sm.add_constant(X)
        preds = np.zeros((len(self.quantiles), len(X)))
        for q, quantile in enumerate(self.quantiles):
            preds[q, :] = X @ self.weights[:, q]
        return preds
