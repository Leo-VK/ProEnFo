from typing import List
from pygam import GAM, s, f
from pygam.distributions import Distribution
from sklearn.linear_model import QuantileRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
from pygam.links import Link, IdentityLink


def quantile_loss(y, y_pred, quantile=0.5):
    residual = y - y_pred
    return np.where(residual >= 0, quantile * residual, (quantile - 1) * residual).sum()

# class QuantileLossGAM(GAM):
#     def __init__(self, quantile: float, n_features: int, *args, **kwargs):
#         terms = s(0, n_splines=10)
#         for i in range(1, n_features):
#             terms += s(i, n_splines=10)
#         if 'distribution' not in kwargs:
#             kwargs['distribution'] = 'gamma'
#         if 'link' not in kwargs:
#             kwargs['link'] = 'identity'
#         super().__init__(terms=terms, *args, **kwargs)
#         self.quantile = quantile

#     def _loss(self, y, mu):
#         return np.where(y > mu, self.quantile * (y - mu), (1 - self.quantile) * (mu - y))

#     def _loglikelihood(self, y, mu, weights=None):
#         if weights is None:
#             weights = np.ones_like(y)
#         return -np.sum(weights * self._loss(y, mu))

#     def fit(self, X, y, *args, **kwargs):
#         return super().fit(X, y, *args, **kwargs)

# class QuantileGAMDistribution(Distribution):
#     def __init__(self, quantile: float):
#         super().__init__()
#         self.quantile = quantile

#     def __repr__(self):
#         return f'QuantileGAMDistribution(quantile={self.quantile})'

#     def log_pdf(self, y, mu, weights=None):
#         q = self.quantile
#         loss = np.where(y > mu, q * (y - mu), (1 - q) * (mu - y))
#         return -loss

#     def gradient(self, y, mu, weights=None):
#         q = self.quantile
#         return np.where(y > mu, q, -1 * (1 - q))

#     def _mu(self, lin_pred, weights=None):
#         return lin_pred

#     def _mu_grad(self, lin_pred, weights=None):
#         return np.ones_like(lin_pred)

#     def V(self, mu):
#         return np.ones_like(mu)

#     def deviance(self, y, mu, weights=None, scaled=False):
#         q = self.quantile
#         loss = np.where(y > mu, q * (y - mu), (1 - q) * (mu - y))
#         return loss

# class QuantileLossGAM(GAM):
#     def __init__(self, quantile: float, n_features: int, *args, **kwargs):
#         terms = s(0, n_splines=10)
#         for i in range(1, n_features):
#             terms += s(i, n_splines=10)
#         if 'distribution' not in kwargs:
#             kwargs['distribution'] = QuantileGAMDistribution(quantile=quantile)
#         if 'link' not in kwargs:
#             kwargs['link'] = IdentityLink()
#         super().__init__(terms=terms, *args, **kwargs)

# class QuantileLossGAM():
#     def __init__(self, quantile: float, n_features: int, *args, **kwargs):
#         terms = s(0, n_splines=10)
#         for i in range(1, n_features):
#             terms += s(i, n_splines=10)
#         self.gam = GAM(s(0, n_splines=10), 
#           link='identity', fit_intercept=True, 
#           callbacks=['deviance', 'diffs'])
#         self.gam._loss = lambda y, mu: quantile_loss(y, mu, quantile=quantile)

#     def fit(self, X, y, *args, **kwargs):
#         return self.gam.fit(X, y)
    
#     def predict(self, X, *args, **kwargs):
#         return self.gam.predict(X)


class QuantileLossGAM(GAM):
    def __init__(self, quantile, *args, **kwargs):
        self.quantile = quantile
        super().__init__(*args, **kwargs)

    def _loglikelihood(self, y, mu, weights=None):
        q = self.quantile
        residuals = y - mu
        loss = np.where(residuals > 0, q * residuals, (1 - q) * -residuals)
        if weights is not None:
            loss *= weights
        return -np.sum(loss)

    def _gradient(self, y, mu, weights=None):
        q = self.quantile
        residuals = y - mu
        grad = np.where(residuals > 0, q, -1 * (1 - q))
        if weights is not None:
            grad *= weights
        return -grad

