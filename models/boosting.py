import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator, RegressorMixin


# class CustomQuantileLossXGBRegressor(xgb.XGBRegressor):
#     def __init__(self, quantile, **kwargs):
#         super().__init__(**kwargs)
#         self.quantile = quantile

#     def fit(self, X, y, eval_set=None, **kwargs):
#         custom_quantile_loss = self.quantile_loss(self.quantile)
#         dtrain = xgb.DMatrix(X, label=y)
#         params = {
#             'objective': 'reg:squarederror',
#             'eval_metric': 'mae',
#             'max_depth': 3
#         }
#         self._Booster = xgb.train(
#             params,
#             dtrain,
#             num_boost_round=100,
#             early_stopping_rounds=10,
#             verbose_eval=False,
#             evals=[(dtrain, 'train')],
#             obj=lambda preds, dtrain: custom_quantile_loss(preds, dtrain, self.quantile)
#         )

#     def quantile_loss(preds, dtrain, alpha):
#         labels = dtrain.get_label()
#         errors = labels - preds
#         grad = np.where(errors >= 0, -alpha * errors, -(alpha - 1) * errors)
#         hess = np.where(errors >= 0, alpha, alpha - 1)
#         return grad, hess

class CustomQuantileLossXGBRegressor(xgb.XGBRegressor):
    def __init__(self, quantile, **kwargs):
        super().__init__(**kwargs)
        self.quantile = quantile

    def fit(self, X, y, eval_set=None, **kwargs):
        custom_quantile_loss = self.quantile_loss
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 10
        }
        self._Booster = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            early_stopping_rounds=10,
            verbose_eval=False,
            evals=[(dtrain, 'train')],
            obj=lambda preds, dtrain: custom_quantile_loss(preds, dtrain)
        )

    def quantile_loss(self, preds, dtrain):
        labels = dtrain.get_label()
        errors = labels - preds
        grad = np.where(errors >= 0, -self.quantile * errors, -(self.quantile - 1) * errors)
        hess = np.where(errors >= 0, self.quantile, self.quantile - 1)
        return grad, hess
