import numpy as np
from sklearn.metrics import mean_pinball_loss


def pinball_score(y_true, y_pred, **kwargs):
    """Pinball score to convert into a scorer through: make_scorer(pinball_score, quantile=q)"""
    diff = y_pred - y_true
    return np.mean(((diff > 0) - kwargs['quantile']) * diff)


def pinball_loss(y_true, y_pred, **kwargs):
    """Pinball loss to convert into a scorer through: make_scorer(pinball_loss, quantiles=[q_1, ..., q_Q])"""
    quantiles = kwargs['quantiles']
    quantiles_length = len(quantiles)
    if len(y_pred) == quantiles_length:
        y_pred = y_pred.T
    diff = y_pred - np.tile(y_true, (quantiles_length, 1)).T
    return np.mean(((diff > 0) - quantiles) * diff)
