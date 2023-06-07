import os

import numpy as np
from torch import save, Tensor


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Taken from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=10, verbose=False, delta=1e-4, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease"""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def clean_up_checkpoint(self) -> bool:
        os.remove(os.path.realpath(f"{self.path}"))
        return True


class PinballScore:
    """Pinball loss averaged over samples"""

    def __init__(self, quantile: float = 0.5):
        self.quantile = quantile

    def __call__(self, y_pred: Tensor, y: Tensor):
        error = y_pred - y
        quantile_coef = (error > 0).float() - self.quantile
        return (error * quantile_coef).mean()


class PinballLoss:
    """Pinball loss averaged over quantiles and samples"""

    def __init__(self, quantiles: list[float]):
        self.quantiles = Tensor(quantiles)

    def __call__(self, y_pred: Tensor, y: Tensor):
        error = y_pred.sub(y)
        quantile_coef = (error > 0).float().sub(self.quantiles)
        return (error * quantile_coef).mean()
