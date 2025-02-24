from typing import Any, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from torch import nn, optim, from_numpy, no_grad, load, zeros, unsqueeze
from torch.utils.data import DataLoader, TensorDataset

from utils.pytorchtools import EarlyStopping, PinballScore
import math
from math import sqrt
import torch.nn.functional as F
from tqdm import trange
from typing import List
from models.shift import DishTS,RevIN
from itertools import chain
from torch.utils.data import Dataset


class Gaussian_head(nn.Module):
    def __init__(self, c_in,c_out):
        super(Gaussian_head, self).__init__()
        self.mu = nn.Linear(c_in,c_out)
        self.presigma = nn.Linear(c_in,c_out)
        self.sigma = nn.Softplus()

    def forward(self, X):
        mu = self.mu(X)
        sigma = self.sigma(self.presigma(X))
        return mu,sigma

class StudentT_head(nn.Module):
    def __init__(self, c_in,c_out):
        super(StudentT_head, self).__init__()
        self.mu = nn.Linear(c_in,c_out)
        self.presigma = nn.Linear(c_in,c_out)
        self.sigma = nn.Softplus()
        self.prenu = nn.Linear(c_in,c_out)
        self.nu = nn.Softplus()

    def forward(self, X):
        mu = self.mu(X)
        sigma = self.sigma(self.presigma(X))
        nu = self.nu(self.prenu(X))+1
        return mu,sigma,nu
    
class Pinball_head(nn.Module):
    def __init__(self, c_in,c_out):
        super(Pinball_head, self).__init__()
        self.mu = nn.Linear(c_in,c_out)

    def forward(self, X):
        mu = self.mu(X)
        return mu