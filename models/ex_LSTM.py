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


class ex_LSTM(nn.Module):
    def __init__(self, configs):
        super(ex_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=configs.c_in+configs.ex_dim, hidden_size=configs.d_model, num_layers=configs.n_layers, batch_first=True)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.c_in)
        )
        self.proj = nn.Linear(1,configs.ex_c_out)
        self.initialize_to_zeros(self.lstm)
        self.initialize_to_zeros(self.seq)
        self.initialize_to_zeros(self.proj)

    def forward(self, X,X_ex):
        inputs = torch.cat([X.mean(dim=-1),X_ex],-1)
        output,_ = self.lstm(inputs)
        return self.proj(self.seq(output).unsqueeze(-1))+X
    def initialize_to_zeros(self,model):
        for param in model.parameters():
            if param.requires_grad:  # 确保只对可训练参数进行初始化
                nn.init.zeros_(param)