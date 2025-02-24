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



class ex_HT(nn.Module):
    def __init__(self, configs):
        super(ex_HT, self).__init__()
        self.device = configs.device
        self.net = nn.Linear(configs.c_out+configs.ht_ex_dim, configs.ex_c_out)
        self.initialize_to_zeros(self.net)
    
    def ht_transform(self,input_tensor):
        # Assuming input tensor shape is (batch, seq, dim)
        batch = input_tensor.shape[0]
        seq = input_tensor.shape[1]
        # where dim order is [hour, day, weekday, month, temp1, temp2, ..., tempN]

        # Extract features
        days = input_tensor[:, :, 0]
        days = days-1
        hours = input_tensor[:, :, 1]
        months = input_tensor[:, :, 2]
        months = months-1
        weekdays = input_tensor[:, :, 3]
        temps = input_tensor[:, :, 4:].float()

        # Apply one-hot encoding to categorical variables
        hours_onehot = torch.nn.functional.one_hot(hours.to(torch.int64), num_classes=24).to(self.device)
        days_onehot = torch.nn.functional.one_hot(days.to(torch.int64), num_classes=31).to(self.device)
        weekdays_onehot = torch.nn.functional.one_hot(weekdays.to(torch.int64), num_classes=7).to(self.device)
        months_onehot = torch.nn.functional.one_hot(months.to(torch.int64), num_classes=12).to(self.device)
        temps_squared = torch.pow(temps, 2)
        temps_cubed = torch.pow(temps, 3)

        temps = temps.unsqueeze(3)  # Add an extra dimension
        temps_squared = temps_squared.unsqueeze(3)
        temps_cubed = temps_cubed.unsqueeze(3)

        hours_onehot = hours_onehot.unsqueeze(2)
        months_onehot = months_onehot.unsqueeze(2)

        temps_hours = temps * hours_onehot
        temps_squared_hours = temps_squared * hours_onehot
        temps_cubed_hours = temps_cubed * hours_onehot

        temps_months = temps * months_onehot
        temps_squared_months = temps_squared * months_onehot
        temps_cubed_months = temps_cubed * months_onehot

        # Remove the extra dimension
        temps_hours = temps_hours.view(batch,seq,-1)
        temps_squared_hours = temps_squared_hours.view(batch,seq,-1)
        temps_cubed_hours = temps_cubed_hours.view(batch,seq,-1)

        temps_months = temps_months.view(batch,seq,-1)
        temps_squared_months = temps_squared_months.view(batch,seq,-1)
        temps_cubed_months = temps_cubed_months.view(batch,seq,-1)

        # Concatenate all features
        output_tensor = torch.cat([
            hours_onehot.squeeze(2), 
            days_onehot, 
            weekdays_onehot, 
            months_onehot.squeeze(2), 
            temps.squeeze(3), 
            temps_squared.squeeze(3), 
            temps_cubed.squeeze(3), 
            temps_hours, 
            temps_squared_hours, 
            temps_cubed_hours, 
            temps_months, 
            temps_squared_months, 
            temps_cubed_months], dim=2)

        return output_tensor
   
    def forward(self, X,X_ex):
        X_ex = self.ht_transform(X_ex)
        X_ex = X_ex.unsqueeze(-1).permute(0,1,3,2).repeat(1, 1,X.shape[-2],1)
        return self.net(torch.cat([X,X_ex],-1))+X
    def initialize_to_zeros(self,model):
        for param in model.parameters():
            if param.requires_grad:  # 确保只对可训练参数进行初始化
                nn.init.zeros_(param)

class ex_Linear(nn.Module):
    def __init__(self, configs):
        super(ex_Linear, self).__init__()
        self.net = nn.Linear(configs.c_out+configs.ex_dim, configs.ex_c_out)
        self.initialize_to_zeros(self.net)
   
    def forward(self, X,X_ex):
        X_ex = X_ex.unsqueeze(-1).permute(0,1,3,2).repeat(1, 1,X.shape[-2],1)
        return self.net(torch.cat([X,X_ex],-1))+X
    def initialize_to_zeros(self,model):
        for param in model.parameters():
            if param.requires_grad:  # 确保只对可训练参数进行初始化
                nn.init.zeros_(param)


class ex_MLP(nn.Module):
    def __init__(self, configs):
        super(ex_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(configs.c_out+configs.ex_dim, configs.ex_neurons),
            nn.ReLU(),
            nn.Linear(configs.ex_neurons, configs.ex_c_out))
        self.initialize_to_zeros(self.net)

    def forward(self, X,X_ex):
        X_ex = X_ex.unsqueeze(-1).permute(0,1,3,2).repeat(1, 1,X.shape[-2],1)
        return self.net(torch.cat([X,X_ex],-1))+X
    def initialize_to_zeros(self,model):
        for param in model.parameters():
            if param.requires_grad:  # 确保只对可训练参数进行初始化
                nn.init.zeros_(param)