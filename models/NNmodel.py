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
import pickle


class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.configs = configs
        self.name = self.__class__.__name__
        n_input = (configs.c_in)*configs.seq_len
        n_output = configs.ex_c_out*configs.pred_len*configs.c_in
        n_neurons = configs.d_model*4
        self.net = nn.Sequential(
            nn.Linear(n_input, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_output))

    def forward(self, X):
        X = X.view(X.shape[0],-1)
        X = self.net(X)
        X = X.view(X.shape[0],self.configs.pred_len,self.configs.c_in,-1)
        return X

class LSTMF(nn.Module):
    def __init__(self, configs):
        super(LSTMF, self).__init__()
        self.configs = configs
        self.name = self.__class__.__name__
        self.n_layers = configs.n_layers
        self.n_neurons = configs.d_model
        self.lstm = nn.LSTM(input_size=configs.c_in, hidden_size=configs.d_model, num_layers=self.n_layers, batch_first=True)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.c_in)
        )
        self.pred_length = configs.pred_len
        self.adjust = nn.Linear(configs.c_in,configs.c_in)
        self.proj = nn.Linear(1,configs.c_out)

    def forward(self, X) -> torch.Tensor:
        outputs = []
        h, c = self.init_hidden(X.size(0))
        
        for _ in range(self.pred_length):
            output, (h, c) = self.lstm(X, (h, c))
            prediction = self.seq(output[:, -1, :])
            outputs.append(prediction.unsqueeze(1) )
            X = prediction.unsqueeze(1)  # Use the prediction as the next input

        outputs =  self.adjust(torch.cat(outputs, dim=1)).unsqueeze(-1)
        outputs = self.proj(outputs)
        
        return outputs
    
    def init_hidden(self, batch_size: int):
        # Initialize hidden and cell states with zeros
        h = torch.zeros(self.n_layers, batch_size, self.n_neurons).to(self.configs.device)
        c = torch.zeros(self.n_layers, batch_size, self.n_neurons).to(self.configs.device)
        return h, c

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.configs = configs
        self.name = self.__class__.__name__
        self.c_input = configs.c_in
        self.c_output = configs.c_out
        self.conv1_out_channels = configs.d_model
        self.conv2_out_channels = configs.d_model

        self.net = nn.Sequential(
            nn.Conv1d(self.c_input, self.conv1_out_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(self.conv1_out_channels,  self.conv2_out_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(self.conv2_out_channels, configs.c_in, kernel_size=1, stride=1),
        )
        self.temporal_proj = nn.Linear(configs.seq_len,configs.pred_len)
        self.proj = nn.Linear(1,configs.c_out)

    def forward(self, X):
        X = X.permute(0, 2, 1)  # 将输入张量 X 的形状从 (batch, seq_len, c_input) 转换为 (batch, c_input, seq_len)
        X = self.net(X)
        X = X.permute(0, 2, 1)  # 将输出张量 X 的形状从 (batch, c_output, seq_len) 转换为 (batch, seq_len, c_output)
        X = self.temporal_proj(X.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.proj(X.unsqueeze(-1))
        return X
    
class LSTNet(nn.Module):
    def __init__(self, configs):
        super(LSTNet, self).__init__()
        self.name = self.__class__.__name__
        self.P = configs.seq_len
        self.pred_len = configs.pred_len
        self.m = configs.c_in
        self.c_out = configs.c_out
        self.hidR = configs.hidRNN
        self.hidC = configs.hidCNN
        self.hidS = configs.hidSkip
        self.Ck = configs.CNN_kernel
        self.skip = configs.skip
        self.pt = (self.P - self.Ck)//self.skip
        self.hw = configs.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = configs.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.pred_len)
        else:
            self.linear1 = nn.Linear(self.hidR, self.pred_len)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.adjust = nn.Linear(1,configs.c_in)
        self.proj = nn.Linear(1,configs.c_out)
    def forward(self, x):
        batch_size = x.size(0)
        
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
        res = self.proj(self.adjust(res.unsqueeze(-1)).unsqueeze(-1))
        return res



def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


class Block(nn.Module):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace = linear_space(backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space(backcast_length, forecast_length, is_forecast=True)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(
            self,
            configs,
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=3,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            nb_harmonics=None
    ):
        super(NBeatsNet, self).__init__()
        self.name = self.__class__.__name__
        self.forecast_length = configs.pred_len
        self.backcast_length = configs.seq_len
        self.hidden_layer_units = configs.d_model
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = [[] for _ in range(configs.c_in)]
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = configs.device
        for i in range(configs.c_in):
            for stack_id in range(len(self.stack_types)):
                self.stacks[i].append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []
        self.proj = nn.Linear(1,configs.c_out)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units, self.thetas_dim[stack_id],
                    self.device, self.backcast_length, self.forecast_length,
                    self.nb_harmonics
                )
                self.parameters.extend(block.parameters())
            blocks.append(block)
        return blocks


    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def forward(self, backcasts):
        self._intermediary_outputs = [[] for _ in range(backcasts.shape[-1])]
        forecasts = []
        for i in range(backcasts.shape[-1]):
            backcast = squeeze_last_dim(backcasts[:,:,i])
            forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
            for stack_id in range(len(self.stacks[i])):
                for block_id in range(len(self.stacks[i][stack_id])):
                    b, f = self.stacks[i][stack_id][block_id](backcast)
                    backcast = backcast.to(self.device) - b
                    forecast = forecast.to(self.device) + f
                    block_type = self.stacks[i][stack_id][block_id].__class__.__name__
                    layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
                    if self._gen_intermediate_outputs:
                        self._intermediary_outputs[i].append({'value': f.detach().numpy(), 'layer': layer_name})
            forecasts.append(forecast.unsqueeze(-1))
        forecasts = torch.cat(forecasts,-1)
        return self.proj(forecasts.unsqueeze(-1))



# class NBeatsNetX(nn.Module):
#     SEASONALITY_BLOCK = 'seasonality'
#     TREND_BLOCK = 'trend'
#     GENERIC_BLOCK = 'generic'

#     def __init__(
#             self,
#             configs,
#             stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
#             nb_blocks_per_stack=3,
#             thetas_dim=(4, 8),
#             share_weights_in_stack=False,
#             nb_harmonics=None
#     ):
#         super(NBeatsNetX, self).__init__()
#         self.name = self.__class__.__name__
#         self.forecast_length = configs.pred_len
#         self.backcast_length = configs.seq_len+configs.pred_len*configs.ex_dim
#         self.hidden_layer_units = configs.d_model
#         self.nb_blocks_per_stack = nb_blocks_per_stack
#         self.share_weights_in_stack = share_weights_in_stack
#         self.nb_harmonics = nb_harmonics
#         self.stack_types = stack_types
#         self.stacks = [[] for _ in range(configs.c_in)]
#         self.thetas_dim = thetas_dim
#         self.parameters = []
#         self.device = configs.device
#         for i in range(configs.c_in):
#             for stack_id in range(len(self.stack_types)):
#                 self.stacks[i].append(self.create_stack(stack_id))
#         self.parameters = nn.ParameterList(self.parameters)
#         self.to(self.device)
#         self._loss = None
#         self._opt = None
#         self._gen_intermediate_outputs = False
#         self._intermediary_outputs = []
#         self.proj = nn.Linear(1,configs.ex_c_out)

#     def create_stack(self, stack_id):
#         stack_type = self.stack_types[stack_id]
#         blocks = []
#         for block_id in range(self.nb_blocks_per_stack):
#             block_init = NBeatsNet.select_block(stack_type)
#             if self.share_weights_in_stack and block_id != 0:
#                 block = blocks[-1]  # pick up the last one when we share weights.
#             else:
#                 block = block_init(
#                     self.hidden_layer_units, self.thetas_dim[stack_id],
#                     self.device, self.backcast_length, self.forecast_length,
#                     self.nb_harmonics
#                 )
#                 self.parameters.extend(block.parameters())
#             blocks.append(block)
#         return blocks


#     @staticmethod
#     def select_block(block_type):
#         if block_type == NBeatsNet.SEASONALITY_BLOCK:
#             return SeasonalityBlock
#         elif block_type == NBeatsNet.TREND_BLOCK:
#             return TrendBlock
#         else:
#             return GenericBlock

#     def get_generic_and_interpretable_outputs(self):
#         g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
#         i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
#         outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
#         return g_pred, i_pred, outputs

#     def forward(self, backcasts,backcasts_ex):
#         self._intermediary_outputs = [[] for _ in range(backcasts.shape[-1])]
#         forecasts = []
#         for i in range(backcasts.shape[-1]):
#             backcast = squeeze_last_dim(backcasts[:,:,i])
#             backcast = torch.cat([backcast,backcasts_ex.reshape(backcast.shape[0],-1)],-1)
#             forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
#             for stack_id in range(len(self.stacks[i])):
#                 for block_id in range(len(self.stacks[i][stack_id])):
#                     b, f = self.stacks[i][stack_id][block_id](backcast)
#                     backcast = backcast.to(self.device) - b
#                     forecast = forecast.to(self.device) + f
#                     block_type = self.stacks[i][stack_id][block_id].__class__.__name__
#                     layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
#                     if self._gen_intermediate_outputs:
#                         self._intermediary_outputs[i].append({'value': f.detach().numpy(), 'layer': layer_name})
#             forecasts.append(forecast.unsqueeze(-1))
#         forecasts = torch.cat(forecasts,-1)
#         return self.proj(forecasts.unsqueeze(-1))


class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(CustomConv1d, self).__init__()
        k = np.sqrt(1 / (in_channels * kernel_size))
        weight_data = -k + 2 * k * torch.rand((out_channels, in_channels, kernel_size))
        bias_data = -k + 2 * k * torch.rand((out_channels))
        self.weight = nn.Parameter(weight_data, requires_grad=True)
        self.bias = nn.Parameter(bias_data, requires_grad=True)
        self.dilation = dilation
        self.padding = padding

    def forward(self, x):
        xp = F.pad(x, (self.padding, 0))
        return F.conv1d(xp, self.weight, self.bias, dilation=self.dilation)

class wavenet_cell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(wavenet_cell, self).__init__()
        self.conv_dil = CustomConv1d(in_channels, out_channels * 2, kernel_size, padding, dilation)
        self.conv_skipres = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, x):
        h_prev, skip_prev = x
        f, g = self.conv_dil(h_prev).chunk(2, 1)
        h_next, skip_next = self.conv_skipres(torch.tanh(f) * torch.sigmoid(g)).chunk(2, 1)
        
        return (h_prev + h_next, skip_prev + skip_next)

class WaveNet(nn.Module):
    def __init__(self, configs):
        super(WaveNet, self).__init__()
        self.name = self.__class__.__name__

        d_lag = configs.c_in
        d_cov = configs.ex_dim
        d_output = configs.c_in
        d_hidden = configs.d_model
        kernel_size = configs.kernel_size
        Nl = configs.Nl
        self.upscale = nn.Linear(d_lag + d_cov, d_hidden)
        # Wavenet
        wnet_layers = nn.ModuleList([wavenet_cell(
                    d_hidden, d_hidden, 
                    kernel_size, padding=(kernel_size-1) * 2**i, 
                    dilation = 2**i) for i in range(Nl)])  
        self.wnet = nn.Sequential(*wnet_layers)
        # Output layer
        self.loc = nn.Linear(d_hidden, d_output)
        self.proj = nn.Linear(1, configs.c_out)
        
    def forward(self, x_lag, x_cov, d_outputseqlen):       
        # Concatenate inputs
        dim_seq = x_lag.shape[0]
        h = torch.cat((x_lag, x_cov[:dim_seq]), dim=-1)        
        h = self.upscale(h)
        # Apply wavenet
        _, h = self.wnet((h.permute(1, 2, 0), 0))
        # Output layers - location & scale of the distribution
        output = h[:, :, -d_outputseqlen:].permute(2, 0, 1)
        loc = self.loc(output)
        return self.proj(loc.permute(1,0,2).unsqueeze(-1))