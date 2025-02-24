import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import Parameter
import numpy as np

# This implementation of causal conv is faster than using normal conv1d module
class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, mode='backward', groups=1):
        super(CustomConv1d, self).__init__()
        k = np.sqrt(1 / (in_channels * kernel_size))
        weight_data = -k + 2 * k * torch.rand((out_channels, in_channels // groups, kernel_size))
        bias_data = -k + 2 * k * torch.rand((out_channels))
        self.weight = Parameter(weight_data, requires_grad=True)
        self.bias = Parameter(bias_data, requires_grad=True)  
        self.dilation = dilation
        self.groups = groups
        if mode == 'backward':
            self.padding_left = padding
            self.padding_right= 0
        elif mode == 'forward':
            self.padding_left = 0
            self.padding_right= padding            

    def forward(self, x):
        xp = F.pad(x, (self.padding_left, self.padding_right))
        return F.conv1d(xp, self.weight, self.bias, dilation=self.dilation, groups=self.groups)

class tcn_cell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, mode, groups, dropout):
        super(tcn_cell, self).__init__()
        self.conv1 = weight_norm(CustomConv1d(in_channels, out_channels, kernel_size, padding, dilation, mode, groups))
        self.conv2 = weight_norm(CustomConv1d(out_channels, in_channels * 2, 1))
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        h_prev, out_prev = x
        h = self.drop(F.gelu(self.conv1(h_prev)))
        h_next, out_next = self.conv2(h).chunk(2, 1)
        return (h_prev + h_next, out_prev + out_next)

class BiTCN(nn.Module):
    def __init__(self, configs):
        super(BiTCN, self).__init__()
        # Embedding layer for time series ID
        self.name = self.__class__.__name__
        d_lag = configs.c_in
        d_cov = configs.ex_dim
        d_output = configs.c_in
        d_hidden = configs.d_model
        dropout = configs.dropout
        kernel_size = configs.kernel_size
        Nl = configs.Nl
        self.upscale_lag = nn.Linear(d_lag  + d_cov, d_hidden)
        self.upscale_cov = nn.Linear(d_cov, d_hidden)
        self.drop_lag = nn.Dropout(dropout)
        self.drop_cov = nn.Dropout(dropout)
        # tcn
        layers_fwd = nn.ModuleList([tcn_cell(
                    d_hidden, d_hidden * 4, 
                    kernel_size, padding=(kernel_size-1)*2**i, 
                    dilation=2**i, mode='forward', 
                    groups=d_hidden, 
                    dropout=dropout) for i in range(Nl)])  
        layers_bwd = nn.ModuleList([tcn_cell(
                    d_hidden, d_hidden * 4, 
                    kernel_size, padding=(kernel_size-1)*2**i, 
                    dilation=2**i, mode='backward', 
                    groups=1, 
                    dropout=dropout) for i in range(Nl)])
        self.net_fwd = nn.Sequential(*layers_fwd)
        self.net_bwd = nn.Sequential(*layers_bwd)
        # Output layer
        self.loc = nn.Linear(d_hidden*2, d_output)
        self.proj = nn.Linear(1, configs.ex_c_out)
        
    def forward(self, x_lag, x_cov, d_outputseqlen):       
        # Concatenate inputs
        dim_seq = x_lag.shape[0]
        h_cov = x_cov
        h_lag = torch.cat((x_lag, h_cov[:dim_seq]), dim=-1)
        h_lag = self.drop_lag(self.upscale_lag(h_lag)).permute(1,2,0)
        h_cov = self.drop_cov(self.upscale_cov(h_cov)).permute(1,2,0)
        # Apply bitcn
        _, out_cov = self.net_fwd((h_cov, 0))
        _, out_lag = self.net_bwd((h_lag, 0))
        # Output layers - location & scale of the distribution
        out = torch.cat((out_cov[:, :, :dim_seq], out_lag), dim = 1)
        output = out[:, :, -d_outputseqlen:].permute(2, 0, 1)
        loc =  self.loc(output)
        return self.proj(loc.permute(1,0,2).unsqueeze(-1))