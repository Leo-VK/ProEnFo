import os
import torch
import torch.backends
import random
import numpy as np

class Configs:
    def __init__(self):
        self.data = 'GEF12'
        self.features = 'M'
        self.freq = 'h'
        self.model = 'Autoformer'

        # Forecasting task
        self.seq_len = 24
        self.label_len = 12
        self.pred_len = 24
        self.seasonal_patterns = 'Monthly'
        self.inverse = False

        # Model define
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 256
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 1024
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.use_norm = 1
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = None
        self.seg_len = 24

        # GPU
        self.use_gpu = True
        self.gpu = 0
        self.gpu_type = 'cuda'
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

        # TimeXer
        self.patch_len = 16

    def setup_device(self):
        if torch.cuda.is_available() and self.use_gpu:
            self.device = torch.device(f'cuda:{self.gpu}')
            print('Using GPU')
        else:
            if hasattr(torch.backends, "mps"):
                self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            else:
                self.device = torch.device("cpu")
            print('Using cpu or mps')

        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id_) for id_ in device_ids]
            self.gpu = self.device_ids[0]