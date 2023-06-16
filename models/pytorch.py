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

class PytorchRegressor(BaseEstimator, RegressorMixin):
    """Class representing a pytorch regression module"""

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 loss_function: Any = PinballScore(),
                 batch_size: int = 256,
                 epochs: int = 1000,
                 patience: int = 15,
                 validation_ratio: float = 0.2):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optim.Adam(params=model.parameters(),lr = 0.0005)
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_ratio = validation_ratio

    def forward(self, X):
        return self.model(X)

    def fit(self, X: torch.Tensor, y: torch.Tensor, target_lags:List):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y.reshape(-1, 1), shuffle=False, test_size=self.validation_ratio)
        training_loader = DataLoader(dataset=TensorDataset(X_tr.float(), y_tr.float()),
                                     batch_size=self.batch_size,
                                     shuffle=True,num_workers=0)
        validation_loader = DataLoader(dataset=TensorDataset(X_val.float(), y_val.float()),
                                       batch_size=self.batch_size,
                                       shuffle=False,num_workers=0)
        # data_dict = {'train_loader': training_loader, 'validation_loader': validation_loader}
        # torch.save(data_dict, '/home/wzx3/benchmark/proenfo/notebooks/data_loader.pt')
        early_stopping = EarlyStopping(loss_function_name= self.loss_function.name,model_name = self.model.__class__.__name__, patience=self.patience, verbose=False)
        with trange(self.epochs) as t:
            for i in t:
                # Training mode
                self.model.train()
                losses = []
                for X_tr_batch_o, y_tr_batch in training_loader:
                    # Clear gradient buffers
                    self.optimizer.zero_grad()

                    # get output from the model, given the inputs
                    if self.model.__class__.__name__ in ['LongShortTermMemory','TransformerTS','WaveNet','LSTN','NBeatsNet']:
                        X_tr_batch = X_tr_batch_o[:,:len(target_lags)]
                        X_tr_batch_ex = X_tr_batch_o[:,len(target_lags):]
                        pred = self.model(X_tr_batch,X_tr_batch_ex)
                    else:
                        pred = self.model(X_tr_batch_o)

                    # get loss for the predicted output
                    loss = self.loss_function(pred, y_tr_batch)
                    losses.append(loss.item())
                    # get gradients w.r.t to parameters
                    loss.backward()

                    # update parameters
                    self.optimizer.step()
                # t.set_description("train_loss %i" % np.mean(losses))
                # Evaluation mode
                self.model.eval()
                validation_losses = []
                for X_val_batch_o, y_val_batch in validation_loader:
                    if self.model.__class__.__name__ in ['LongShortTermMemory','TransformerTS','WaveNet','LSTN','NBeatsNet']:
                        X_val_batch = X_val_batch_o[:,:len(target_lags)]
                        X_val_batch_ex = X_val_batch_o[:,len(target_lags):]
                        pred = self.model(X_val_batch,X_val_batch_ex)
                    else:
                        pred = self.model(X_val_batch_o)
                    loss = self.loss_function(pred, y_val_batch)
                    validation_losses.append(loss.item())

                # Stop if patience is reached
                early_stopping(np.average(validation_losses), self.model)
                if early_stopping.early_stop:
                    break
                t.set_description("train_loss %0.4f,val_loss %0.4f" % (np.mean(losses),np.mean(validation_losses)))
            # Load saved model
            # self.model.load_state_dict(load('./pkl_folder/checkpoint.pt'))
            self.model.load_state_dict(load(early_stopping.save_path))

            # Clean up checkpoint
            early_stopping.clean_up_checkpoint()

            return self

    def predict(self, X: torch.Tensor,target_lags:List) -> np.ndarray:
        with no_grad():
            if self.model.__class__.__name__ in  ['LongShortTermMemory','TransformerTS','WaveNet','LSTN','NBeatsNet']:
                X_batch = X[:,:len(target_lags)]
                X_batch_ex = X[:,len(target_lags):]
                return self.model(X_batch.float(),X_batch_ex.float()).cpu().data.numpy().squeeze()
            else:
                return self.model(X.float()).cpu().data.numpy().squeeze()
           


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, n_features: int, n_neurons: int = 50, n_output: int = 1):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_output))

    def forward(self, X_batch):
        return self.net(X_batch)


class LongShortTermMemory(nn.Module):
    def __init__(self,n_features: int, external_features_diminsion:int, n_neurons: int = 64, n_layers: int = 2, n_output: int = 1):
        super(LongShortTermMemory, self).__init__()
 
   
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.external_features_diminsion = external_features_diminsion
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_neurons, num_layers=n_layers, batch_first=True)
        self.seq = nn.Sequential(nn.ReLU(),nn.Linear(n_neurons+external_features_diminsion, n_output))

    def forward(self, X_batch,X_batch_ex):
        # batch_size = len(X_batch)
        # hidden_state = zeros(self.n_layers, batch_size, self.n_neurons).to('cuda:0')
        # cell_state = zeros(self.n_layers, batch_size, self.n_neurons).to('cuda:0')
        # output, _ = self.lstm(unsqueeze(X_batch, dim=-1), (hidden_state, cell_state))
        output, _ = self.lstm(unsqueeze(X_batch, dim=-1))
        concat_feature = torch.concat([output[:, -1],X_batch_ex.reshape(-1,self.external_features_diminsion)],1)
        return self.seq(concat_feature)


class ConvolutionalNeuralNetwork(nn.Module):
    """Adapted from (https://github.com/nidhi-30/CNN-Regression-Pytorch/blob/master/1095526_1dconv.ipynb)"""

    def __init__(self, n_features: int, batch_size: int = 256, n_output: int = 1):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.n_features = n_features
        self.net = nn.Sequential(nn.Conv1d(n_features, batch_size, 1, stride=1),
                                 nn.ReLU(),
                                 nn.MaxPool1d(1),
                                 nn.ReLU(),
                                 nn.Conv1d(batch_size, 128, 1, stride=3),
                                 nn.MaxPool1d(1),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 256, 1, stride=3),
                                 nn.Flatten(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, n_output))

    def forward(self, X):
        return self.net(X.reshape((len(X), self.n_features, 1)))


###########Code for Transformer############
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x
class TransformerTS(nn.Module):
    def __init__(self,
                 n_features,
                 external_features_diminsion:int,
                 dec_seq_len = 7,
                 #out_seq_len,
                 n_output = 1,
                 d_model=256,
                 nhead=4,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=512,
                 dropout=0,
                 activation='relu',
                 custom_encoder=None,
                 custom_decoder=None):
        r"""A transformer model. User is able to modify the attributes as needed. The architecture
        is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
        Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
        Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
        Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
        model with corresponding parameters.

        Args:
            input_dim: dimision of imput series
            d_model: the number of expected features in the encoder/decoder inputs (default=512).
            nhead: the number of heads in the multiheadattention models (default=8).
            num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
            num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
            custom_encoder: custom encoder (default=None).
            custom_decoder: custom decoder (default=None).

        Examples::
            >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
            >>> src = torch.rand((10, 32, 512)) (time length, N, feature dim)
            >>> tgt = torch.rand((20, 32, 512))
            >>> out = transformer_model(src, tgt)

        Note: A full example to apply nn.Transformer module for the word language model is available in
        https://github.com/pytorch/examples/tree/master/word_language_model
        """
        super(TransformerTS, self).__init__()
        self.transform = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )
        self.pos = PositionalEncoding(d_model)
        self.enc_input_fc = nn.Linear(n_features, d_model)
        self.dec_input_fc = nn.Linear(n_features, d_model)
        self.out_fc = nn.Sequential(nn.ReLU(),nn.Linear(dec_seq_len * d_model+external_features_diminsion, n_output))
        self.dec_seq_len = dec_seq_len
        self.external_features_diminsion = external_features_diminsion

    def forward(self, X_batch,X_batch_ex):
        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], -1)
        X_batch = X_batch.transpose(0, 1)
        # embedding
        embed_encoder_input = self.pos(self.enc_input_fc(X_batch))
        embed_decoder_input = self.dec_input_fc(X_batch[-self.dec_seq_len:, :])
        # transform
        X_batch = self.transform(embed_encoder_input, embed_decoder_input)

        # output
        X_batch = X_batch.transpose(0, 1)
        X_batch = self.out_fc(torch.concat([X_batch.flatten(start_dim=1),X_batch_ex.reshape(-1,self.external_features_diminsion)],1))
        return X_batch

###########Code for Wavenet############
class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    """
    def __init__(self, in_size, out_size, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]  
        return x
class ResidualLayer(nn.Module):    
    def __init__(self, residual_size, skip_size, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size,
                                         kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size, residual_size,
                                         kernel_size=2, dilation=dilation)        
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)
        
   
    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)  
        fx = F.tanh(conv_filter) * F.sigmoid(conv_gate)
        fx = self.resconv1_1(fx) 
        skip = self.skipconv1_1(fx) 
        residual = fx + x  
        #residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual
class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, 2**layer)
                         for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)
        
    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            #skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_size,seq_len]
class WaveNet(nn.Module):

    def __init__(self,n_features,external_features_diminsion,n_output,
                 out_size = 1,
                 residual_size = 16, 
                 skip_size = 4 , 
                 dilation_cycles = 2, 
                 dilation_depth = 2):

        super(WaveNet, self).__init__()

        self.input_conv = CausalConv1d(n_features,residual_size, kernel_size=2)        

        self.dilated_stacks = nn.ModuleList(

            [DilatedStack(residual_size, skip_size, dilation_depth)

             for cycle in range(dilation_cycles)]

        )

        self.convout_1 = nn.Conv1d(skip_size, out_size, kernel_size=1)

        self.convout_2 = nn.Conv1d(out_size, out_size, kernel_size=1)
        self.linear = nn.Sequential(nn.ReLU(),nn.Linear(out_size+external_features_diminsion,n_output))
        self.external_features_diminsion = external_features_diminsion

    def forward(self, X_batch,X_batch_ex):

        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], -1)

        X_batch = X_batch.permute(0,2,1)# [batch,input_feature_dim, seq_len]

        X_batch = self.input_conv(X_batch) # [batch,residual_size, seq_len]             

        skip_connections = []

        for cycle in self.dilated_stacks:

            skips, X_batch = cycle(X_batch)             
            skip_connections.append(skips)

        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)        

        # gather all output skip connections to generate output, discard last residual output

        out = skip_connections.sum(dim=0) # [batch,skip_size,seq_len]

        out = F.relu(out)

        out = self.convout_1(out) # [batch,out_size,seq_len]
        out = F.relu(out)

        out=self.convout_2(out)

        out=out.permute(0,2,1)
        out = self.linear(torch.concat([out[:,-1,:],X_batch_ex.reshape(-1,self.external_features_diminsion)],1))
        #[bacth,seq_len,out_size]
        return out  


###########Code for LSTN############
class LSTN(nn.Module):
    def __init__(self,n_feature, external_features_diminsion,n_output,
                 P = 7
                 ,m=1,
                 hidR = 16,hidC = 16 ,hidS = 32,Ck = 2,skip = 1,hw = 7):
        super(LSTN, self).__init__()
        self.P = P
        self.m = m
        self.hidR = hidR
        self.hidC = hidC
        self.hidS = hidS
        self.Ck = Ck
        self.skip = skip
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = hw
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = 0)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output_linear = nn.Sequential(nn.ReLU(),nn.Linear(m+external_features_diminsion,n_output))
        self.external_features_diminsion = external_features_diminsion
            
    def forward(self, X_batch,X_batch_ex):
        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], -1)
        batch_size = X_batch.size(0)
        
        #CNN
        c = X_batch.view(-1, 1, self.P, self.m)
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
            z = X_batch[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
        res = self.output_linear(torch.concat([res,X_batch_ex.reshape(-1,self.external_features_diminsion)],1))
            
        return res

###########Code for NBEATS############
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
            n_feature,
            external_features_diminsion,
            n_output,
            device=torch.device('cuda'),
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=3,
            forecast_length=1,
            backcast_length=7,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            hidden_layer_units=64,
            nb_harmonics=None
    ):
        super(NBeatsNet, self).__init__()
        forecast_length = 1
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = device
        # print('| N-Beats')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []
        self.external_features_diminsion = external_features_diminsion
        self.linear = nn.Sequential(nn.ReLU(),nn.Linear(forecast_length+external_features_diminsion,n_output))

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        # print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
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
            # print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock


    def predict(self, x, return_backcast=False):
        self.eval()
        b, f = self(torch.tensor(x, dtype=torch.float).to(self.device))
        b, f = b.detach().numpy(), f.detach().numpy()
        if len(x.shape) == 3:
            b = np.expand_dims(b, axis=-1)
            f = np.expand_dims(f, axis=-1)
        if return_backcast:
            return b
        return f


    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def forward(self, backcast,X_batch_ex):
        self._intermediary_outputs = []
        backcast = squeeze_last_dim(backcast)
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f'stack_{stack_id}-{block_type}_{block_id}'
                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append({'value': f.detach().numpy(), 'layer': layer_name})
        forecast = self.linear(torch.concat([forecast,X_batch_ex.reshape(-1,self.external_features_diminsion)],1))
        
        return forecast


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
