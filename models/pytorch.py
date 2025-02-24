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
import warnings
warnings.filterwarnings("ignore")

def moidify_data_tensor(input_tensor,targets_dim,lags_length,preds_length,X_scaler,ex_time_dim,train = True):
    input_tensor_copy = input_tensor.copy()
    if train:
        input_tensor = X_scaler.fit_transform(input_tensor)
    else:
        input_tensor = X_scaler.transform(input_tensor)
    targets_seq_dim = targets_dim*lags_length
    historical = input_tensor[:,:targets_seq_dim].reshape(-1,lags_length,targets_dim)
    ex = input_tensor[:,targets_seq_dim:].reshape(historical.shape[0],lags_length+preds_length,-1)
    ex_nnr = input_tensor_copy[:,targets_seq_dim:].reshape(historical.shape[0],lags_length+preds_length,-1)
    ex_nnr  = np.concatenate([ex_nnr[:,:,:ex_time_dim],ex[:,:,ex_time_dim:]],-1)
    return historical,ex,ex_nnr,X_scaler

class CustomDataset(Dataset):
    def __init__(self, historical,ex_nr,ex_nnr,targets,targets_dim,lags_length,preds_length):
        self.targets  = targets
        self.targets_dim = targets_dim
        self.lags_length = lags_length
        self.preds_length = preds_length
        # self.his,self.ex = moidify_data_tensor(data,targets_dim,lags_length,preds_length)
        self.his,self.ex_nr, self.ex_nnr = historical,ex_nr,ex_nnr
        if targets is not None:
            self.y = targets.reshape(targets.shape[0],preds_length,-1)

    def rearrange_tensor(self,input_tensor, lags_length, preds_length):
        # 计算不同特征的数量
        time_step = lags_length + preds_length
        num_features = (input_tensor.shape[1] - lags_length) // time_step

        # 提取lag_features和else_features
        lag_features = input_tensor[:, :lags_length]
        else_features = input_tensor[:, lags_length:]

        else_features_list = []
        # print(else_features.shape)
        for i in range(time_step):
            else_features_list.append(else_features[:,i::time_step].unsqueeze(1))
            
        else_features = torch.cat(else_features_list, dim=1)
        # 提取pred_ex_features和lag_ex_features
        pred_ex_features = else_features[:, lags_length:]
        lag_ex_features = else_features[:, :lags_length]


        return pred_ex_features[:,:,:-1], lag_features.unsqueeze(-1), lag_ex_features[:,:,:-1], pred_ex_features[:,:,-1:],lag_ex_features[:,:,-1:]

    def __len__(self):
        return len(self.his)

    def __getitem__(self, index):
        if self.targets is not None:
            x = self.his[index]
            x_ex = self.ex_nr[index]
            x_ex_nnr = self.ex_nnr[index]
            y = self.y[index]
            return x, x_ex,y,x_ex_nnr
            # return x, x_ex,y
        else:
            x = self.his[index]
            x_ex = self.ex_nr[index]
            x_ex_nnr = self.ex_nnr[index]
            return x,x_ex,x_ex_nnr
            return x,x_ex

class Combined_model(nn.Module):
    def __init__(self, model,ex_model):
        super(Combined_model, self).__init__()
        self.time_series_model = model
        self.external_model = ex_model
        self.name = self.time_series_model.__class__.__name__
        for param in self.time_series_model.parameters():
            param.requires_grad = False
    def forward(self,*args):
        last_arg = args[-1]
        other_args = args[:-1]
        time_series_predict = self.time_series_model(*other_args)
        output = self.external_model(time_series_predict,last_arg)
        return output




class PytorchRegressor(BaseEstimator, RegressorMixin):
    """Class representing a pytorch regression module"""

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 ex_model:Optional[nn.Module] = None,
                 loss_function: Any = PinballScore(),
                 batch_size: int = 32,
                 epochs: int = 50,
                 patience: int = 5,
                 validation_ratio: float = 0.2,
                 learning_rate = 0.0005,
                 device = None,
                 ex_learning_rate = 1e-5):
        self.learning_rate = learning_rate
        if device:
            self.device = device
        self.model = model
        if ex_model:
            self.ex_model = ex_model
            self.use_ex_model = True
            # self.model = Combined_model(self.model,self.ex_model).to(self.device)
            self.ex_epochs = 20
        else:
            self.ex_model = None
            self.use_ex_model = False
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_ratio = validation_ratio
        
        self.optimizer = optim.Adam(params=self.model.parameters(),lr = learning_rate)
        self.targets_dim = 1
        self.ex_learning_rate = ex_learning_rate

    def forward(self, X):
        raise ValueError('Please use model.fit or model.predict instead')
    
    def train_main_model(self,training_loader,validation_loader,lags_length,preds_length):
        early_stopping = EarlyStopping(loss_function_name= self.loss_function.name,model_name = self.model.__class__.__name__, patience=self.patience, verbose=False)
        with trange(self.epochs) as t:
            for i in t:
                # Training model
                self.model.train()
                losses = []
                for X_tr_batch,X_tr_ex_batch, y_tr_batch,X_tr_ex_batch_nnr in training_loader:
                    # Clear gradient buffers
                    self.optimizer.zero_grad()
                    # get output from the model, given the inputs
                    if self.model.name in ['DLinear','LSTMF','MLP','CNN','LSTNet','TSMixer']:
                        pred = self.model(X_tr_batch)
            
                    elif self.model.name in ['Transformer','Informer','Autoformer','Fedformer','FiLM',
                                                            'iTransformer','NSTransformer','PatchTST','SegRNN','TimeMixer',
                                                            'TimesNet','FreTS','Reformer','MICN','WPMixer']:
                        X_enc_mark = X_tr_ex_batch[:,:lags_length]
                        X_enc = X_tr_batch
                        X_dec = X_enc[:,X_enc.shape[1]//2:]
                        X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                        pred = self.model(X_enc,None,X_dec,None,)
                    elif self.model.name in ['NBeatsNet']:
                        pred = self.model(X_tr_batch)
                    elif self.model.name in ['NBeatsNetX']:
                        # pred = self.model(X_tr_batch,X_tr_ex_batch[:,-preds_length:])
                        pred = self.model(X_tr_batch,X_tr_ex_batch)
                    elif self.model.name in ['TimeXer']:
                        pred = self.model(X_tr_batch,X_tr_ex_batch[:,:lags_length],None,None)
                    elif self.model.name in ['TSMixerExt']:
                        
                        pred = self.model(X_tr_batch,X_tr_ex_batch[:,:lags_length],X_tr_ex_batch[:,lags_length:],
                                            torch.zeros((X_tr_batch.shape[0], 1)).to(X_tr_batch.device))
                    elif self.model.name in ['WaveNet','BiTCN']:
                        pred = self.model(X_tr_batch.permute(1,0,2),X_tr_ex_batch.permute(1,0,2),preds_length)
                    elif self.model.name in ['TiDE']:
                        y_batch_mark = X_tr_ex_batch
                        X_tr_ex_batch = X_tr_ex_batch[:,:lags_length]
                        X_enc= X_tr_batch
                        X_dec = X_enc[:,X_enc.shape[1]//2:]
                        X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros_like(X_enc)],1)
                        pred = self.model(X_tr_batch,X_tr_ex_batch,X_dec,y_batch_mark)
                    elif self.model.name in ['TFT']:
                        X_enc = X_tr_batch
                        X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                        X_enc_mark = X_tr_ex_batch[:,:lags_length,:]
                        X_dec_mark = X_tr_ex_batch[:,-(preds_length+lags_length//2):,:]
                        pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark)
                    # get loss for the predicted output
                    loss = self.loss_function(pred, y_tr_batch.unsqueeze(-1))
                    losses.append(loss.item())
                    # get gradients w.r.t to parameters
                    loss.backward()

                    # update parameters
                    self.optimizer.step()
                # t.set_description("train_loss %i" % np.mean(losses))
                # Evaluation mode
                self.model.eval()
                validation_losses = []
                with no_grad():
                    for X_val_batch,X_val_ex_batch, y_val_batch, X_val_ex_batch_nnr in validation_loader:
                        if self.model.name in ['DLinear','LSTMF','MLP','CNN','LSTNet','TSMixer']:
                            pred = self.model(X_val_batch)
                        elif self.model.name in ['Transformer','Informer','Autoformer','Fedformer','FiLM',
                                                            'iTransformer','NSTransformer','PatchTST','SegRNN','TimeMixer',
                                                            'TimesNet','FreTS','Reformer','MICN','WPMixer']:
                            X_enc_mark = X_val_ex_batch[:,:lags_length]
                            X_enc = X_val_batch
                            X_dec = X_enc[:,X_enc.shape[1]//2:]
                            X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                            pred = self.model(X_enc,None,X_dec,None,)
                        elif self.model.name in ['NBeatsNet']:
                            pred = self.model(X_val_batch)
                        elif self.model.name in ['NBeatsNetX']:
                            # pred = self.model(X_val_batch,X_val_ex_batch[:,-preds_length:])
                            pred = self.model(X_val_batch,X_val_ex_batch)
                        elif self.model.name in ['TimeXer']:
                            pred = self.model(X_val_batch,X_val_ex_batch[:,:lags_length],None,None)
                        elif self.model.name in ['TSMixerExt']:
                            
                            pred = self.model(X_val_batch,X_val_ex_batch[:,:lags_length],X_val_ex_batch[:,lags_length:],
                                            torch.zeros((X_val_batch.shape[0], 1)).to(X_val_batch.device))
                        elif self.model.name in ['WaveNet','BiTCN']:
                            pred = self.model(X_val_batch.permute(1,0,2),X_val_ex_batch.permute(1,0,2),preds_length)
                        elif self.model.name in ['TiDE']:
                            y_batch_mark = X_val_ex_batch
                            X_val_ex_batch = X_val_ex_batch[:,:lags_length]
                            X_enc= X_val_batch
                            X_dec = X_enc[:,X_enc.shape[1]//2:]
                            X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros_like(X_enc)],1)
                            
                            pred = self.model(X_val_batch,X_val_ex_batch,X_dec,y_batch_mark)
                        elif self.model.name in ['TFT']:
                            X_enc = X_val_batch
                            X_dec = X_enc[:,X_enc.shape[1]//2:]
                            X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                            X_enc_mark = X_val_ex_batch[:,:lags_length,:]
                            X_dec_mark = X_val_ex_batch[:,-(preds_length+lags_length//2):,:]
                            pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark)
                        loss = self.loss_function(pred, y_val_batch.unsqueeze(-1))
                        validation_losses.append(loss.item())

                # Stop if patience is reached
                early_stopping(np.average(validation_losses), self.model)
                if early_stopping.early_stop:
                    break
                t.set_description("train_loss %0.4f,val_loss %0.4f" % (np.mean(losses),np.mean(validation_losses)))
            return early_stopping


    def train_ex_model(self,training_loader,validation_loader,lags_length,preds_length):
        ex_optimizer = torch.optim.Adam(self.model.external_model.parameters(), lr=self.ex_learning_rate)
        ex_early_stopping = EarlyStopping(loss_function_name= self.loss_function.name,model_name = self.model.__class__.__name__, patience=self.patience, verbose=False)
        with trange(self.ex_epochs) as t:
            for i in t:
                # Training model
                self.model.train()
                losses = []
                for X_tr_batch,X_tr_ex_batch, y_tr_batch,X_tr_ex_batch_nnr in training_loader:
                    # Clear gradient buffers
                    ex_optimizer.zero_grad()
                    # get output from the model, given the inputs
                    if self.model.name in ['DLinear','LSTMF','MLP','CNN','LSTNet','TSMixer']:
                        
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_tr_batch,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_tr_batch,X_tr_ex_batch[:,-preds_length:,:])
                        
                    elif self.model.name in ['Transformer','Informer','Autoformer','Fedformer','FiLM',
                                                            'iTransformer','NSTransformer','PatchTST','SegRNN','TimeMixer',
                                                            'TimesNet','FreTS','Reformer','MICN','WPMixer']:
                        X_enc_mark = X_tr_ex_batch[:,:lags_length]
                        X_enc = X_tr_batch
                        X_dec = X_enc[:,X_enc.shape[1]//2:]
                        X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                        
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_enc,None,X_dec,None,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_enc,None,X_dec,None,X_tr_ex_batch[:,-preds_length:,:])
                        
                    elif self.model.name in ['NBeatsNet']:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_tr_batch,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_tr_batch,X_tr_ex_batch[:,-preds_length:,:])
                        
                    elif self.model.name in ['NBeatsNetX']:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            # pred = self.model(X_tr_batch,X_tr_ex_batch[:,-preds_length:],X_tr_ex_batch_nnr[:,-preds_length:,:])
                            pred = self.model(X_tr_batch,X_tr_ex_batch,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            # pred = self.model(X_tr_batch,X_tr_ex_batch[:,-preds_length:],X_tr_ex_batch[:,-preds_length:,:])
                            pred = self.model(X_tr_batch,X_tr_ex_batch,X_tr_ex_batch[:,-preds_length:,:])
                    elif self.model.name in ['TimeXer']:
                        
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_tr_batch,X_tr_ex_batch[:,:lags_length],None,None,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_tr_batch,X_tr_ex_batch[:,:lags_length],None,None,X_tr_ex_batch[:,-preds_length:,:])
                        
                    elif self.model.name in ['TSMixerExt']:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_tr_batch,X_tr_ex_batch[:,:lags_length],X_tr_ex_batch[:,lags_length:],
                                        torch.zeros((X_tr_batch.shape[0], 1)).to(X_tr_batch.device),X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_tr_batch,X_tr_ex_batch[:,:lags_length],X_tr_ex_batch[:,lags_length:],
                                            torch.zeros((X_tr_batch.shape[0], 1)).to(X_tr_batch.device),X_tr_ex_batch[:,-preds_length:,:])
                        
                    elif self.model.name in ['WaveNet','BiTCN']:
                        
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_tr_batch.permute(1,0,2),X_tr_ex_batch.permute(1,0,2),preds_length,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_tr_batch.permute(1,0,2),X_tr_ex_batch.permute(1,0,2),preds_length,X_tr_ex_batch[:,-preds_length:,:])
                       
                    elif self.model.name in ['TiDE']:
                        y_batch_mark = X_tr_ex_batch
                        X_tr_ex_batch = X_tr_ex_batch[:,:lags_length]
                        X_enc= X_tr_batch
                        X_dec = X_enc[:,X_enc.shape[1]//2:]
                        X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros_like(X_enc)],1)
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_tr_batch,X_tr_ex_batch,X_dec,y_batch_mark,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_tr_batch,X_tr_ex_batch,X_dec,y_batch_mark,y_batch_mark[:,-preds_length:,:])
                        
                    elif self.model.name in ['TFT']:
                        X_enc = X_tr_batch
                        X_dec = X_enc[:,X_enc.shape[1]//2:]
                        X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                        X_enc_mark = X_tr_ex_batch[:,:lags_length,:]
                        X_dec_mark = X_tr_ex_batch[:,-(preds_length+lags_length//2):,:]
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark,X_tr_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark,X_tr_ex_batch[:,-preds_length:,:])
                        
                    # get loss for the predicted output
                    loss = self.loss_function(pred, y_tr_batch.unsqueeze(-1))
                    losses.append(loss.item())
                    # get gradients w.r.t to parameters
                    loss.backward()

                    # update parameters
                    ex_optimizer.step()
                # t.set_description("train_loss %i" % np.mean(losses))
                # Evaluation mode
                self.model.eval()
                validation_losses = []
                with no_grad():
                    for X_val_batch,X_val_ex_batch, y_val_batch, X_val_ex_batch_nnr in validation_loader:
                        if self.model.name in ['DLinear','LSTMF','MLP','CNN','LSTNet','TSMixer']:
                            
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_val_batch,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_val_batch,X_val_ex_batch[:,-preds_length:,:])
                            
                        elif self.model.name in ['Transformer','Informer','Autoformer','Fedformer','FiLM',
                                                            'iTransformer','NSTransformer','PatchTST','SegRNN','TimeMixer',
                                                            'TimesNet','FreTS','Reformer','MICN','WPMixer']:
                            X_enc_mark = X_val_ex_batch[:,:lags_length]
                            X_enc = X_val_batch
                            # X_enc= torch.cat([X_val_batch,X_enc_mark],-1)
                            X_dec = X_enc[:,X_enc.shape[1]//2:]
                            X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                           
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_enc,None,X_dec,None,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_enc,None,X_dec,None,X_val_ex_batch[:,-preds_length:,:])
                            
                        elif self.model.name in ['NBeatsNet']:
                            
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_val_batch,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_val_batch,X_val_ex_batch[:,-preds_length:,:])
                            
                        elif self.model.name in ['NBeatsNetX']:
                            
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                # pred = self.model(X_val_batch,X_val_ex_batch[:,-preds_length:],X_val_ex_batch_nnr[:,-preds_length:,:])
                                pred = self.model(X_val_batch,X_val_ex_batch,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                # pred = self.model(X_val_batch,X_val_ex_batch[:,-preds_length:],X_val_ex_batch[:,-preds_length:,:])
                                pred = self.model(X_val_batch,X_val_ex_batch,X_val_ex_batch[:,-preds_length:,:])
                            
                        elif self.model.name in ['TimeXer']:
                           
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_val_batch,X_val_ex_batch[:,:lags_length],None,None,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_val_batch,X_val_ex_batch[:,:lags_length],None,None,X_val_ex_batch[:,-preds_length:,:])
                            
                        elif self.model.name in ['TSMixerExt']:
                            
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_val_batch,X_val_ex_batch[:,:lags_length],X_val_ex_batch[:,lags_length:],
                                        torch.zeros((X_val_batch.shape[0], 1)).to(X_val_batch.device),X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_val_batch,X_val_ex_batch[:,:lags_length],X_val_ex_batch[:,lags_length:],
                                        torch.zeros((X_val_batch.shape[0], 1)).to(X_val_batch.device),X_val_ex_batch[:,-preds_length:,:])
                           
                        elif self.model.name in ['WaveNet','BiTCN']:
                            
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_val_batch.permute(1,0,2),X_val_ex_batch.permute(1,0,2),preds_length,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_val_batch.permute(1,0,2),X_val_ex_batch.permute(1,0,2),preds_length,X_val_ex_batch[:,-preds_length:,:])
                            
                        elif self.model.name in ['TiDE']:
                            y_batch_mark = X_val_ex_batch
                            X_val_ex_batch = X_val_ex_batch[:,:lags_length]
                            X_enc= X_val_batch
                            X_dec = X_enc[:,X_enc.shape[1]//2:]
                            X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros_like(X_enc)],1)
                            
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_val_batch,X_val_ex_batch,X_dec,y_batch_mark,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_val_batch,X_val_ex_batch,X_dec,y_batch_mark,y_batch_mark[:,-preds_length:,:])
                            
                        elif self.model.name in ['TFT']:
                            X_enc = X_val_batch
                            X_dec = X_enc[:,X_enc.shape[1]//2:]
                            X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                            X_enc_mark = X_val_ex_batch[:,:lags_length,:]
                            X_dec_mark = X_val_ex_batch[:,-(preds_length+lags_length//2):,:]
                           
                            if 'ex_HT' in self.ex_model.__class__.__name__:
                                pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark,X_val_ex_batch_nnr[:,-preds_length:,:])
                            else:
                                pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark,X_val_ex_batch[:,-preds_length:,:])
                            
                        loss = self.loss_function(pred, y_val_batch.unsqueeze(-1))
                        validation_losses.append(loss.item())

                # Stop if patience is reached
                ex_early_stopping(np.average(validation_losses), self.model)
                if ex_early_stopping.early_stop:
                    break
                t.set_description("train_loss %0.4f,val_loss %0.4f" % (np.mean(losses),np.mean(validation_losses)))
            return ex_early_stopping



    def fit(self, X: torch.Tensor, y: torch.Tensor, target_lags: List,target_preds: List,X_scaler,ex_time_dim):

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, shuffle=False, test_size=self.validation_ratio)
        lags_length = len(target_lags)
        preds_length = len(target_preds)+1
        targets_dim = y_tr.shape[-1]//preds_length
        self.targets_dim  = targets_dim
        X_tr_his,X_tr_ex,X_tr_ex_nnr,X_scaler = moidify_data_tensor(X_tr,targets_dim,lags_length,preds_length,X_scaler,ex_time_dim)
        X_tr_his,X_tr_ex,X_tr_ex_nnr = torch.Tensor(X_tr_his).to(y_tr.device),torch.Tensor(X_tr_ex).to(y_tr.device),torch.Tensor(X_tr_ex_nnr).to(y_tr.device)
        X_val_his,X_val_ex, X_val_ex_nnr,X_scaler = moidify_data_tensor(X_val,targets_dim,lags_length,preds_length,X_scaler,ex_time_dim,train  = False)
        X_val_his,X_val_ex,X_val_ex_nnr = torch.Tensor(X_val_his).to(y_val.device),torch.Tensor(X_val_ex).to(y_val.device),torch.Tensor(X_val_ex_nnr).to(y_tr.device)
        training_loader = DataLoader(dataset=CustomDataset(X_tr_his.float(),X_tr_ex.float(),X_tr_ex_nnr.float(), y_tr.float(),targets_dim,lags_length,preds_length),
                                     batch_size=self.batch_size,
                                     shuffle=True,num_workers=0)
        validation_loader = DataLoader(dataset=CustomDataset(X_val_his.float(),X_val_ex.float(), X_val_ex_nnr.float(),y_val.float(),targets_dim,lags_length,preds_length),
                                       batch_size=self.batch_size,
                                       shuffle=False,num_workers=0)
        # data_dict = {'train_loader': training_loader, 'validation_loader': validation_loader}
        # torch.save(data_dict, '/home/wzx3/benchmark/proenfo/notebooks/data_loader.pt')
        early_stopping = self.train_main_model(training_loader,validation_loader,lags_length,preds_length)
        # Load saved model
        # self.model.load_state_dict(load('./pkl_folder/checkpoint.pt'))
        self.model.load_state_dict(load(early_stopping.save_path))
        # Clean up checkpoint
        early_stopping.clean_up_checkpoint()

        if self.use_ex_model:

            self.model = Combined_model(self.model,self.ex_model).to(self.device)

            ex_early_stopping = self.train_ex_model(training_loader,validation_loader,lags_length,preds_length)

            self.model.load_state_dict(load(ex_early_stopping.save_path))

            ex_early_stopping.clean_up_checkpoint()

        return self,X_scaler

    def predict(self, X: torch.Tensor,y,target_lags:List,target_preds:List,X_scaler,ex_time_dim,device) -> np.ndarray:
        lags_length = len(target_lags)
        preds_length = len(target_preds)+1
        targets_dim = y.shape[-1]//preds_length
        self.model.eval()
        X_test_his,X_test_ex,X_test_ex_nnr,X_scaler = moidify_data_tensor(X,targets_dim,lags_length,preds_length,X_scaler,ex_time_dim,train  =False)
        X_test_his,X_test_ex,X_test_ex_nnr = torch.Tensor(X_test_his).to(device),torch.Tensor(X_test_ex).to(device),torch.Tensor(X_test_ex_nnr).to(device)
        test_loader = DataLoader(dataset=CustomDataset(X_test_his.float(),X_test_ex.float(),X_test_ex_nnr.float(), None,self.targets_dim,lags_length,preds_length),
                                     batch_size=self.batch_size,
                                     shuffle=False,num_workers=0)
        preds = []
        with no_grad():
            for X_test_batch,X_test_ex_batch, X_test_ex_batch_nnr in test_loader:
                if self.model.name in ['DLinear','LSTMF','MLP','CNN','LSTNet','TSMixer']:
                    X_enc_mark = X_test_ex_batch[:,:lags_length]
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_test_batch,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_test_batch,X_test_ex_batch[:,-preds_length:,:])
                    else:
                        pred = self.model(torch.cat([X_test_batch,X_enc_mark],-1))
                elif self.model.name in ['Transformer','Informer','Autoformer','Fedformer','FiLM',
                                                    'iTransformer','NSTransformer','PatchTST','SegRNN','TimeMixer',
                                                    'TimesNet','FreTS','Reformer','MICN','WPMixer']:
                    X_enc_mark = X_test_ex_batch[:,:lags_length]
                    X_enc = X_test_batch
                    # X_enc= torch.cat([X_test_batch,X_enc_mark],-1)
                    X_dec = X_enc[:,X_enc.shape[1]//2:]
                    X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_enc,None,X_dec,None,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_enc,None,X_dec,None,X_test_ex_batch[:,-preds_length:,:])
                    else:
                        pred = self.model(X_enc,None,X_dec,None,)
                elif self.model.name in ['NBeatsNet']:
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_test_batch,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_test_batch,X_test_ex_batch[:,-preds_length:,:])
                    else:
                        pred = self.model(X_test_batch)
                elif self.model.name in ['NBeatsNetX']:
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            # pred = self.model(X_test_batch,X_test_ex_batch[:,-preds_length:],X_test_ex_batch_nnr[:,-preds_length:,:])
                            pred = self.model(X_test_batch,X_test_ex_batch,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            # pred = self.model(X_test_batch,X_test_ex_batch[:,-preds_length:],X_test_ex_batch[:,-preds_length:,:])
                            pred = self.model(X_test_batch,X_test_ex_batch,X_test_ex_batch[:,-preds_length:,:])
                    else:
                        # pred = self.model(X_test_batch,X_test_ex_batch[:,:lags_length])
                        pred = self.model(X_test_batch,X_test_ex_batch)
                elif self.model.name in ['TimeXer']:
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_test_batch,X_test_ex_batch[:,:lags_length],None,None,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_test_batch,X_test_ex_batch[:,:lags_length],None,None,X_test_ex_batch[:,-preds_length:,:])
                    else:
                        pred = self.model(X_test_batch,X_test_ex_batch[:,:lags_length],None,None)
                elif self.model.name in ['TSMixerExt']:
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_test_batch,X_test_ex_batch[:,:lags_length],X_test_ex_batch[:,lags_length:],
                                    torch.zeros((X_test_batch.shape[0], 1)).to(X_test_batch.device),X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_test_batch,X_test_ex_batch[:,:lags_length],X_test_ex_batch[:,lags_length:],
                                    torch.zeros((X_test_batch.shape[0], 1)).to(X_test_batch.device),X_test_ex_batch[:,-preds_length:,:])
                    else:
                        pred = self.model(X_test_batch,X_test_ex_batch[:,:lags_length],X_test_ex_batch[:,lags_length:],
                                    torch.zeros((X_test_batch.shape[0], 1)).to(X_test_batch.device))
                elif self.model.name in ['WaveNet','BiTCN']:
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_test_batch.permute(1,0,2),X_test_ex_batch.permute(1,0,2),preds_length,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_test_batch.permute(1,0,2),X_test_ex_batch.permute(1,0,2),preds_length,X_test_ex_batch[:,-preds_length:,:])
                    else:
                        pred = self.model(X_test_batch.permute(1,0,2),X_test_ex_batch.permute(1,0,2),preds_length)
                elif self.model.name in ['TiDE']:
                    y_batch_mark = X_test_ex_batch
                    X_test_ex_batch = X_test_ex_batch[:,:lags_length]
                    X_enc= X_test_batch
                    X_dec = X_enc[:,X_enc.shape[1]//2:]
                    X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros_like(X_enc)],1)
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_test_batch,X_test_ex_batch,X_dec,y_batch_mark,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_test_batch,X_test_ex_batch,X_dec,y_batch_mark,y_batch_mark[:,-preds_length:,:])
                    else:
                        pred = self.model(X_test_batch,X_test_ex_batch,X_dec,y_batch_mark)
                elif self.model.name in ['TFT']:
                    X_enc = X_test_batch
                    X_dec = X_enc[:,X_enc.shape[1]//2:]
                    X_dec = torch.cat([X_enc[:,X_enc.shape[1]//2:],torch.zeros((X_enc.shape[0],preds_length,X_enc.shape[-1])).to(X_enc.device)],1)
                    X_enc_mark = X_test_ex_batch[:,:lags_length,:]
                    X_dec_mark = X_test_ex_batch[:,-(preds_length+lags_length//2):,:]
                    if self.use_ex_model:
                        if 'ex_HT' in self.ex_model.__class__.__name__:
                            pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark,X_test_ex_batch_nnr[:,-preds_length:,:])
                        else:
                            pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark,X_test_ex_batch[:,-preds_length:,:])
                    else:
                        pred = self.model(X_enc,X_enc_mark,X_dec,X_dec_mark)
                preds.append(pred)
            return torch.cat(preds, dim=0)

           
