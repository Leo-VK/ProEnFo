from abc import ABC, abstractmethod
from typing import Any, Union, Optional, Literal

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor as QuantileRegression
from sklearn.preprocessing import StandardScaler
from sklearn_quantile import KNeighborsQuantileRegressor, RandomForestQuantileRegressor, \
    SampleRandomForestQuantileRegressor, ExtraTreesQuantileRegressor, SampleExtraTreesQuantileRegressor
from sklearn.decomposition import PCA
import models.pytorch
from models.conditional_error_model import ConditionalErrorQuantile, BootstrapConditionalErrorQuantile
from models.optimization_model import QuantileLinearProgram, QuantileSupportVectorRegression, \
    MultiQuantileLinearProgram
from models.boosting import CustomQuantileLossXGBRegressor
from preprocessing.quantile_format import check_quantile_list
from utils import randomness, pytorchtools

from typing import List, Tuple, Dict
import numpy as np 
import torch.nn as nn
import torch
from torch.nn import functional as F

from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

#Mulivariate forecasting models
from .NNmodel import *
from .Autoformer import Autoformer
from .DLinear import DLinear
from .Fedformer import Fedformer
from .FiLM import FiLM
from .Informer import Informer
from .iTransformer import iTransformer
from .NSTransformer import NSTransformer
from .PatchTST import PatchTST
from .SegRNN import SegRNN
from .TimeMixer import TimeMixer
from .TimesNet import TimesNet
from .Transformer import Transformer
from .Tsmixer import TSMixer
from .Reformer import Reformer
from .FreTS import FreTS
from .MICN import MICN
from .WPMixer import WPMixer

#Ex forecasting models

from .TFT import TFT
from .TiDE import TiDE
from .TimeXer import TimeXer
from .Tsmixer import TSMixerExt
from .BiTCN import BiTCN
from .NBEATSX import NBeatsNetX

from .emission_head import *
from .ex_Linear import ex_Linear,ex_MLP,ex_HT
from .ex_LSTM import ex_LSTM
from .ex_Mixer import ex_Mixer
from .ex_Transformer import ex_Transformer


class DummyScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scale_ = None
        self.mean_ = None

    def fit(self, X, y=None):
        self.scale_ = np.ones(X.shape[1])
        self.mean_ = np.zeros(X.shape[1])
        return self

    def transform(self, X, y=None):
        return X

class QuantileRegressor(ABC):
    """Class representing a regressor with optional scaler"""

    def __init__(self, quantiles: Optional[List[float]], X_scaler: Optional[Any] = None, y_scaler: Optional[Any] = None,
                 model: Optional[List[Any]] = None):
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.model = model
        self.quantiles = check_quantile_list(quantiles)
        self.name = self.__class__.__name__

    @abstractmethod
    def set_params(self, input_dim: int) -> "QuantileRegressor":
        """Set parameters for inference"""


class QR(QuantileRegressor):
    """Quantile linear program"""

    def __init__(self, quantiles: List[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [QuantileLinearProgram(quantile=q) for q in self.quantiles]
        return self


class L1QR(QuantileRegressor):
    """Classical quantile regression with optional scaler"""

    def __init__(self, quantiles: List[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [QuantileRegression(quantile=q, alpha=0, solver="highs") for q in self.quantiles]
        return self


class QSVR(QuantileRegressor):
    """Quantile Support Vector regression with cvxpy"""

    def __init__(self, quantiles: List[float], variant: Literal["sparse", "epsilon"] = "epsilon"):
        super().__init__(quantiles=quantiles)
        self.variant = variant

    def set_params(self, input_dim: int):
        self.model = [QuantileSupportVectorRegression(quantile=q, variant=self.variant) for q in self.quantiles]
        return self


class QGBR(QuantileRegressor):
    """Quantile Gradient Boosting Regression from sklearn"""

    def __init__(self, quantiles: List[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [GradientBoostingRegressor(loss='quantile', alpha=q, random_state=0) for q in self.quantiles]
        return self
     
class QXgboost(QuantileRegressor):
    """Quantile Gradient Boosting Regression from sklearn"""

    def __init__(self, quantiles: List[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [CustomQuantileLossXGBRegressor(quantile=q, random_state=0) for q in self.quantiles]
        return self



class QFFNN(QuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float]):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [models.pytorch.PytorchRegressor(
            model=models.pytorch.FeedForwardNeuralNetwork(input_dim),
            loss_function=pytorchtools.PinballScore(q)) for q in self.quantiles]
        return self


class QLSTM(QuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float]):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [models.pytorch.PytorchRegressor(
            model=models.pytorch.LongShortTermMemory(input_dim),
            loss_function=pytorchtools.PinballScore(q)) for q in self.quantiles]
        return self


class QCNN(QuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float]):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [models.pytorch.PytorchRegressor(
            model=models.pytorch.ConvolutionalNeuralNetwork(input_dim),
            loss_function=pytorchtools.PinballScore(q)) for q in self.quantiles]
        return self




class MultiQuantileRegressor(ABC):
    """Class representing a regressor with optional scaler"""

    def __init__(self, quantiles: List[float],  X_scaler: Optional[Any] = None, y_scaler: Optional[Any] = None, 
                 ex_model: Optional[Any] = None,device=None):
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        if ex_model:
            self.ex_model = ex_model
            self.name = self.__class__.__name__+'_'+ex_model
        else:
            self.ex_model = None
            self.name = self.__class__.__name__
        self.quantiles = check_quantile_list(quantiles)
        self.device = device

    @abstractmethod
    def set_params(self, input_dim: int) -> "MultiQuantileRegressor":
        """Set parameters for inference
        Parameters
        ----------
        input_dim
            input dimension of data
        Returns
        -------
        self
        """


class MQCE(MultiQuantileRegressor):
    """Multi Quantile Boostrap Conditional Error Model"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None
    def downsample(self,X):
        self.pca = PCA(n_components=min(100,int(0.1*X.shape[-1])))
        data_pca = self.pca.fit_transform(X)
        return data_pca

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [ConditionalErrorQuantile(quantiles=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))


class MQBCE(MultiQuantileRegressor):
    """Multi Quantile Boostrap Conditional Error Model"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None
    def downsample(self,X):
        self.pca = PCA(n_components=min(100,int(0.1*X.shape[-1])))
        data_pca = self.pca.fit_transform(X)
        return data_pca
    def test_downsample(self,X):
        X = self.pca.transform(X)
        return X

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [BootstrapConditionalErrorQuantile(quantiles=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        X = self.downsample(X)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        X = self.test_downsample(X)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))


class MQLP(MultiQuantileRegressor):
    """Multi Quantile linear program"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [MultiQuantileLinearProgram(quantiles=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(pred)
        return(np.array(preds))
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))


class MQKNNR(MultiQuantileRegressor):
    """Quantile K-Nearst Neighbor Regression from sklearn-quantile"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [KNeighborsQuantileRegressor(q=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))

class MQRFR(MultiQuantileRegressor):
    """Quantile Random Forest regression from sklearn-quantile"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None

    def downsample(self,X):
        self.pca = PCA(n_components=min(100,int(0.1*X.shape[-1])))
        data_pca = self.pca.fit_transform(X)
        return data_pca
    def test_downsample(self,X):
        X = self.pca.transform(X)
        return X

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [RandomForestQuantileRegressor(n_estimators = 1,q=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        X = self.downsample(X)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        X = self.test_downsample(X)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))


class MQSRFR(MultiQuantileRegressor):
    """Quantile Sample Random Forest regression from sklearn-quantile"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None
    def downsample(self,X):
        self.pca = PCA(n_components=min(100,int(0.1*X.shape[-1])))
        data_pca = self.pca.fit_transform(X)
        return data_pca
    def test_downsample(self,X):
        X = self.pca.transform(X)
        return X

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [SampleRandomForestQuantileRegressor(n_estimators = 10,q=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        X = self.downsample(X)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        X = self.test_downsample(X)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))
    
class MQERT(MultiQuantileRegressor):
    """Quantile Extremely Random Trees regression from sklearn-quantile"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None
    def downsample(self,X):
        self.pca = PCA(n_components=min(100,int(0.1*X.shape[-1])))
        data_pca = self.pca.fit_transform(X)
        return data_pca
    def test_downsample(self,X):
        X = self.pca.transform(X)
        return X

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [ExtraTreesQuantileRegressor(n_estimators = 10,q=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        X = self.downsample(X)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        X = self.test_downsample(X)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))

class MQSERT(MultiQuantileRegressor):
    """Quantile Sample Extremely Random Trees regression from sklearn-quantile"""

    def __init__(self, quantiles):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)
        self.models = None
    def downsample(self,X):
        self.pca = PCA(n_components=min(100,int(0.1*X.shape[-1])))
        data_pca = self.pca.fit_transform(X)
        return data_pca
    
    def test_downsample(self,X):
        X = self.pca.transform(X)
        return X

    def set_params(self, seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.models = [SampleExtraTreesQuantileRegressor(n_estimators = 10,q=self.quantiles) for _ in range(seq_len*dim)]
        return self
    def fit(self,X,y):
        X = X.reshape(X.shape[0],-1)
        X = self.downsample(X)
        for i in tqdm(range(len(self.models))):
            self.models[i].fit(X,y[:,i])
    def predict(self,X):
        X = X.reshape(X.shape[0],-1)
        X = self.test_downsample(X)
        preds = []
        for i in tqdm(range(len(self.models))):
            pred = self.models[i].predict(X)
            preds.append(np.transpose(pred, (1, 0)))
        preds = np.array(preds)
        return(np.transpose(preds.reshape(self.seq_len,self.dim,preds.shape[1],preds.shape[2]),(2,0,1,3)))


########Multi quantile forecasting methods#########
########Time series forecasting methods#########

def define_ex_model(ex_model_name,configs):
    if ex_model_name == 'ex_Linear':
        return ex_Linear(configs)
    elif ex_model_name == 'ex_MLP':
        return ex_MLP(configs)
    elif ex_model_name == 'ex_LSTM':
        return ex_LSTM(configs)
    elif ex_model_name == 'ex_Mixer':
        return ex_Mixer(configs)
    elif ex_model_name == 'ex_Transformer':
        return ex_Transformer(configs)
    elif ex_model_name == 'ex_HT':
        return ex_HT(configs)
    else:
        return None

class MQDLinear(MultiQuantileRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=DLinear(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self



class MQMLP(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)
        

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=MLP(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self


class MQLSTM(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=LSTMF(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self


class MQCNN(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model= CNN(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    
    
class MQTransformer(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device,
            )

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Transformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    

class MQLSTNet(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=LSTNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self


class MQNBEATS(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model= NBeatsNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    

class MQInformer(MultiQuantileRegressor):
    """Informer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Informer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    
class MQAutoformer(MultiQuantileRegressor):
    """Fedformer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Autoformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQFedformer(MultiQuantileRegressor):
    """Fedformer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Fedformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    
class MQFiLM(MultiQuantileRegressor):
    """FiLM optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=FiLM(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    
class MQiTransformer(MultiQuantileRegressor):
    """iTransformer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=iTransformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQNSTransformer(MultiQuantileRegressor):
    """NSTransformer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=NSTransformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQPatchTST(MultiQuantileRegressor):
    """PatchTST optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=PatchTST(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQSegRNN(MultiQuantileRegressor):
    """SegRNN optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=SegRNN(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQTimeMixer(MultiQuantileRegressor):
    """TimeMixer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TimeMixer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    
class MQTimesNet(MultiQuantileRegressor):
    """TimesNet optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TimesNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQTSMixer(MultiQuantileRegressor):
    """TSMixer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TSMixer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQFreTS(MultiQuantileRegressor):
    """TSMixer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=FreTS(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQReformer(MultiQuantileRegressor):
    """TSMixer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Reformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQWaveNet(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
             ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=WaveNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQMICN(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
             ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=MICN(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class MQWPMixer(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
             ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=WPMixer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self
    
class MQTimeXer(MultiQuantileRegressor):
    """TimeXer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            ex_model = ex_model,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TimeXer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device),device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

########Time series forecasting with aux methods#########

class MQNBEATSX(MultiQuantileRegressor):
    """TSMixer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=NBeatsNetX(configs).to(self.device),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self
    
class MQBiTCN(MultiQuantileRegressor):
    """BiTCN optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=BiTCN(configs).to(self.device),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self

class MQTFT(MultiQuantileRegressor):
    """TFT optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TFT(configs).to(self.device),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self

class MQTiDE(MultiQuantileRegressor):
    """TiDE optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TiDE(configs).to(self.device),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self

class MQTSMixerExt(MultiQuantileRegressor):
    """TSMixerExt optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TSMixerExt(configs).to(self.device),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self



####################Point Forecasting Method###########################

# class PointRegressor(ABC):
#     """Class representing a regressor with optional scaler"""

#     def __init__(self,  X_scaler: Optional[Any] = None, y_scaler: Optional[Any] = None, model: Optional[Any] = None, loss_function:Optional[Any] = None,device=None,ifshift=True):
#         self.X_scaler = StandardScaler()
#         self.y_scaler = StandardScaler()
#         self.model = model
#         self.name = self.__class__.__name__
#         self.loss_function = loss_function
#         self.device = device
#         self.ifshift = ifshift


#     @abstractmethod
#     def set_params(self, input_dim: int) -> "PointRegressor":
#         """Set parameters for inference
#         Parameters
#         ----------
#         input_dim
#             input dimension of data
#         Returns
#         -------
#         self
#         """

class PointRegressor(ABC):
    """Class representing a regressor with optional scaler"""

    def __init__(self, X_scaler: Optional[Any] = None, y_scaler: Optional[Any] = None, loss_function = None,
                 ex_model: Optional[Any] = None,device=None):
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        if ex_model:
            self.ex_model = ex_model
            self.name = self.__class__.__name__+'_'+ex_model
        else:
            self.ex_model = None
            self.name = self.__class__.__name__
        self.device = device
        self.loss_function = loss_function

    @abstractmethod
    def set_params(self, input_dim: int) -> "PointRegressor":
        """Set parameters for inference
        Parameters
        ----------
        input_dim
            input dimension of data
        Returns
        -------
        self
        """

class PDLinear(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=DLinear(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PMLP(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=MLP(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PLSTM(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=LSTMF(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PCNN(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=CNN(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PTransformer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Transformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PLSTNet(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=LSTNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PNBEATS(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=NBeatsNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PInformer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Informer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PAutoformer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Autoformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PFedformer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Fedformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PFiLM(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=FiLM(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PiTransformer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=iTransformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PNSTransformer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=NSTransformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PPatchTST(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=PatchTST(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PSegRNN(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=SegRNN(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PTimeMixer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TimeMixer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PTimesNet(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TimesNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PTSMixer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TSMixer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PFreTS(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=FreTS(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PReformer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=Reformer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PWaveNet(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=WaveNet(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PMICN(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=MICN(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

class PTimeXer(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            ex_model = ex_model,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TimeXer(configs).to(self.device),
            ex_model = define_ex_model(self.ex_model,configs),
            loss_function=self.loss_function,device =self.device,ex_learning_rate = configs.ex_learning_rate)
        return self

########Time series forecasting with aux methods#########

class PNBEATSX(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=NBeatsNetX(configs).to(self.device),
            loss_function=self.loss_function,device =self.device)
        return self

class PBiTCN(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=BiTCN(configs).to(self.device),
            loss_function=self.loss_function,device =self.device)
        return self

class PTFT(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TFT(configs).to(self.device),
            loss_function=self.loss_function,device =self.device)
        return self

class PTiDE(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TiDE(configs).to(self.device),
            loss_function=self.loss_function,device =self.device)
        return self

class PTSMixerExt(PointRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),device = None,ex_model = None):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            loss_function=loss_function,
            device = device)

    def set_params(self,configs):
        self.model = models.pytorch.PytorchRegressor(
            model=TSMixerExt(configs).to(self.device),
            loss_function=self.loss_function,device =self.device)
        return self



# class PFFNN(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE()):
#         super().__init__(
#             loss_function = loss_function,
#             )

#     def set_params(self, input_dim: int, external_features_diminsion: int):
#         self.model = models.pytorch.PytorchRegressor(
#             model=models.pytorch.FeedForwardNeuralNetwork(input_dim, n_output=1),
#             loss_function=self.loss_function)
#         return self


# class PLSTM(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=models.pytorch.LongShortTermMemory(input_dim, external_features_diminsion,n_output=1),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self


# class PCNN(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE()):
#         super().__init__(
#             loss_function = loss_function)

#     def set_params(self, input_dim: int,external_features_diminsion: int):
#         self.model = models.pytorch.PytorchRegressor(
#             model=models.pytorch.ConvolutionalNeuralNetwork(input_dim, n_output=1),
#             loss_function=self.loss_function)
#         return self
    


# class PTransformer(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=models.pytorch.TransformerTS(input_dim,external_features_diminsion, n_output=1),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self
    

# class PLSTN(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=models.pytorch.LSTN(input_dim,external_features_diminsion, n_output=1),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self


# class PWaveNet(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=models.pytorch.WaveNet(input_dim, external_features_diminsion,n_output=1),
#             loss_function=self.loss_function,ifshift= self.ifshift)
#         return self
    

# class PNBEATS(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,device,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,
#             device = device,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=models.pytorch.NBeatsNet(input_dim, external_features_diminsion,n_output=1,device = self.device),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self


# class PInformer(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int,configs):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=Informer(configs),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self

# class PAutoformer(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int,configs):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=Autoformer(configs),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self

# class PFedformer(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int,configs):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=Fedformer(configs),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self

# class PDLinear(PointRegressor):
#     """Feedforward neural network optimizing quantile loss from pytorch"""

#     def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
#         super().__init__(
#             loss_function = loss_function,ifshift= ifshift)

#     def set_params(self, input_dim: int,external_features_diminsion: int,configs):
#         input_dim = 1
#         self.model = models.pytorch.PytorchRegressor(
#             model=DLinear(configs),
#             loss_function=self.loss_function,ifshift = self.ifshift)
#         return self



