from abc import ABC, abstractmethod
from typing import Any, Union, Optional, Literal

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor as QuantileRegression
from sklearn.preprocessing import StandardScaler
from sklearn_quantile import KNeighborsQuantileRegressor, RandomForestQuantileRegressor, \
    SampleRandomForestQuantileRegressor, ExtraTreesQuantileRegressor, SampleExtraTreesQuantileRegressor

import models.pytorch
from models.conditional_error_model import ConditionalErrorQuantile, BootstrapConditionalErrorQuantile
from models.optimization_model import QuantileLinearProgram, QuantileSupportVectorRegression, \
    MultiQuantileLinearProgram
from models.boosting import CustomQuantileLossXGBRegressor
from models.GAM import QuantileLossGAM
from preprocessing.quantile_format import check_quantile_list
from utils import randomness, pytorchtools

from typing import List, Tuple, Dict
import numpy as np 
import torch.nn as nn
import torch
from torch.nn import functional as F

from sklearn.base import BaseEstimator, TransformerMixin
from models.Informer import Informer
from models.Autoformer import Autoformer
from models.DLinear import DLinear
from models.NLinear import NLinear
from models.Fedformer.FEDformer import Fedformer

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

randomness.set_random_seeds(0)  # reproducibility


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

class QGAM(QuantileRegressor):
    """Quantile GAM Regression"""

    def __init__(self, quantiles: List[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [QuantileLossGAM(quantile=q, n_features=input_dim) for q in self.quantiles]
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

    def __init__(self, quantiles: List[float],  X_scaler: Optional[Any] = None, y_scaler: Optional[Any] = None, model: Optional[Any] = None,device=None):
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.model = model
        self.quantiles = check_quantile_list(quantiles)
        self.name = self.__class__.__name__
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
    """Multi Quantile linear program"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = ConditionalErrorQuantile(quantiles=self.quantiles)
        return self


class MQBCE(MultiQuantileRegressor):
    """Multi Quantile Boostrap Conditional Error Model"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = BootstrapConditionalErrorQuantile(quantiles=self.quantiles)
        return self


class MQLP(MultiQuantileRegressor):
    """Multi Quantile linear program"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = MultiQuantileLinearProgram(quantiles=self.quantiles)
        return self


class MQKNNR(MultiQuantileRegressor):
    """Quantile K-Nearst Neighbor Regression from sklearn-quantile"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = KNeighborsQuantileRegressor(n_neighbors=20, q=self.quantiles)
        return self


class MQRFR(MultiQuantileRegressor):
    """Quantile Random Forest regression from sklearn-quantile"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = RandomForestQuantileRegressor(n_estimators=100, q=self.quantiles, random_state=42)
        return self


class MQSRFR(MultiQuantileRegressor):
    """Quantile Sample Random Forest regression from sklearn-quantile"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = SampleRandomForestQuantileRegressor(n_estimators=100, q=self.quantiles, random_state=42)
        return self


class MQERT(MultiQuantileRegressor):
    """Quantile Extremely Random Trees regression from sklearn-quantile"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = ExtraTreesQuantileRegressor(n_estimators=100, q=self.quantiles,random_state=42)
        return self


class MQSERT(MultiQuantileRegressor):
    """Quantile Sample Extremely Random Trees regression from sklearn-quantile"""

    def __init__(self, quantiles: List[float]):
        super().__init__(X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),quantiles=quantiles)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = SampleExtraTreesQuantileRegressor(n_estimators=100, q=self.quantiles, random_state=42)
        return self


class MQFFNN(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int, external_features_diminsion: int):
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.FeedForwardNeuralNetwork(input_dim, n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self


class MQLSTM(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.LongShortTermMemory(input_dim, external_features_diminsion,n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self


class MQCNN(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.ConvolutionalNeuralNetwork(input_dim, n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self
    


class MQTransformer(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.TransformerTS(input_dim,external_features_diminsion, n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self
    

class MQLSTN(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.LSTN(input_dim,external_features_diminsion, n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self


class MQWaveNet(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.WaveNet(input_dim, external_features_diminsion,n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self
    

class MQNBEATS(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.NBeatsNet(input_dim, external_features_diminsion,n_output=len(self.quantiles),device = self.device),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self
    

class MQInformer(MultiQuantileRegressor):
    """Informer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=Informer(configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self
    
class MQDLinear(MultiQuantileRegressor):
    """DLinear optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=DLinear(configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self


class MQNLinear(MultiQuantileRegressor):
    """NLinear optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=NLinear(configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self

class MQAutoformer(MultiQuantileRegressor):
    """Fedformer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=Autoformer(configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self

class MQFedformer(MultiQuantileRegressor):
    """Fedformer optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: List[float],device):
        super().__init__(
            X_scaler=StandardScaler(),
            y_scaler=StandardScaler(),
            quantiles=quantiles,
            device = device)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=Fedformer(configs),
            loss_function=pytorchtools.PinballLoss(self.quantiles,self.device))
        return self
    





####################Point Forecasting Method###########################

class PointRegressor(ABC):
    """Class representing a regressor with optional scaler"""

    def __init__(self,  X_scaler: Optional[Any] = None, y_scaler: Optional[Any] = None, model: Optional[Any] = None, loss_function:Optional[Any] = None,device=None,ifshift=True):
        if not ifshift:
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
        else:
            self.X_scaler = DummyScaler()
            self.y_scaler = DummyScaler()
        # self.X_scaler = StandardScaler()
        # self.y_scaler = StandardScaler()
        self.model = model
        self.name = self.__class__.__name__
        self.loss_function = loss_function
        self.device = device
        self.ifshift = ifshift


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

class FFNN(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE()):
        super().__init__(
            loss_function = loss_function,
            )

    def set_params(self, input_dim: int, external_features_diminsion: int):
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.FeedForwardNeuralNetwork(input_dim, n_output=1),
            loss_function=self.loss_function)
        return self


class LSTM(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.LongShortTermMemory(input_dim, external_features_diminsion,n_output=1),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self


class CNN(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE()):
        super().__init__(
            loss_function = loss_function)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.ConvolutionalNeuralNetwork(input_dim, n_output=1),
            loss_function=self.loss_function)
        return self
    


class Transformer(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.TransformerTS(input_dim,external_features_diminsion, n_output=1),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self
    

class LSTN(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.LSTN(input_dim,external_features_diminsion, n_output=1),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self


class WaveNet(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.WaveNet(input_dim, external_features_diminsion,n_output=1),
            loss_function=self.loss_function,ifshift= self.ifshift)
        return self
    

class NBEATS(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,device,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,
            device = device,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.NBeatsNet(input_dim, external_features_diminsion,n_output=1,device = self.device),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self


class PInformer(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=Informer(configs),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self

class PAutoformer(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=Autoformer(configs),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self

class PFedformer(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=Fedformer(configs),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self

class PNLinear(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=NLinear(configs),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self

class PDLinear(PointRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self,loss_function = pytorchtools.MSE(),ifshift = False):
        super().__init__(
            loss_function = loss_function,ifshift= ifshift)

    def set_params(self, input_dim: int,external_features_diminsion: int,configs):
        input_dim = 1
        self.model = models.pytorch.PytorchRegressor(
            model=DLinear(configs),
            loss_function=self.loss_function,ifshift = self.ifshift)
        return self

