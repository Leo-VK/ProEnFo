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
from preprocessing.quantile_format import check_quantile_list
from utils import randomness, pytorchtools

randomness.set_random_seeds(0)  # reproducibility


class QuantileRegressor(ABC):
    """Class representing a regressor with optional scaler"""

    def __init__(self, quantiles: Optional[list[float]], scaler: Optional[Any] = None,
                 model: Optional[list[Any]] = None):
        self.scaler = scaler
        self.model = model
        self.quantiles = check_quantile_list(quantiles)
        self.name = self.__class__.__name__

    @abstractmethod
    def set_params(self, input_dim: int) -> "QuantileRegressor":
        """Set parameters for inference"""


class QR(QuantileRegressor):
    """Quantile linear program"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [QuantileLinearProgram(quantile=q) for q in self.quantiles]
        return self


class L1QR(QuantileRegressor):
    """Classical quantile regression with optional scaler"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [QuantileRegression(quantile=q, alpha=0, solver="highs") for q in self.quantiles]
        return self


class QSVR(QuantileRegressor):
    """Quantile Support Vector regression with cvxpy"""

    def __init__(self, quantiles: list[float], variant: Literal["sparse", "epsilon"] = "epsilon"):
        super().__init__(quantiles=quantiles)
        self.variant = variant

    def set_params(self, input_dim: int):
        self.model = [QuantileSupportVectorRegression(quantile=q, variant=self.variant) for q in self.quantiles]
        return self


class QGBR(QuantileRegressor):
    """Quantile Gradient Boosting Regression from sklearn"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [GradientBoostingRegressor(loss='quantile', alpha=q, random_state=0) for q in self.quantiles]
        return self


class QFFNN(QuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: list[float]):
        super().__init__(
            scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [models.pytorch.PytorchRegressor(
            model=models.pytorch.FeedForwardNeuralNetwork(input_dim),
            loss_function=pytorchtools.PinballScore(q)) for q in self.quantiles]
        return self


class QLSTM(QuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: list[float]):
        super().__init__(
            scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [models.pytorch.PytorchRegressor(
            model=models.pytorch.LongShortTermMemory(input_dim),
            loss_function=pytorchtools.PinballScore(q)) for q in self.quantiles]
        return self


class QCNN(QuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: list[float]):
        super().__init__(
            scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = [models.pytorch.PytorchRegressor(
            model=models.pytorch.ConvolutionalNeuralNetwork(input_dim),
            loss_function=pytorchtools.PinballScore(q)) for q in self.quantiles]
        return self


class MultiQuantileRegressor(ABC):
    """Class representing a regressor with optional scaler"""

    def __init__(self, quantiles: list[float], scaler: Optional[Any] = None, model: Optional[Any] = None):
        self.scaler = scaler
        self.model = model
        self.quantiles = check_quantile_list(quantiles)
        self.name = self.__class__.__name__

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

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = ConditionalErrorQuantile(quantiles=self.quantiles)
        return self


class MQBCE(MultiQuantileRegressor):
    """Multi Quantile Boostrap Conditional Error Model"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = BootstrapConditionalErrorQuantile(quantiles=self.quantiles)
        return self


class MQLP(MultiQuantileRegressor):
    """Multi Quantile linear program"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = MultiQuantileLinearProgram(quantiles=self.quantiles)
        return self


class MQKNNR(MultiQuantileRegressor):
    """Quantile K-Nearst Neighbor Regression from sklearn-quantile"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = KNeighborsQuantileRegressor(n_neighbors=20, q=self.quantiles)
        return self


class MQRFR(MultiQuantileRegressor):
    """Quantile Random Forest regression from sklearn-quantile"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = RandomForestQuantileRegressor(n_estimators=100, q=self.quantiles)
        return self


class MQSRFR(MultiQuantileRegressor):
    """Quantile Sample Random Forest regression from sklearn-quantile"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = SampleRandomForestQuantileRegressor(n_estimators=100, q=self.quantiles)
        return self


class MQERT(MultiQuantileRegressor):
    """Quantile Extremely Random Trees regression from sklearn-quantile"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = ExtraTreesQuantileRegressor(n_estimators=100, q=self.quantiles)
        return self


class MQSERT(MultiQuantileRegressor):
    """Quantile Sample Extremely Random Trees regression from sklearn-quantile"""

    def __init__(self, quantiles: list[float]):
        super().__init__(quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = SampleExtraTreesQuantileRegressor(n_estimators=100, q=self.quantiles)
        return self


class MQFFNN(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: list[float]):
        super().__init__(
            scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.FeedForwardNeuralNetwork(input_dim, n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles))
        return self


class MQLSTM(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: list[float]):
        super().__init__(
            scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.LongShortTermMemory(input_dim, n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles))
        return self


class MQCNN(MultiQuantileRegressor):
    """Feedforward neural network optimizing quantile loss from pytorch"""

    def __init__(self, quantiles: list[float]):
        super().__init__(
            scaler=StandardScaler(),
            quantiles=quantiles)

    def set_params(self, input_dim: int):
        self.model = models.pytorch.PytorchRegressor(
            model=models.pytorch.ConvolutionalNeuralNetwork(input_dim, n_output=len(self.quantiles)),
            loss_function=pytorchtools.PinballLoss(self.quantiles))
        return self
