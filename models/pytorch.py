from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from torch import nn, optim, from_numpy, no_grad, load, zeros, unsqueeze
from torch.utils.data import DataLoader, TensorDataset

from utils.pytorchtools import EarlyStopping, PinballScore


class PytorchRegressor(BaseEstimator, RegressorMixin):
    """Class representing a pytorch regression module"""

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 loss_function: Any = PinballScore(),
                 batch_size: int = 32,
                 epochs: int = 1000,
                 patience: int = 10,
                 validation_ratio: float = 0.2):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optim.Adam(params=model.parameters())
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_ratio = validation_ratio

    def forward(self, X):
        return self.model(X)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y.reshape(-1, 1), shuffle=False, test_size=self.validation_ratio)
        training_loader = DataLoader(dataset=TensorDataset(from_numpy(X_tr).float(), from_numpy(y_tr).float()),
                                     batch_size=self.batch_size,
                                     shuffle=True)
        validation_loader = DataLoader(dataset=TensorDataset(from_numpy(X_val).float(), from_numpy(y_val).float()),
                                       batch_size=self.batch_size,
                                       shuffle=False)
        early_stopping = EarlyStopping(patience=self.patience, verbose=False)
        for _ in range(self.epochs):

            # Training mode
            self.model.train()
            for X_tr_batch, y_tr_batch in training_loader:
                # Clear gradient buffers
                self.optimizer.zero_grad()

                # get output from the model, given the inputs
                pred = self.model(X_tr_batch)

                # get loss for the predicted output
                loss = self.loss_function(pred, y_tr_batch)

                # get gradients w.r.t to parameters
                loss.backward()

                # update parameters
                self.optimizer.step()

            # Evaluation mode
            self.model.eval()
            validation_losses = []
            for X_val_batch, y_val_batch in validation_loader:
                pred = self.model(X_val_batch)
                loss = self.loss_function(pred, y_val_batch)
                validation_losses.append(loss.item())

            # Stop if patience is reached
            early_stopping(np.average(validation_losses), self.model)
            if early_stopping.early_stop:
                break

        # Load saved model
        self.model.load_state_dict(load('checkpoint.pt'))

        # Clean up checkpoint
        early_stopping.clean_up_checkpoint()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with no_grad():
            return self.model(from_numpy(X).float()).data.numpy().squeeze()


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
    def __init__(self, n_features: int, n_neurons: int = 50, n_layers: int = 1, n_output: int = 1):
        super(LongShortTermMemory, self).__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_neurons, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(n_neurons, n_output)

    def forward(self, X_batch):
        batch_size = len(X_batch)
        hidden_state = zeros(self.n_layers, batch_size, self.n_neurons)
        cell_state = zeros(self.n_layers, batch_size, self.n_neurons)
        output, _ = self.lstm(unsqueeze(X_batch, dim=1), (hidden_state, cell_state))
        return self.linear(output[:, -1])


class ConvolutionalNeuralNetwork(nn.Module):
    """Adapted from (https://github.com/nidhi-30/CNN-Regression-Pytorch/blob/master/1095526_1dconv.ipynb)"""

    def __init__(self, n_features: int, batch_size: int = 32, n_output: int = 1):
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
