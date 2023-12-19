import pandas as pd

from models.model_init import QuantileRegressor, MultiQuantileRegressor, PointRegressor
import torch
import numpy as np
from typing import List
import os


def quantile_forecasting(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         method: QuantileRegressor,
                         external_features_diminsion: int) -> pd.DataFrame:
    """Quantile forecasting workflow"""
    # Scale data if required
    if method.X_scaler:
        X_train_val = method.scaler.fit_transform(X_train.values)
        X_test_val = method.scaler.transform(X_test.values)
    else:
        X_train_val = X_train.values
        X_test_val = X_test.values
    y_train_val = y_train.values

    # Set params
    method.set_params(X_train.shape[1])

    # Initialize, train and test models
    preds = {}
    print(f"\n{method.name}")
    for q, quantile in enumerate(method.quantiles):
        method.model[q].fit(X_train_val, y_train_val)
        preds[quantile] = method.model[q].predict(X_test_val)
        print(f"\r q={quantile}", end="")

    return pd.DataFrame(preds, index=X_test.index)


def multi_quantile_forecasting(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_test: pd.DataFrame,
                               method: MultiQuantileRegressor,
                               external_features_diminsion: int,
                               target_lags: List,
                               data_name,
                               strategy_name,
                               device) -> pd.DataFrame:
    """Multi-quantile forecasting workflow"""
    # Scale data if required
    if method.X_scaler:
        X_train_val = method.X_scaler.fit_transform(X_train.values)
        X_test_val = method.X_scaler.transform(X_test.values)
        y_train_val = method.y_scaler.fit_transform(y_train.values.reshape(-1,1)).reshape(-1)
    else:
        X_train_val = X_train.values
        X_test_val = X_test.values
        y_train_val = y_train.values
    # y_train_val = y_train.values
    
    # Set params
    method.set_params(X_train.shape[1],external_features_diminsion)

    # Initialize, train and test models
    print(method.name)
    if method.name in ['MQCNN','MQLSTM','MQFFNN','MQTransformer','MQLSTN','MQWaveNet','MQNBEATS','MQInformer','MQDLinear','MQNLinear','MQFedformer','MQAutoformer']:
        method.model.model = method.model.model.to(device)
        X_train_val = torch.Tensor(X_train_val).to(device)
        X_test_val = torch.Tensor(X_test_val).to(device)
        y_train_val = torch.Tensor(y_train_val).to(device)
        method.model.fit(X_train_val, y_train_val,target_lags)
        print('Save the model')
        if os.path.isdir('./pkl_folder')!=True:
            os.mkdir('./pkl_folder')
        torch.save(method.model.model.state_dict(), './pkl_folder/'+strategy_name+'_'+data_name+'_'+method.name+'.pkl')
        # # if method.name in ['MQTransformer']:
        # #     if len(X_test_val)>=256:
        # #         X_test_val_list = torch.split(X_test_val, 256)
        # #         pred_list = []
        # #         for i in range(len(X_test_val_list)):
        # #             pred_list.append(method.model.predict(X_test_val_list[i],target_lags))
        # #         preds = np.concatenate(pred_list, axis=0)
        # else:
        preds = method.model.predict(X_test_val,target_lags)
    else:
        method.model.fit(X_train_val, y_train_val)
        preds = method.model.predict(X_test_val)
    # prediction #
    # preds = method.model.predict(X_test_val,target_lags)
    if len(preds) == len(method.quantiles):
        preds = preds.T
    preds = preds.cpu().detach().numpy()*method.y_scaler.scale_+method.y_scaler.mean_
    return pd.DataFrame(preds, columns=method.quantiles, index=X_test.index)




def point_forecasting(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_test: pd.DataFrame,
                               method: PointRegressor,
                               external_features_diminsion: int,
                               target_lags: List,
                               data_name,
                               strategy_name,
                               device) -> pd.DataFrame:
    """Multi-quantile forecasting workflow"""
    # Scale data if required
    if method.X_scaler:
        X_train_val = method.X_scaler.fit_transform(X_train.values)
        X_test_val = method.X_scaler.transform(X_test.values)
        y_train_val = method.y_scaler.fit_transform(y_train.values.reshape(-1,1)).reshape(-1)
    else:
        X_train_val = X_train.values
        X_test_val = X_test.values
        y_train_val = y_train.values
    # y_train_val = y_train.values
    # Set params
    method.set_params(X_train.shape[1],external_features_diminsion)
    # Initialize, train and test models
    print(method.name)
    if method.name in ['CNN','LSTM','FFNN','Transformer','LSTN','WaveNet','NBEATS']:
        method.model.model = method.model.model.to(device)
        X_train_val = torch.Tensor(X_train_val).to(device)
        X_test_val = torch.Tensor(X_test_val).to(device)
        y_train_val = torch.Tensor(y_train_val).to(device)
    method.model.fit(X_train_val, y_train_val,target_lags)

    # prediction #
    print('Save the model')
    if os.path.isdir('./pkl_folder')!=True:
        os.mkdir('./pkl_folder')
    torch.save(method.model.model.state_dict(), './pkl_folder/'+strategy_name+'_'+data_name+'_'+method.name+'.pkl')
    if method.name in ['Transformer']:
        if len(X_test_val)>=256:
            X_test_val_list = torch.split(X_test_val, 256)
            pred_list = []
            for i in range(len(X_test_val_list)):
                pred_list.append(method.model.predict(X_test_val_list[i],target_lags))
            preds = np.concatenate(pred_list, axis=0)
        else:
            preds = method.model.predict(X_test_val,target_lags)
    else:
        preds = method.model.predict(X_test_val,target_lags)
    preds = preds*method.y_scaler.scale_+method.y_scaler.mean_
    return pd.DataFrame(preds, index=X_test.index)
