import pandas as pd

from models.model_init import QuantileRegressor, MultiQuantileRegressor, PointRegressor
import torch
import numpy as np
from typing import List
import os
import xarray as xr


def quantile_forecasting(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         method: QuantileRegressor,
                         external_features_diminsion: int,
                         target_lags: List,
                         target_preds:List,
                         data_name,
                         strategy_name) -> pd.DataFrame:
    """Quantile forecasting workflow"""
    # Scale data if required
    if method.X_scaler:
        X_train_val = method.scaler.fit_transform(X_train.values)
        X_test_val = method.scaler.transform(X_test.values)
        y_train_val = method.y_scaler.fit_transform(y_train.values.reshape(-1,1)).reshape(y_train.values.shape[0],-1)
    else:
        X_train_val = X_train.values
        X_test_val = X_test.values
        y_train_val = y_train.values

    # Set params
    method.set_params(X_train.shape[1])

    # Initialize, train and test models
    print(method.name)

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
                               target: List,
                               external_features_diminsion: int,
                               target_lags: List,
                               target_preds:List,
                               data_name,
                               strategy_name,
                               device,
                               configs = None) -> pd.DataFrame:
    """Multi-quantile forecasting workflow"""
    # Scale data if required
    # if method.y_scaler:
    # #     X_train_val = method.X_scaler.fit_transform(X_train.values)
    # #     X_test_val = method.X_scaler.transform(X_test.values)
    #     y_train_val = method.y_scaler.fit_transform(y_train.values)
    # else:
    # X_train_val = X_train.values
    # X_test_val = X_test.values
    y_train_val = y_train
    # Initialize, train and test models
    print(method.name)
    if method.name.split("_")[0] in ['MQDLinear','MQLSTM','MQMLP','MQCNN','MQTransformer','MQLSTNet',
                                     'MQInformer','MQAutoformer','MQFedformer','MQFiLM','MQiTransformer',
                                     'MQNSTransformer','MQPatchTST','MQSegRNN','MQTimeMixer','MQTimesNet',
                                     'MQTSMixer','MQFreTS','MQReformer','MQNBEATS','MQNBEATSX',
                                     'MQTSMixerExt','MQWaveNet','MQBiTCN','MQTiDE','MQTimeXer','MQMICN',
                                     'MQWPMixer','MQTFT']:
        X_train_val = X_train.values
        X_test_val = X_test.values
        # Set params
        method.set_params(configs)
        # X_train_val = torch.Tensor(X_train_val).to(device)
        # X_test_val = torch.Tensor(X_test_val).to(device)
        y_train_val = torch.Tensor(y_train_val).to(device)
        method.model,X_scaler = method.model.fit(X_train_val, y_train_val,target_lags,target_preds,method.X_scaler,configs.ex_time_dim)
        print('Save the model')
        if os.path.isdir('./pkl_folder')!=True:
            os.mkdir('./pkl_folder')
        torch.save(method.model.model.state_dict(), './pkl_folder/'+strategy_name+'_'+data_name+'_'+method.name+'.pkl')
        preds = method.model.predict(X_test_val,y_train_val,target_lags,target_preds,X_scaler,configs.ex_time_dim,device)
        preds = preds.cpu().detach().numpy()
    # prediction #
    else:
        X_scaler = method.X_scaler
        X_train_val = X_scaler.fit_transform(X_train.values)
        X_test_val = X_scaler.transform(X_test.values)
        method.set_params(len(target_preds)+1,y_train_val.shape[-1]//(len(target_preds)+1))
        method.fit(X_train_val, y_train_val)
        preds = method.predict(X_test_val)
    # preds = method.model.predict(X_test_val,target_lags)
    if len(preds) == len(method.quantiles):
        preds = preds.T
    # preds = preds.cpu().detach().numpy()
    # preds = preds*np.mean(method.y_scaler.scale_)+np.mean(method.y_scaler.mean_)
    new_target_preds = [0] + target_preds
    result = xr.DataArray(
    preds,
    dims=("time", "pred_length", "dimension","quantiles"),
    coords={
        "time": X_test.index,
        "pred_length": new_target_preds,
        "dimension": target,
        "quantiles": method.quantiles, 
    },
    )
    return result




def point_forecasting(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_test: pd.DataFrame,
                               method: MultiQuantileRegressor,
                               target: List,
                               external_features_diminsion: int,
                               target_lags: List,
                               target_preds:List,
                               data_name,
                               strategy_name,
                               device,
                               configs = None) -> pd.DataFrame:
    """Multi-quantile forecasting workflow"""
    # Scale data if required
    # if method.y_scaler:
    # #     X_train_val = method.X_scaler.fit_transform(X_train.values)
    # #     X_test_val = method.X_scaler.transform(X_test.values)
    #     y_train_val = method.y_scaler.fit_transform(y_train.values)
    # else:
    # X_train_val = X_train.values
    # X_test_val = X_test.values
    y_train_val = y_train
    # Initialize, train and test models
    print(method.name)
    if method.name.split("_")[0] in ['PDLinear','PLSTM','PMLP','PCNN','PTransformer','PLSTNet',
                                     'PInformer','PAutoformer','PFedformer','PFiLM','PiTransformer',
                                     'PNSTransformer','PPatchTST','PSegRNN','PTimeMixer','PTimesNet',
                                     'PTSMixer','PFreTS','PReformer','PNBEATS','PNBEATSX',
                                     'PTSMixerExt','PWaveNet','PBiTCN','PTiDE','PTimeXer','PMICN',
                                     'PWPMixer','PTFT']:
        X_train_val = X_train.values
        X_test_val = X_test.values
        # Set params
        method.set_params(configs)
        # X_train_val = torch.Tensor(X_train_val).to(device)
        # X_test_val = torch.Tensor(X_test_val).to(device)
        y_train_val = torch.Tensor(y_train_val).to(device)
        method.model,X_scaler = method.model.fit(X_train_val, y_train_val,target_lags,target_preds,method.X_scaler,configs.ex_time_dim)
        print('Save the model')
        if os.path.isdir('./pkl_folder')!=True:
            os.mkdir('./pkl_folder')
        torch.save(method.model.model.state_dict(), './pkl_folder/'+strategy_name+'_'+data_name+'_'+method.name+'.pkl')
        preds = method.model.predict(X_test_val,y_train_val,target_lags,target_preds,X_scaler,configs.ex_time_dim,device)
        preds = preds.cpu().detach().numpy()
    # preds = preds.cpu().detach().numpy()
    # preds = preds*np.mean(method.y_scaler.scale_)+np.mean(method.y_scaler.mean_)
    preds = np.squeeze(preds, axis=-1)
    new_target_preds = [0] + target_preds
    result = xr.DataArray(
    preds,
    dims=("time", "pred_length", "dimension"),
    coords={
        "time": X_test.index,
        "pred_length": new_target_preds,
        "dimension": target,
    },
    )
    return result
