import pandas as pd

from evaluation.metrics import ErrorMetric

from typing import List, Dict
import numpy as np
import xarray as xr


# def probabilistic_evaluation(y_true: pd.Series,
#                              forecasts: Dict[str, pd.DataFrame],
#                              metrics: List[ErrorMetric]) -> Dict[str, Dict[str, pd.Series]]:
#     """Calculate probabilistic metrics between true value and forecasts"""
#     errors = {}
#     for model in forecasts:

#         # Extract prediction
#         y_pred = forecasts[model]
#         # Calculate error metrics
#         errors[model] = {}
#         for metric in metrics:
#             name = metric.__class__.__name__
#             errors[model][f"{name}"] = metric.calculate_mean_error(y_true, y_pred)
#             errors[model][f"Instant_{name}"] = metric.calculate_instant_error(y_true, y_pred)

#     return errors

# def probabilistic_evaluation(y_true: pd.DataFrame,
#                              forecasts: Dict[str, xr.DataArray],
#                              metrics: List[ErrorMetric]) -> Dict[str, Dict[str, xr.DataArray]]:
#     """Calculate probabilistic metrics between true value and forecasts"""
#     y_true_da = xr.DataArray.from_series(y_true.stack()).rename({'level_0': 'time', 'level_1': 'pred_length'})
#     errors = {}
#     for model in forecasts:

#         # Extract prediction
#         y_pred = forecasts[model]
#         # Calculate error metrics
#         errors[model] = {}
#         for metric in metrics:
#             name = metric.__class__.__name__
#             mean_errors = []
#             instant_errors = []
            
#             # Changed: Iterate over the second dimension of y_pred
#             for t in range(y_pred.sizes[y_pred.dims[1]]):
#                 y_true_t = y_true_da.isel(pred_length=t).expand_dims('pred_length').to_series()
#                 # y_pred_t = y_pred.isel(pred_length=t).stack(index=("location", "time")).to_pandas().unstack("quantiles")
#                 y_pred_t = y_pred.isel(pred_length=t).to_pandas()
#                 y_pred_t.columns = y_pred.coords['quantiles'].values
#                 y_true_t.index = y_pred_t.index
#                 mean_error_t = metric.calculate_mean_error(y_true_t, y_pred_t)
#                 instant_error_t = metric.calculate_instant_error(y_true_t, y_pred_t)
#                 mean_errors.append(mean_error_t)
#                 instant_errors.append(instant_error_t)

#             # errors[model][f"{name}"] = xr.concat(mean_errors, dim=y_pred.dims[1])
#             errors[model][f"{name}"] = xr.concat([xr.DataArray(e) for e in mean_errors], dim=y_pred.dims[1])
#             # errors[model][f"Instant_{name}"] = xr.concat(instant_errors, dim=y_pred.dims[1])
#             errors[model][f"{name}"] = xr.concat([xr.DataArray(e) for e in instant_errors], dim=y_pred.dims[1])

#     return errors

def probabilistic_evaluation(y_true: xr.DataArray,
                             forecasts: Dict[str, xr.DataArray],
                             metrics: List[ErrorMetric]) -> Dict[str, Dict[str, xr.DataArray]]:
    """Calculate probabilistic metrics between true value and forecasts"""
    y_true_da = y_true.rename({'time': 'time', 'pred_length': 'pred_length', 'dimension': 'dimension'})
    errors = {}
    for model in forecasts:

        # Extract prediction
        y_pred = forecasts[model]
        # Calculate error metrics
        errors[model] = {}
        for metric in metrics:
            name = metric.__class__.__name__
            mean_errors = []
            instant_errors = []
            
            # Changed: Iterate over the second dimension of y_pred
            for t in range(y_pred.sizes[y_pred.dims[1]]):
                for new_dim in range(y_pred.sizes[y_pred.dims[2]]):  # iterate over new dimension
                    y_true_t = y_true_da.isel(pred_length=t, dimension=new_dim).expand_dims(['pred_length', 'dimension']).to_series()
                    y_pred_t = y_pred.isel(pred_length=t, dimension=new_dim).to_pandas()
                    y_pred_t.columns = y_pred.coords['quantiles'].values
                    y_true_t.index = y_pred_t.index
                    mean_error_t = metric.calculate_mean_error(y_true_t, y_pred_t)
                    instant_error_t = metric.calculate_instant_error(y_true_t, y_pred_t)
                    mean_errors.append(mean_error_t)
                    instant_errors.append(instant_error_t)
            mean_errors = np.array([e for e in mean_errors])
            instant_errors = np.array([e for e in instant_errors])
        
            if len(mean_errors.shape) ==2:
                mean_errors = xr.DataArray(mean_errors.reshape(y_pred.sizes[y_pred.dims[1]],y_pred.sizes[y_pred.dims[2]],-1),
                                            dims=("pred_length", "dimension","quantiles"),
                                            coords={
                                            "pred_length": y_pred.coords['pred_length'].values,
                                            "dimension": y_pred.coords['dimension'].values,
                                        })
                
                instant_errors = xr.DataArray(instant_errors.reshape(y_pred.sizes[y_pred.dims[1]],y_pred.sizes[y_pred.dims[2]],-1,instant_error_t.shape[-1]),
                                            dims=("pred_length","dimension", "time","quantiles"),
                                                coords={
                                                    "time":y_pred.coords['time'].values,
                                                "pred_length": y_pred.coords['pred_length'].values,
                                                "dimension": y_pred.coords['dimension'].values,
                                            })
            else:
                mean_errors = xr.DataArray(mean_errors.reshape(y_pred.sizes[y_pred.dims[1]],y_pred.sizes[y_pred.dims[2]]),
                                            dims=("pred_length", "dimension"),
                                            coords={
                                            "pred_length": y_pred.coords['pred_length'].values,
                                            "dimension": y_pred.coords['dimension'].values,
                                        })
                
                instant_errors = xr.DataArray(instant_errors.reshape(y_pred.sizes[y_pred.dims[1]],y_pred.sizes[y_pred.dims[2]],-1),
                                            dims=("pred_length", "dimension","time"),
                                                coords={
                                                    "time":y_pred.coords['time'].values,
                                                "pred_length": y_pred.coords['pred_length'].values,
                                                "dimension": y_pred.coords['dimension'].values,
                                            })

            errors[model][f"{name}"] = mean_errors
            errors[model][f"Instant_{name}"] = instant_errors

    return errors






def point_evaluation(y_true: pd.Series,
                             forecasts: Dict[str, pd.DataFrame],
                             metrics: List[ErrorMetric]) -> Dict[str, Dict[str, pd.Series]]:
    """Calculate point metrics between true value and forecasts"""
    y_true_da = y_true.rename({'time': 'time', 'pred_length': 'pred_length', 'dimension': 'dimension'})
    errors = {}
    for model in forecasts:

        # Extract prediction
        y_pred = forecasts[model]
        # Calculate error metrics
        errors[model] = {}
        for metric in metrics:
            name = metric.__class__.__name__
            mean_errors = []
            instant_errors = []
            
            # Changed: Iterate over the second dimension of y_pred
            for t in range(y_pred.sizes[y_pred.dims[1]]):
                for new_dim in range(y_pred.sizes[y_pred.dims[2]]):  # iterate over new dimension
                    y_true_t = y_true_da.isel(pred_length=t, dimension=new_dim).expand_dims(['pred_length', 'dimension']).to_series()
                    y_pred_t = y_pred.isel(pred_length=t, dimension=new_dim).to_pandas()
                    y_true_t.index = y_pred_t.index
                    mean_error_t = metric.calculate_mean_error(y_true_t, y_pred_t)
                    instant_error_t = metric.calculate_instant_error(y_true_t, y_pred_t)
                    mean_errors.append(mean_error_t)
                    instant_errors.append(instant_error_t)
            mean_errors = np.array([e for e in mean_errors])
            instant_errors = np.array([e for e in instant_errors])
            mean_errors = xr.DataArray(mean_errors.reshape(y_pred.sizes[y_pred.dims[1]],y_pred.sizes[y_pred.dims[2]]),
                                        dims=("pred_length", "dimension"),
                                        coords={
                                        "pred_length": y_pred.coords['pred_length'].values,
                                        "dimension": y_pred.coords['dimension'].values,
                                    })
            
            instant_errors = xr.DataArray(instant_errors.reshape(y_pred.sizes[y_pred.dims[1]],y_pred.sizes[y_pred.dims[2]],-1),
                                        dims=("pred_length","dimension", "time"),
                                            coords={
                                            "pred_length": y_pred.coords['pred_length'].values,
                                            "dimension": y_pred.coords['dimension'].values,
                                            "time":y_pred.coords['time'].values,
                                        })

            errors[model][f"{name}"] = mean_errors
            errors[model][f"Instant_{name}"] = instant_errors
            
        

    return errors
