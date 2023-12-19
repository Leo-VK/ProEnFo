import os

import numpy as np
from torch import save, Tensor
from typing import List
import torch
import torch.nn as nn 
import pandas as pd

from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Taken from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self,loss_function_name,model_name,patience=10, verbose=False, delta=1e-4, path='checkpoint.pt', trace_func=print, strategy=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.model_name = model_name
        self.loss_function_name = loss_function_name
        self.root_path  ='./pkl_folder'
        self.save_path = self.root_path+'/'+self.model_name+'_'+self.loss_function_name+'_'+self.path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease"""
        if os.path.isdir(self.root_path)!=True:
            os.mkdir(self.root_path)
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

    def clean_up_checkpoint(self) -> bool:
        os.remove(os.path.realpath(f"{self.save_path}"))
        return True


class PinballScore:
    """Pinball loss averaged over samples"""

    def __init__(self, quantile: float = 0.5):
        self.quantile = quantile
        self.name = 'PS'

    def __call__(self, y_pred: Tensor, y: Tensor):
        error = y_pred - y
        quantile_coef = (error > 0).float() - self.quantile
        return (error * quantile_coef).mean()


class PinballLoss:
    """Pinball loss averaged over quantiles and samples"""

    def __init__(self, quantiles: List[float],device):
        self.quantiles = Tensor(quantiles).to(device)
        self.name = 'PL'

    def __call__(self, y_pred: Tensor, y: Tensor):
        error = y_pred.sub(y)
        quantile_coef = (error > 0).float().sub(self.quantiles)
        return (error * quantile_coef).mean()
    
def breakpoint_generator(breakpoint_raw):
    x_y_cord_sep=[]
    for i in range(len(breakpoint_raw)):
        if 0 in breakpoint_raw[i][:,0]:
            x_cord_sort=np.sort(breakpoint_raw[i][:,0])
            ind=np.argsort(breakpoint_raw[i][:,0])
            y_cord_sort=breakpoint_raw[i][ind,1]
            sd=np.concatenate((x_cord_sort.reshape(-1,1),y_cord_sort.reshape(-1,1)),axis=1)
            x_y_cord_sep.append(sd)
        else:
            x_cord_int=np.append(breakpoint_raw[i][:,0],0)
            y_cord_int=np.append(breakpoint_raw[i][:,1],0)
            x_cord_sort=np.sort(x_cord_int)
            ind=np.argsort(x_cord_int)
            y_cord_sort=y_cord_int[ind]
            sd=np.concatenate((x_cord_sort.reshape(-1,1),y_cord_sort.reshape(-1,1)),axis=1)
            x_y_cord_sep.append(sd)
    return x_y_cord_sep     



class ContinuousPiecewiseLinearFunction(nn.Module):
    def __init__(self, break_points):
        super(ContinuousPiecewiseLinearFunction, self).__init__()
        self.break_points = break_points
        self.name = 'CPLF'
        
        # 根据输入的断点计算线性模型参数
        slopes = []
        intercepts = []
        x1, y1 = break_points[0]
        x2, y2 = break_points[1]
        slope = (y1 - y2) / (x1 - x2)
        intercept = y1 - slope * x1
        slopes.append(slope)
        intercepts.append(intercept)
        
        for i in range(len(break_points)):
            x1, y1 = break_points[i]
            if i == len(break_points) - 1:
                x2, y2 = break_points[i - 1]
                slope = (y1 - y2) / (x1 - x2)
                intercept = y1 - slope * x1
            else:
                x2, y2 = break_points[i + 1]
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
            slopes.append(slope)
            intercepts.append(intercept)
        
        self.linear_models = list(zip(slopes, intercepts))
    def forward(self, pred, true):
        x = (pred - true)
        y = torch.zeros_like(x)
        linear_masks = []

        # 计算线性部分的mask
        for i in range(len(self.linear_models) - 1):
            if i == 0:
                linear_masks.append(x < self.break_points[i, 0])
            else:
                linear_masks.append((x >= self.break_points[i - 1, 0]) & (x < self.break_points[i, 0]))
                
        linear_masks.append((x >= self.break_points[-1, 0]))
        
        # 使用mask计算y值
        for i, (m, b) in enumerate(self.linear_models):
            y = torch.where(linear_masks[i], m * x + b, y)

        return torch.mean(y)

class ContinuousPiecewiseFunction(nn.Module):
    def __init__(self, break_points, overlap, linear_models, quadratic_models):
        super(ContinuousPiecewiseFunction, self).__init__()
        self.break_points = break_points
        self.overlap = overlap
        self.linear_models = linear_models
        self.quadratic_models = quadratic_models
        self.name = 'CPF'

    def forward(self, pred,true):
        x = true-pred
        y = torch.zeros_like(x)
        linear_masks = []
        quadratic_masks = []

        # 计算线性部分的mask
        for i in range(len(self.linear_models)-1):
            if i == 0:
                linear_masks.append(x < self.break_points[i] - self.overlap)
            else:
                linear_masks.append((x >= self.break_points[i - 1] + self.overlap) & (x < self.break_points[i] - self.overlap))
        linear_masks.append((x >= self.break_points[-1] + self.overlap))

        # for i, _ in enumerate(self.linear_models):
        #     if i == 0:
        #         linear_masks.append(x < self.break_points[i] - self.overlap)
        #     else:
        #         linear_masks.append((x >= self.break_points[i - 1] + self.overlap) & (x < self.break_points[i] - self.overlap))

        #     linear_masks.append(x >= self.break_points[-1] + self.overlap)


        # 计算二次部分的mask
        for i in range(len(self.quadratic_models)):
            quadratic_masks.append((x >= self.break_points[i] - self.overlap) & (x < self.break_points[i] + self.overlap))

        # 使用mask计算y值
        for i, (m, b) in enumerate(self.linear_models):
            y = torch.where(linear_masks[i], m * x + b, y)

        for i, (a, b, c) in enumerate(self.quadratic_models):
            y = torch.where(quadratic_masks[i], a * x**2 + b * x + c, y)

        return torch.mean(y)

    
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.name = 'MSE'

    def forward(self, pred,true):
        
        
        return torch.mean(torch.pow((pred-true),2))

def fun_y(x):
    y = []
    for i in range(len(x)):
        if x[i]<=-0.05:
            q = float(72*x[i]**8+128*x[i]**6+16*x[i]**4+44*x[i]**2+np.random.normal(0,0.005,1))
            if q<=0:
                y.append(0)
            elif q>1:
                y.append(1)
            else:
                y.append(q)
        elif x[i]<=-0.02:
            q = float(72*x[i]**8+64*x[i]**6+16*x[i]**4+36*x[i]**2+np.random.normal(0,0.005,1))
            if q<=0:
                y.append(0)
            elif q>=1:
                y.append(1)
            else:
                y.append(q)
        elif x[i]<=-0.01:
            q = float(72*x[i]**8+48*x[i]**6+14*x[i]**4+36*x[i]**2+np.random.normal(0,0.005,1))
            if q<=0:
                y.append(0)
            elif q>=1:
                y.append(1)
            else:
                y.append(q)
        elif x[i]<=0:
            q = float(24*x[i]**8+24*x[i]**6+12*x[i]**4+36*x[i]**2+np.random.normal(0,0.005,1))
            if q<=0:
                y.append(0)
            elif q>=1:
                y.append(1)
            else:
                y.append(q)
        elif x[i]<=0.02:
            q = float(24*x[i]**8+24*x[i]**6+12*x[i]**4+34*x[i]**2+np.random.normal(0,0.005,1))
            if q<=0:
                y.append(0)
            elif q>=1:
                y.append(1)
            else:
                y.append(q)
        elif x[i]<=0.1:
            q = float(24*x[i]**8+20*x[i]**6+12*x[i]**4+30*x[i]**2+np.random.normal(0,0.005,1))
            if q<=0:
                y.append(0)
            elif q>=1:
                y.append(1)
            else:
                y.append(q)
        else:
            q = float(20*x[i]**8+20*x[i]**6+12*x[i]**4+16*x[i]**3+32*x[i]**2+np.random.normal(0,0.005,1))
            if q<=0:
                y.append(0)
            elif q>=1:
                y.append(1)
            else:
                y.append(q)
        
    return y

def generated(generated=False):
    if generated:
        x = np.random.uniform(-0.15, 0.15, 5000)
        y = fun_y(x)
        df = pd.DataFrame({'col1': x, 'col2': y})
        df = df.sort_values(by='col1')
        df.to_csv('./simulated_data.csv',index=False)
    else:
        df = pd.read_csv('./simulated_data.csv')
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    # 进行样条函数拟合
    spline = UnivariateSpline(x, y, k = 3,s=0.5)
    threshold = 0.15
    length = 15000
    K_x = np.random.uniform(-threshold, threshold, length)
    K_x.sort()
    K_y_smooth_2 = spline.derivative(n=2)(K_x)
    K_yy = np.power(K_y_smooth_2,2/5)
    # 使用cumtrapz函数计算累积积分
    K_y_int = cumtrapz(K_yy, K_x, axis=0, initial=0)
    K_y_int = np.power(K_y_int,5/2)
    for i in range(1,500):
        if K_y_int[-1]/(np.sqrt(120)*(i**2))<=0.005:
            K=i
            break
    point = []
    point.append(-threshold)
    all_x = np.random.uniform(-threshold, threshold, length)
    all_x.sort()
    all_y_smooth_2 = spline.derivative(n=2)(all_x)
    all_yy = np.power(np.abs(all_y_smooth_2),2/5)
    all = cumtrapz(all_yy, all_x, axis=0, initial=0)[-1]
    for i,k in enumerate(np.linspace(-threshold,threshold,length*10)):
        if len(point)==K:
            break
        else:
            start_point = point[-1]
            end_point = k
            x = np.linspace(start_point,end_point,length)
            y = spline.derivative(n=2)(x)
            yy = np.power(np.abs(y),2/5)
            iter = (cumtrapz(yy, x, axis=0, initial=0)[-1])/all
            if np.abs(iter-1/K)<=0.0001:
                point.append(end_point)
    point.pop(0)

    # 示例数据
    X = np.array(df.iloc[:,0])
    Y = np.array(df.iloc[:,1])

    # 分段点
    break_points = point

    # 指定重叠区域宽度
    overlap = 0.000001

    # 对每个部分进行线性拟合
    linear_models = []
    for i in range(len(break_points) + 1):
        if i == 0:
            mask = X < break_points[i]
        elif i == len(break_points):
            mask = X >= break_points[i - 1]
        else:
            mask = (X >= break_points[i - 1]) & (X < break_points[i])

        X_part = X[mask]
        Y_part = Y[mask]
        model = np.polyfit(X_part, Y_part, 1)
        linear_models.append(model)

    # 在间断点附近构建二次函数
    quadratic_models = []
    for i in range(1, len(linear_models)):
        m1, b1 = linear_models[i - 1]
        m2, b2 = linear_models[i]
        x0 = break_points[i - 1]

        # 二次函数的系数
        a = (m2 - m1) / (4 * overlap)
        b = m1 - 2 * a * (x0 - overlap)
        c = m1 * (x0 - overlap) + b1 - a * (x0 - overlap)**2 - b * (x0 - overlap)

        quadratic_models.append((a, b, c))
    
    return(linear_models,quadratic_models,overlap,break_points)

    

def fillna_with_rolling_mean(df, column_name):
    # 创建一个副本，以避免修改原始DataFrame
    df_copy = df.copy()
    
    # 获取缺失值的索引
    missing_indices = df_copy[column_name].isna()
    
    # 对于每个缺失值，获取上下7行的平均值
    for index in missing_indices.index[missing_indices]:
        start_upper = index - 7 if index - 7 >= 0 else 0
        start_lower = index + 1
        upper_values = df_copy[column_name].iloc[start_upper:index].dropna()
        lower_values = df_copy[column_name].iloc[start_lower:start_lower + 7].dropna()
        values = pd.concat([upper_values, lower_values])
        mean_value = values.mean()
        
        df_copy.at[index, column_name] = mean_value
    
    return df_copy

