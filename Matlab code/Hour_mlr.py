# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:41:06 2020

@author: reagan
"""

import torch
import numpy as np
from scipy.io import loadmat
import pandas as pd
from torch.utils.data import Dataset, DataLoader,TensorDataset
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas.tseries.offsets import DateOffset
# %%%%%%%%%%%%%%%%%%%%%% Old data version_20200531%%%%%%%%%%%%%%%%%%%%%%%%%%
# x_cord=loadmat('E:/ETH learning material/Master Thesis/Code/Allplots/x_cor.mat')['x_cor']
# y_cord = loadmat('E:/ETH learning material/Master Thesis/Code/Allplots/y_cor.mat')['y_cor']
# pos_coe=np.array(loadmat('E:/ETH learning material/Master Thesis/Code/Allplots/pos_coe.mat')['pos_coe']).reshape(-1,1)
# neg_coe = np.array(loadmat('E:/ETH learning material/Master Thesis/Code/Allplots/neg_coe.mat')['neg_coe']).reshape(-1,1)

# %%%%%%%%%%%%%%%%%%%%%% New data version_20200802%%%%%%%%%%%%%%%%%%%%%%%%%%
torch.manual_seed(1)
breakpoint_new=loadmat('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200802/breakpoint_new.mat')
breakpoint_new.pop("__header__")
breakpoint_new.pop("__version__")
breakpoint_new.pop("__globals__")
breakpoint_raw=list(breakpoint_new.values())
store_slope=loadmat('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200802/store_slope.mat')['store_slope']
neg_coe=store_slope[:,0].reshape(-1,1)
pos_coe=store_slope[:,1].reshape(-1,1)

# %%%%%%%%%%%%%%%%%%%%%% Training dataset%%%%%%%%%%%%%%%%%%%%%%%%%%
Train_x = np.array(pd.read_csv('E:/ETH learning material/Master Thesis/Code/train_test_d_3_h_4/X_train_zone_1_station_2_REC_norm_minmax.csv', sep=',',header=None))
Train_y = np.array(pd.read_csv('E:/ETH learning material/Master Thesis/Code/train_test_d_3_h_4/y_train_zone_1_station_2_REC_norm_minmax.csv', sep=',',header=None))
Test_xin = np.array(pd.read_csv('E:/ETH learning material/Master Thesis/Code/train_test_d_3_h_4/X_test_zone_1_station_2_REC_norm_minmax.csv', sep=',',header=None))
Test_yin = np.array(pd.read_csv('E:/ETH learning material/Master Thesis/Code/train_test_d_3_h_4/y_test_zone_1_station_2_REC_norm_minmax.csv', sep=',',header=None))
# %% Functions
#% breakpoint_generator can insert 0 and re-sort the sequence of 24 hours' breakpoints%%%
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
#% calculate slope after obtaining breakpoints%%%
def slope_gen(breakpoint):
    num_line=breakpoint.shape[0]-1
    slope=np.zeros(num_line)
    for i in range(0,num_line):
        slope[i]=(breakpoint[i+1,1]-breakpoint[i,1])/(breakpoint[i+1,0]-breakpoint[i,0])
    return slope,num_line

def custom_loss(eps,t):
    eps1=eps.data.numpy()
    mask_1=np.searchsorted(breakpoint[t][:,0],eps1)
    pw=torch.zeros(len(mask_1))
    slope,num_line=slope_gen(breakpoint[t])
    for k in range(len(mask_1)):
        mask=mask_1.item(k)
        indices=torch.tensor([k])
        if mask<num_line+1:
            if mask==0:#negative extropolation
                pw[k]= (slope[mask]/breakpoint[t][0,0]-breakpoint[t][0,1]/(breakpoint[t][0,0]**2))*torch.index_select(eps,0,indices)**2+torch.index_select(eps,0,indices)*(2*breakpoint[t][0,1]/breakpoint[t][0,0]-slope[mask]) #slope[mask]*(eps.item(k)-breakpoint[0,0])+breakpoint[0,1] #extrapolate line
            else:
                pw[k]=slope[mask-1]*(torch.index_select(eps,0,indices)-breakpoint[t][mask-1,0])+breakpoint[t][mask-1,1]
        else:#positive extropolation
            pw[k]=(slope[mask-2]/breakpoint[t][mask-1,0]-breakpoint[t][mask-1,1]/(breakpoint[t][mask-1,0]**2))*torch.index_select(eps,0,indices)**2+torch.index_select(eps,0,indices)*(2*breakpoint[t][mask-1,1]/breakpoint[t][mask-1,0]-slope[mask-2]) #slope[mask]*(eps.item(k)-breakpoint[mask-1,0])+breakpoint[mask-1,1]#extrapolate line
    return pw


def custom_loss_hour(MPE):
    pw_over=torch.Tensor()
    for t in range(24):
        pw_over=torch.cat((pw_over,custom_loss(MPE[t::24],t)))
    return pw_over

def pinball_loss(eps,t):
    pw = torch.max(neg_coe[t].item()*eps,pos_coe[t].item()*eps)
    return pw

def pinball(MPE):    
    pw_over=torch.Tensor()
    for t in range(24):
        pw_over=torch.cat((pw_over,pinball_loss(MPE[t::24],t)))
    return pw_over
#%% neural network 
breakpoint = breakpoint_generator(breakpoint_raw)
delta_margin=1e-4

N, D_in, H1, H2,D_out = 64, 1019, 1, 2048,1

# Create random Tensors to hold inputs and outputs

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
        # torch.nn.ReLU(),
        # torch.nn.Linear(H1, H2),
        # torch.nn.ReLU(),
        # torch.nn.Linear(H2, H1),
        # torch.nn.ReLU(),
    # torch.nn.Dropout(p=0.5),
    torch.nn.Linear(H1, D_out),
)
#loss_fn = torch.nn.MSELoss(reduction='sum')
store_loss=[]
train_loss=[]
MAPE_err=[]
FEPC_average_store=[]
OFR_store=[]
UFR_store=[]
MAPE_store=[]
# %% Training process
epochs = 150
learning_rate = 0.1#1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99),weight_decay=1e-8)
# learning_rate_sgd = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_sgd, momentum=0, dampening=0, weight_decay=0, nesterov=False)
x_tra=np.concatenate((Train_x,Test_xin[:24*(31),:]),axis=0)#[0:24*(31+28+31+30+31+30+31+31+30+31+30+31),:]
y_tra=np.concatenate((Train_y,Test_yin[:24*(31),:]),axis=0)
x = torch.from_numpy(x_tra).float()
y = torch.from_numpy(y_tra).float()
# x = torch.from_numpy(Train_x).float()
# y = torch.from_numpy(Train_y).float()
Test_x=Test_xin#[24*(31+28+31+30+31+30+31+31+30+31+30):24*(31+28+31+30+31+30+31+31+30+31+30+31),:]#)
Test_y=Test_yin#[24*(31+28+31+30+31+30+31+31+30+31+30):24*(31+28+31+30+31+30+31+31+30+31+30+31),:]#[0:24*(31+28+31+30+31+30+31+31+30+31+30+31),:]
x_test=torch.from_numpy(Test_x).float()
data_set = TensorDataset(x,y)
data_loader = DataLoader(dataset = data_set, batch_size = N, shuffle = True)
decayRate = 0.5
decay_schedule= [50,100]
my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=decay_schedule, gamma=decayRate)

for j in range(epochs):
    for i,data_batch in enumerate(data_loader):      
        x, y = data_batch
        y_pred = model(x)
        err=y_pred-y
        loss = custom_loss_hour(err).sum()
        # loss = pinball(err).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    modeleval=model.eval()
    Y_predict=modeleval(x_test).data.numpy()
    error_dist=np.array(Y_predict-Test_y)
    losssum=np.zeros((365,24))
    for t in range(24):
        losssum[:,t]=custom_loss(torch.Tensor(error_dist[t::24]),t)    
    FEPC_for_hour=np.sum(losssum,axis=0)/365*100
    FEPC_average_id= np.sum(FEPC_for_hour)/len(FEPC_for_hour)
    FEPC_average_store = np.append(FEPC_average_store,FEPC_average_id)
    OFR_for_hour=np.zeros(24)
    for t in range(24):
        OFR_for_hour[t]=np.sum(np.array(error_dist[t::24])>0)/365*100        
    OFR=np.average(OFR_for_hour)    
    OFR_store=np.append(OFR_store,OFR)
    UFR_for_hour=np.zeros(24)
    for t in range(24):
        UFR_for_hour[t]=np.sum(np.array(error_dist[t::24])<0)/365*100
    UFR=np.average(UFR_for_hour) 
    UFR_store=np.append(UFR_store,UFR)
    MAPE=np.abs(np.divide(Y_predict-Test_y,Test_y))
    MAPE_for_hour=np.zeros(24)
    for t in range(24):
        MAPE_for_hour[t]=np.sum(MAPE[t::24])/365*100
    MAPE_AVE= np.average(MAPE_for_hour)
    MAPE_store=np.append(MAPE_store,MAPE_AVE)
    pw_over=loss.item()
    store_loss=np.append(store_loss,pw_over)
    print('This is Epoch ', j+1, 'FEPC ',FEPC_average_id,'MAPE ', MAPE_AVE,'OFR ', OFR)#torch.sum(err1-err2).data
    my_lr_scheduler.step()
    
    
# err_store=Y_predict-Test_y
Y_predict_store=Y_predict
Y_test_store=Test_y

loss_plot=np.array(store_loss).reshape(-1,1)
f3=plt.figure(2)   
plt.plot(range(store_loss.shape[0]),store_loss)
f3.suptitle('Train loss', fontsize=20)
f4=plt.figure(3)   
plt.plot(range(FEPC_average_store.shape[0]),FEPC_average_store)
f4.suptitle('FEPC loss', fontsize=20)
f5=plt.figure(4)   
plt.plot(range(OFR_store.shape[0]),OFR_store)
f5.suptitle('OFR', fontsize=20)
f6=plt.figure(1)   
plt.plot(range(MAPE_store.shape[0]),MAPE_store)
f6.suptitle('MAPE', fontsize=20)
# %% Plot
# # err_store=np.concatenate((err_store,Y_predict-Test_y),axis=0)
# Y_predict_store=np.concatenate((Y_predict_store,Y_predict))
# Y_test_store=np.concatenate((Y_test_store,Test_y))
# plt.plot(err_store.reshape(-1,1))
# losssum=np.array([])
# for t in range(24):
#     losssum=np.hstack([losssum,piecewise_function_loss(error_dist[t::24],t)])
# losssum=np.sum(losssum)
# date_rng = pd.date_range(start='1/1/2007', end='01/01/2008', freq='H')
# date_rng  = date_rng[1:]
# df = pd.DataFrame(date_rng, columns=['date'])
# df['True load'] = Test_y
# df['Predict load']=Y_predict
# df['Predict time'] = pd.to_datetime(df['date'])
# df = df.set_index('Predict time')
# df.drop(['date'], axis=1, inplace=True)
# df.plot(grid=True)

# sns.distplot(error_dist, hist=True, kde=True, 
#               bins=int(0.70/0.01), color = 'darkblue', 
#               hist_kws={'edgecolor':'black'},
#               kde_kws={'linewidth': 4})
#  # plt.xlim(0,8000)
#  # plt.title('Density Plot and Histogram')
# plt.xlabel('EP')
# plt.ylabel('Density')
#  # plt.grid(True)