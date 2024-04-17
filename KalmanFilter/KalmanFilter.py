# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:57:42 2024

@author: Dapeng WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY
"""

import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from KalmanFilterModel import KalmanFilter
data = pd.read_csv('./KalmanFilter.csv',sep=',',header=None)

#data = data.drop([0,1,2,3,4])
data = data.drop([0,1])
data.index = list(range(len(data)))

data = data.astype(float)
data.columns = ['Time','Velocity_X']
data['Velocity_Z'] = data['Velocity_X'].values/3600*60

data['InputVelocity'] = np.ones(len(data))*440
data['dtime'] = np.zeros(len(data))
data['Predictdistance'] = np.zeros(len(data))
data['Obeservedistance'] = np.zeros(len(data))
data['KalmanFilterdistance'] = np.zeros(len(data))
data['KalmanFiltervelocity'] = np.zeros(len(data))

DistanceTemp = 0
PreDistanceTemp = 0

for i in range(1,len(data)):
    data['dtime'].values[i] = data['Time'].values[i] - data['Time'].values[i-1]
    DistanceTemp += (data['Velocity_Z'].values[i]+data['Velocity_Z'].values[i-1])*data['dtime'].values[i]/2
    PreDistanceTemp += data['InputVelocity'].values[i]*data['dtime'].values[i]
    data['Predictdistance'].values[i] = PreDistanceTemp
    data['Obeservedistance'].values[i] = DistanceTemp

#data = data.drop([0,1,2])
#data.index = list(range(len(data)))

A = np.array([[1,0.05],[0,1]])
B = np.array([[0,0]])
H = np.array([[1,0],[0,1]])
P = np.array([[0,0],[0,0]])
KalmanFilter = KalmanFilter(A, B, H, P)

for i in range(0,len(data)-1):
    data_slice = data[i:i+2]
    ans = KalmanFilter.Filtering(data_slice,i)
    data['KalmanFilterdistance'].values[i+1] = ans[0]
    data['KalmanFiltervelocity'].values[i+1] = ans[1]

fig,ax = plt.subplots()
ax.plot(data['Time'].values[5:],data['Predictdistance'].values[5:],label = 'Prediction')
ax.plot(data['Time'].values[5:],data['Obeservedistance'].values[5:],label = 'Observed')
ax.plot(data['Time'].values[5:],data['KalmanFilterdistance'].values[5:],label = 'KalmanFilter')
ax.legend()
plt.show()

fig1,ax1 = plt.subplots()
ax1.plot(data['Time'].values[5:],data['InputVelocity'].values[5:],label = 'Prediction')
ax1.plot(data['Time'].values[5:],data['Velocity_Z'].values[5:],label = 'Observed')
ax1.plot(data['Time'].values[5:],data['KalmanFiltervelocity'].values[5:],label = 'KalmanFilter')
ax1.legend()
plt.show()
