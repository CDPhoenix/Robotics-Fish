# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:00:13 2024

@author: Dapeng WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY
"""
import pandas as pd
import numpy as np
import random

class KalmanFilter():
    def __init__(self,A,B,H,P):
        
        """
        A:系统的状态矩阵
        B：输入的状态矩阵
        H：观测者的变化矩阵
        P：预测的噪声协方差矩阵
        S：观测的噪声协方差矩阵
        w: 预测的噪声
        v: 观测的噪声
        """
        
        self.A = A
        self.B = B
        self.H = H
        self.P = P
        self.S = 0
        self.I = np.array([[1,0],[0,1]])
        self.w = 0
        self.v = 0
        
    def Filtering(self,data,i):
        """
        构建数据集，分别为上一时刻的系统预测值，当下的系统根据实际时间的预测值以及观测值
        """
        if i == 0:
            lastdis = data['Predictdistance'].values[0]
            lastV = data['InputVelocity'].values[0]
        else:
            lastdis = data['KalmanFilterdistance'].values[0]
            lastV = data['KalmanFiltervelocity'].values[0]
        
        nowdis = data['Predictdistance'].values[1]
        nowV = data['InputVelocity'].values[1]
        
        obdis = data['Obeservedistance'].values[1]
        obV = data['Velocity_Z'].values[1]
        
        x0 = np.array([[lastdis],[lastV]])#上一时刻的系统预测
        x1 = np.array([[nowdis],[nowV]])#当下的系统状态根据实际响应时间的无噪音预测
        xobs = np.array([[obdis],[obV]])#当下系统的实际观测值
        
        x1_est = np.dot(self.A,x0)#预测当前系统
        self.w = xobs - x1_est #预测残差
        self.UpdatingCov(0)#更新预测协方差矩阵
        
        self.v = xobs - np.dot(self.H,x1)#观测残差
        self.UpdatingCov(1)#更新观测协方差矩阵
        
        self.K = np.dot(self.P,self.H.T)#计算卡尔曼增益
        self.K = np.dot(self.K,np.linalg.inv(self.S))
        self.UpdatingCov(2)#更新最终的预测协方差矩阵
        
        return x1_est + np.dot(self.K,self.v)#返回修正后的预测值
        
        
    def UpdatingCov(self,mode = 0):
        #Q = np.cov(np.dot(w,w.T))
        if mode == 0:
            Q = np.cov(np.dot(self.w,self.w.T))
            P_trans = np.dot(self.A,self.P)
            P_trans = np.dot(P_trans,self.A)
            self.P = P_trans + Q
        elif mode == 1:
            R = np.cov(np.dot(self.v,self.v.T))
            S_trans = np.dot(self.H,self.P)
            S_trans = np.dot(S_trans,self.H.T)
            self.S = S_trans + R
        elif mode == 2:
            P_trans = self.I - np.dot(self.K,self.H)
            self.P = np.dot(P_trans,self.P)
        else:
            raise ValueError("Invalid option! Please reinput 0 for updating prediction cov or 1 for observation cov or 2 for final predict cov")
        
        