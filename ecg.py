# -*- coding: utf-8 -*-
"""
Created on Thu May  4 07:52:16 2017

@author: lenovo
"""

import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  

from keras.models import Sequential
from keras.layers import Dense ,Input
from keras import regularizers


def MaxMinNormalization(x,Max,Min):  
    x = (x - Min) / (Max - Min)  
    return x


#加载数据
dictTorso='C:/Users/lenovo/Desktop/pythonProj/ecg/ECG_AV_block.mat'

torso=sio.loadmat(dictTorso) 

ecgbody=torso['ECG']

bodyecg=ecgbody['potvals']

#体表电位分布和心房电位分布
bspd=bodyecg[0][0] #54*15191


#数据归一化
#每个点的所有时刻是一个样本
bspdn=MaxMinNormalization(bspd,np.max(bspd),np.min(bspd))

#转置，梅个时刻的所有点是一个样本
bspdnT=bspdn.T


#自编码器
modelbody=Sequential()
encoding_Dim=2002
#多层编码
modelbody.add(Dense(128,activation='relu',input_dim=54))
modelbody.add(Dense(256,activation='relu'))
modelbody.add(Dense(encoding_Dim))
#多层解码
"""
dense层的正则项可以加在权重上（kernel_regularizer）,
偏置上（bias_regularizer）
输出上（activity_regularizer）
"""
#对输出施加L1正则项约束
modelbody.add(Dense(256,activation='relu',input_dim=encoding_Dim,activity_regularizer=regularizers.l1(0.01)))
modelbody.add(Dense(128,activation='relu'))
modelbody.add(Dense(54,activation='tanh'))


modelbody.compile(optimizer='adam',loss='mse')

#model.compile(optimizer='adadelta',loss='binary_crossentropy')
#所谓自编码器，样本和标签都是x_train
modelbody.fit(bspdnT,bspdnT,epochs=10,batch_size=256)

#获取训练后各层的权重
LayerW=modelbody.get_weights()

#将15191*54的体表电位映射成15191*2002
BSPD1=np.dot(bspdnT,LayerW[0])
BSPD2=np.dot(BSPD1,LayerW[2])
BSPD=np.dot(BSPD2,LayerW[4])
#显示映射后的某个电极处的心电信号和原来某个电极处的心电信号
signal0=BSPD[:,1]
signal1=bspd[1,:]
signal2=bspdn[1,:]
#产生时间序列
time=np.zeros(15191)
for i in range(0,15191):
    time[i]=i
#作图
"""
从左到又：
映射后的信号，原始信号，归一化后的信号

"""
plt.figure(1)
plt.subplot(1,3,1)
plt.scatter(time,-signal0)
plt.subplot(1,3,2)
plt.scatter(time,signal1)
plt.subplot(1,3,3)
plt.scatter(time,signal2)
plt.show()

dataNew = 'C:/Users/lenovo/Desktop/pythonProj/ecg/1.mat'

sio.savemat(dataNew,{'array':BSPD})



