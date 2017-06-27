# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:56:56 2017

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


dictAtria='C:/Users/lenovo/Desktop/pythonProj/ecg/EGM_AV_block.mat'
atria=sio.loadmat(dictAtria) 
ecgatria=atria['EGM']
atriaecg=ecgatria['potvals']
epd=atriaecg[0][0]  #62*15191
epdn=MaxMinNormalization(epd,np.max(epd),np.min(epd))
epdnT=epdn.T


modelatria=Sequential()
encoding_Dim=1998
#多层编码
modelatria.add(Dense(128,activation='relu',input_dim=62))
modelatria.add(Dense(256,activation='relu'))
modelatria.add(Dense(encoding_Dim))
#多层解码
"""
dense层的正则项可以加在权重上（kernel_regularizer）,
偏置上（bias_regularizer）
输出上（activity_regularizer）
"""
#对输出施加L1正则项约束
modelatria.add(Dense(256,activation='relu',input_dim=encoding_Dim,activity_regularizer=regularizers.l1(0.01)))
modelatria.add(Dense(128,activation='relu'))
modelatria.add(Dense(62,activation='tanh'))


modelatria.compile(optimizer='adam',loss='mse')

#model.compile(optimizer='adadelta',loss='binary_crossentropy')
#所谓自编码器，样本和标签都是x_train
modelatria.fit(epdnT,epdnT,epochs=7,batch_size=256)

#获取训练后各层的权重
LayerWa=modelatria.get_weights()

#将15191*54的体表电位映射成15191*2002
EPD1=np.dot(epdnT,LayerWa[0])
EPD2=np.dot(EPD1,LayerWa[4])
EPD=np.dot(EPD2,LayerWa[6])
#显示映射后的某个电极处的心电信号和原来某个电极处的心电信号
signalA0=EPD[:,1]
signalA1=epd[1,:]
signalA2=epdn[1,:]
#产生时间序列
timeA=np.zeros(15191)
for i in range(0,15191):
    timeA[i]=i
#作图
"""
从左到又：
映射后的信号，原始信号，归一化后的信号

"""
"""

plt.subplot(1,3,1)
plt.scatter(timeA,epdnT[:,1])
plt.subplot(1,3,2)
plt.scatter(timeA,-epdnT[:,3])
plt.subplot(1,3,3)
plt.scatter(timeA,epdnT[:,7])

plt.subplot(1,3,1)
plt.scatter(timeA,epdnT[:,8])
plt.subplot(1,3,2)
plt.scatter(timeA,-epdnT[:,100])
plt.subplot(1,3,3)

plt.scatter(timeA,epdnT[:,50])


"""
#保存产生的EPD，以及各层权重
dataNew = 'C:/Users/lenovo/Desktop/pythonProj/ecg/epcardialPD.mat'

sio.savemat(dataNew,{'array':EPD})
