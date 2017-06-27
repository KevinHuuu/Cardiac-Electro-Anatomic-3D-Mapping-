# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:10:34 2017

@author: lijianning

"""

import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
import random

from keras.models import Sequential
from keras.layers import Dense ,Input
from keras import regularizers

random.seed(3)
#归一化函数
def MaxMinNormalization(x,Max,Min):  
    x = (x - Min) / (Max - Min)  
    return x


#sigmoid 归一化
def sigmoid(X,useStatus):  
    if useStatus:  
        return 1.0 / (1 + np.exp(-float(X)))  
    else:  
        return float(X)
  #加载原始心房电位  
        
dictAtria='C:/Users/lenovo/Desktop/pythonProj/ecg/EGM_AV_block.mat'
atria=sio.loadmat(dictAtria) 
ecgatria=atria['EGM']
atriaecg=ecgatria['potvals']
epd=atriaecg[0][0]  #62*15191
#epdn=sigmoid(epd,1)
epdn=MaxMinNormalization(epd,np.max(epd),np.min(epd))
epdnT=epdn.T

#加载原始体表电位数据
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

#分配训练集和测试集
"""
取提标电位和心房电位的10000时刻左训练，剩下的做测试
"""
#取训练集和测试机的下标


bsTestIndex=[]   
bsTrainIndex=random.sample(range(0,15191),10000)

for i in range(0,15191):
    if i not in bsTrainIndex:
        bsTestIndex.append(i)

atrTrainIndex=bsTrainIndex
atrTestIndex=bsTestIndex

bsTrain=bspdnT[bsTrainIndex,:]  #体表训练，10000*54
bsTest=bspdnT[bsTestIndex,:]    #体表测试，5191*54

atrTrain=epdnT[atrTrainIndex,:]
atrTest=epdnT[atrTestIndex,:]

#将10000*54的体表电位通过自编码器映射成10000*62的

modelbs=Sequential()
encoding_Dim=62
#多层编码
modelbs.add(Dense(128,activation='relu',input_dim=54))
modelbs.add(Dense(encoding_Dim))
#多层解码
"""
dense层的正则项可以加在权重上（kernel_regularizer）,
偏置上（bias_regularizer）
输出上（activity_regularizer）
"""
#对输出施加L1正则项约束
modelbs.add(Dense(128,activation='relu',input_dim=encoding_Dim,activity_regularizer=regularizers.l1(0.01)))
modelbs.add(Dense(54,activation='tanh'))
modelbs.compile(optimizer='adam',loss='mse')

#model.compile(optimizer='adadelta',loss='binary_crossentropy')
modelbs.fit(bspdnT,bspdnT,epochs=50,batch_size=50)

LayerWbs=modelbs.get_weights()
#体表电位映射
bs62Train1=np.dot(bsTrain,LayerWbs[0])  #体表训练，10000*62
bs62TrainFinal=np.dot(bs62Train1,LayerWbs[4])

bs62Test1=np.dot(bsTest,LayerWbs[0])    #体表测试，5191*62
bs62TestFinal=np.dot(bs62Test1,LayerWbs[4])
"""
Codes to be added here
 extract weights for modelbs
 
 
 

"""

#自编码器，建立从体表电位到心房电位的映射

modelbs2atr=Sequential()
#多层编码
modelbs2atr.add(Dense(128,activation='relu',input_dim=62))
modelbs2atr.add(Dense(256,activation='relu'))
#多层解码
#对输出施加L1正则项约束
modelbs2atr.add(Dense(128,activation='relu',input_dim=256,activity_regularizer=regularizers.l1(0.01)))
modelbs2atr.add(Dense(62,activation='tanh'))
modelbs2atr.compile(optimizer='adam',loss='mse')


#训练
modelbs2atr.fit(bs62TrainFinal,atrTrain,epochs=50,batch_size=156)

"""
Codes to be added here

analyse the weight for modelbs2atr
"""
LayerWbs2atr=modelbs2atr.get_weights()



#predict 
acc=modelbs2atr.evaluate(bs62TestFinal,atrTest,batch_size=32,verbose=1)

predictedAtr=modelbs2atr.predict(bs62TestFinal,batch_size=32,verbose=1)

#visualize the predict atr and the actual atr
time=np.zeros(5191)
for i in range(0,5191):
    time[i]=i



fig=plt.figure(1)
plt.subplot(111)
plt.subplot(221)
plt.scatter(time,predictedAtr[:,1])
plt.subplot(222)
plt.scatter(time,predictedAtr[:,35])
plt.subplot(223)
plt.scatter(time,atrTest[:,1])
plt.subplot(224)
plt.scatter(time,atrTest[:,35])

plt.ylim(np.min(predictedAtr[:,1]),np.max(predictedAtr[:,1])) 
plt.plot(time,predictedAtr[:,1])

#将预测得到的5191*62的心脏电位分布和原始心脏电位分布映射5191*1998


modelatr=Sequential()
encoding_Dim=1998
modelatr.add(Dense(128,activation='relu',input_dim=62))
modelatr.add(Dense(256,activation='relu'))
modelatr.add(Dense(encoding_Dim))

modelatr.add(Dense(256,activation='relu',input_dim=encoding_Dim,activity_regularizer=regularizers.l1(0.01)))
modelatr.add(Dense(128,activation='relu'))
modelatr.add(Dense(62,activation='tanh'))
modelatr.compile(optimizer='adam',loss='mse')

modelatr.fit(predictedAtr,predictedAtr,epochs=10,batch_size=256)
layerWatr=modelatr.get_weights()

predictAtr1=np.dot(predictedAtr,layerWatr[0])
predictAtr2=np.dot(predictedAtr,layerWatr[3])
predictAtrfinal=np.dot(predictedAtr,layerWatr[5])



    
    
    
    
