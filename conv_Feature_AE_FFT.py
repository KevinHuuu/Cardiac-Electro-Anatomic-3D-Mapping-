# -*- coding: utf-8 -*-
"""
Created on Fri May  5 21:26:34 2017

@author: lenovo

基本思路：
（一）傅里叶变换+自编码器
1.对体表和心脏信号做傅里叶变换
2.用变换后的信号重复forward_inverse.py中的思路
（二）傅里叶变换+卷积神经网络+自编码器

目前可行的两种解决方案：
1.想办法训练一个万能的映射器，虽然可能是用某几个患者的数据训练的，但对于所有的患者都能用
即在训练时不能加入特定患者的信息（如躯干与心脏的几何信息）
2.像有限元，边界元那样，根据特定患者的信息一步步计算，实现定制。
"""

import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
import random

from keras.models import Sequential
from keras.layers import Dense ,Input
from keras import regularizers

dictAtria='C:/Users/lenovo/Desktop/pythonProj/ecg/EGM_AV_block.mat'
atria=sio.loadmat(dictAtria) 
ecgatria=atria['EGM']
atriaecg=ecgatria['potvals']
epd=atriaecg[0][0]  #62*15191
#epdn=sigmoid(epd,1)
#epdn=MaxMinNormalization(epd,np.max(epd),np.min(epd))
#epdnT=epdn.T


#加载原始体表电位数据
dictTorso='C:/Users/lenovo/Desktop/pythonProj/ecg/ECG_AV_block.mat'
torso=sio.loadmat(dictTorso) 
ecgbody=torso['ECG']
bodyecg=ecgbody['potvals']
#体表电位分布和心房电位分布
bspd=bodyecg[0][0] #54*15191
#数据归一化
#每个点的所有时刻是一个样本
#bspdn=MaxMinNormalization(bspd,np.max(bspd),np.min(bspd))
#转置，梅个时刻的所有点是一个样本
#bspdnT=bspdn.T

#对信号进行傅里叶变换
a=np.fft.fft(epd.T[:,1])

b=np.fft.fftshift(a)

fftepd=[]
fftepdshifted=[]

for i in range(62):
    fftepd.append(np.fft.fft(epd.T[:,i]))
for i in range(62):
    fftepdshifted.append(np.fft.fftshift(fftepd[i]))

