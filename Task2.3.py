# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:06:17 2020

@author: Nobody
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = list(np.random.randn(10000))
#生成标准正态分布的随机数（10000个）
plt.hist(data,1000,facecolor='g',alpha=0.5)
'''
plt.hist(arr, bins=10, facecolor, edgecolor,alpha，histtype='bar')
bins：直方图的柱数，可选项，默认为10
alpha: 透明度
'''
plt.show()
s = pd.Series(data) #将数组转化为序列
print('偏度系数',s.skew())
print('峰度系数',s.kurt()) #超值峰度
