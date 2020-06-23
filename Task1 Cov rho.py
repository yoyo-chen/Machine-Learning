# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:51:58 2020

@author: Nobody

实现协方差和相关系数

"""
import math
import numpy as np

## 未中心化
x = np.array([1, 2, 3, 5, 8])
y = np.array([0.11, 0.12, 0.13, 0.15, 0.18])
## 中心化
#x = np.array([-2.8, -1.8, -0.8, 1.2, 4.2])
#y = np.array([-0.028, -0.018, -0.008, 0.012, 0.042])

n = np.size(x)
Ex = np.sum(x)/n
Ey = np.sum(y)/n
x1 = (x-Ex)**2
y1 = (y-Ey)**2
Varx = np.sum(x1)/n
Vary = np.sum(y1)/n
xy = (x-Ex)*(y-Ey)
Cov = np.sum(xy)/n
Rho = Cov/math.sqrt(Varx)/math.sqrt(Vary)

Ex2 = np.average(x)
Ey2 = np.average(y)
Varx2 = np.var(x)
Vary2 = np.var(y)
Cov2 = np.cov(x,y)[0,1]*(n-1)/n # numpy中的cov()是无偏协方差，除的是n-1
Cov3 = np.cov(x, ddof=0) # 有偏协方差
Rho2 = np.corrcoef(x,y)[0,1]

print("协方差为：",Cov)
print("相关系数为：",Rho)