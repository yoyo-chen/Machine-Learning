# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:51:58 2020

@author: Nobody

实现贝叶斯公式

"""
import numpy as np

n = 2
m = 1
# B0：肝癌患者; B1：非肝癌患者
# P(A|B_i) = [0.95, 0.1] 肝癌患者阳性概率 0.95，非肝癌患者阳性概率 0.1
Patb = np.array([0.95, 0.1])
# P(B_i) = [0.0004, 0.9996] 肝癌患者概率 0.0004，非肝癌患者概率 0.9996
Pb = np.array([0.0004, 0.9996])

Pabm = Patb[m-1]*Pb[m-1]

Pa = 0
for i in range(n):
    Pa = Pa + Patb[i]*Pb[i]

P = Pabm/Pa
print("P(B_m|A)为：",P)