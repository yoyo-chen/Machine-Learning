# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:06:17 2020

@author: Nobody
"""
import numpy as np 
a = [1,2,4,5,3,12,12,23,43,52,11,22,22,22]
a_var = np.var(a)  #方差
a_std1 = np.sqrt(a_var) #标准差
a_std2 = np.std(a) #标准差
a_mean = np.mean(a)  #均值
a_cv =  a_std2 /a_mean #变异系数
print("a的方差:",a_var)
print("a的标准差:",a_std1)
print("a的标准差:",a_std2)
print("a的变异系数:",a_cv)
