# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:06:17 2020

@author: Nobody
"""
#NumPy系统是Python的一种开源的数值计算扩展。用来存储和处理大型矩阵。
import numpy as np 
a = [1,2,4,5,3,12,12,23,43,52,11,22,22,22]
a_mean = np.mean(a)  #均值
a_med = np.median(a)  #中位数
a_m75 = np.percentile(a,75) # 75th百分位数
Cnt_22 = a.count(22) # 频数
a_R1 = np.percentile(a,75)-np.percentile(a,25) # 四分位差
a_R = np.amax(a)-np.amin(a) #极差
print("a的平均数:",a_mean)
print("a的中位数:",a_med)
print("a的75th百分位数:",a_m75)
print("a中22的频数",Cnt_22)
print("a的四分位差:",a_R1)
print("a的极差",a_R)
#------------------------------------------------------------
from scipy import stats   
'''
Scipy是一个高级的科学计算库，Scipy一般都是操控Numpy数组来进行科学计算，
Scipy包含的功能有最优化、线性代数、积分、插值、拟合、特殊函数、快速傅里叶变换、
信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。
'''
a_m1 =stats.mode(a)[0][0]
print("a的众数:",a_m1)
#-------------------------------------------------------------
import pandas as pd
'''
Pandas是基于NumPy的一个数据分析包，是为了解决数据分析任务而创建的。
Pandas纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。
pandas提供了大量能使我们快速便捷地处理数据的函数和方法。
你很快就会发现，它是使Python成为强大而高效的数据分析环境的重要因素之一。
'''
#将一维数组转成Pandas的Series，然后调用Pandas的mode()方法
ser = pd.Series(a)
a_m2 = ser.mode()[0]
print("a的众数:",a_m2)
