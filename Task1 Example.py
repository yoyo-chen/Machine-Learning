# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:18:37 2020

@author: Nobody
"""

#我们采用函数的递归的方法计算阶乘：
def factorial(n):
    if n == 0:
        return 1;
    else:
        return (n*factorial(n-1)) 
    
l_fac = factorial(365);          #l的阶乘
l_k_fac = factorial(365-40)      #l-k的阶乘
l_k_exp = 365**40                #l的k次方

P_B =  l_fac /(l_k_fac * l_k_exp)     #P(B）
print("事件B的概率为：",P_B)
print("40个同学中至少两个人同一天过生日的概率是：",1 - P_B)
