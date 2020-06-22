# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 13:39:58 2020

@author: Nobody

计算n重Bernoulli试验中概率为p的事件A出现a次的概率
输入: n, p, a
"""

#我们采用函数的递归的方法计算阶乘：
def factorial(m):
    if m == 0:
        return 1;
    else:
        return (m*factorial(m-1))
    
print("计算n重Bernoulli试验中概率为p的事件A出现a次的概率")
#print("Bernoulli试验次数为")
#n = int(input())
#print("事件A出现的概率为")
#p = float(input())
#print("A出现的次数为")
#a = int(input())
n = 4
a = 4
p = 0.5
n_fac = factorial(n); 
a_fac = factorial(a);
x=n-a;
x_fac = factorial(x);
Ber = n_fac/a_fac/x_fac*p**a*(1-p)**(n-a);

print(str(n)+"重Bernolli试验中概率为"+str(p)+"的事件A出现"+str(a)+"次的概率为：",Ber)