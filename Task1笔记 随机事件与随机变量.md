理论部分

#### 1. 随机事件

##### 1.1 基本概念：随机事件，样本空间等

##### 1.2 概率

##### 1.3 古典概型

##### 1.4 条件概率

* 条件概率： 
  $P(A|B) =  \frac {P(AB)}  {P(B)} $ 

##### 1.5 全概率公式和贝叶斯公式

* 全概率公式： 
  $P(A) = \sum_{i=1}^{\infty } {P(B_i)}P(A|B_i)$ 
* 贝叶斯公式：
  $P(B_i|A) =\frac {P(B_i A)}  {P(A)}  =  \frac {P(A|B_i )P(B_i)}  {\sum_{j=1}^{\infty }P( B_j)P(A|B_j)}  ,i=1,2,... $ 
* 先验概率： 
  $P(B_i)(i=1,2,...)$
* 后验概率： 
  $P(B_i|A)（i=1,2,...）$ 当结果为A时Bi发生的概率

#### 2. 随机变量

##### 2.1 随机变量及其分布

##### 2.2 离散型随机变量

##### 2.3 常见的离散型分布

* 伯努利分布/二项分布：

$$
P(A_k） =C^k_np^k(1-p)^{n-k},k=0,1,2,...n.
$$

##### 2.4 随机变量的数字特征

###### 2.4.1 数学期望

* 若 $X, Y$ 相互独立，则$E(XY) = E(X)E(Y)$

###### 2.4.2 方差 

$$
Var （X） =E\{  [X-E(X)]^2\} 
$$
* 若 $X, Y$ 相互独立，则$Var(X+Y) = Var(X) +Var(Y)$ 

###### 2.4.3 协方差和相关系数(线性相关，皮尔逊相关系数)

* 协方差：

$$
Cov(X, Y) = E\{  [X-E(X)] [Y-E(Y)]\}
$$
* 若 $X, Y$ 相互独立，则$Cov(X，Y) = 0$

* 相关系数：

$$
\rho（X,Y） = \frac{Cov(X，Y)}{\sqrt {Var(X)} \sqrt {Var(Y)}}
$$
* 相关系数在-1到1之间，绝对值 $|\rho（X,Y）|$ 表示$X,Y$之间的线性相关程度。越接近1，相关度越大。小于零表示负相关，大于零表示正相关。

* 两个变量的位置和尺度的变化并不会引起该系数的改变，即：
  $$
  \rho（aX+b,cY+d） = \rho（X,Y）
  $$




### 练习部分

###### python实现二项分布，协方差和相关系数以及贝叶斯公式

##### 练习1.1 二项分布

```python
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
```



##### 练习1.2 协方差和相关系数

```python
import math
import numpy as np

## 未中心化
x = np.array([1, 2, 3, 5, 8])
y = np.array([0.11, 0.12, 0.13, 0.15, 0.18])
## 中心化，相关系数不变
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
Cov2 = np.cov(x,y)[0,1]*(n-1)/n # numpy中的cov()是样本协方差，除的是n-1
Rho2 = np.corrcoef(x,y)[0,1]

print("协方差为：",Cov)
print("相关系数为：",Rho)
```



##### 练习1.3 贝叶斯公式

```python
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
```



#### 以上 

#### 2020.06.22

