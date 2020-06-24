#### 理论部分

##### 1. 数理统计的概念
* 数理统计是数学的一个分支，分为描述统计和推断统计。

  描述统计：获取样本，分析样本数据特征。

  推断统计：根据样本推断总体数据的特征。

###### 1.1 基本概念
* 总体，个体，样本
* 抽样条件：随机 独立 同分布
* 样本的两重性：一组随机变量，一组观测值

###### 1.2 统计量与抽样分布
* 统计量是反应样本特定性质的函数
* 统计量的分布为抽样分布

###### 1.3 常用统计量
* 样本均值

$$
\overline X =  \frac{1} {n} {\sum_{i=1}^{n}X_i}
$$

* 样本方差

$$
S^2 =  \frac{1} {n-1} {\sum_{i=1}^{n}(X_i-\overline X)^2}
$$

* **k阶样本原点矩**

$$
A_k =  \frac{1} {n} {\sum_{i=1}^{n}X_i^k}
$$

* **k阶样本中心矩**

$$
M_k =  \frac{1} {n} {\sum_{i=1}^{n}(X_i-\overline X)^k}
$$

* 顺序统计量

$$
x_{（1）}<=x_{（2）}<=...<=x_{（n）}
$$

##### 2. 描述性统计
###### 2.1 数据集中趋势的度量
* 平均数 $\overline x$
  $$
  \overline x =  \frac{1} {n} {\sum_{i=1}^{n}x_i}
  $$
  
* 中位数 $m_{e}$

* 频数

* 众数

* 比较

|       |               优点                         |    缺点       |
| :---: | :------------------------------------------: | :-----------: |
|  均值  |       充分利用所有数据，适用性强              | 容易受极端值影响 |
| 中位数 |             不受极端值影响                  |    缺乏敏感性    |
|  众数  | 不受极端值影响；当数据具有明显的集中趋势时，代表性好 |    缺乏唯一性    |

* **百分位数** $m_{p}$   0.5分位数（50th百分位数）=中位数

###### 2.2 数据离散趋势的度量
* 方差 $s^2$
  $$
  s^2 =  \frac{1} {n-1} {\sum_{i=1}^{n}(x_i-\overline x)^2}
  $$
  
* 标准差 $s$
  $$
  s=\sqrt{s^2}
  $$
  
* **极差**
$$
R = x_{（n）}-x_{（1）} = max(x)-min(x)
$$
* **变异系数** 

  消除尺度和量纲的影响
$$
{\rm CV} = 100*\frac{s}{\overline x}(\%)
$$
* **四分位差**

  75th百分位与25th百分位数的差值。

  四分位差反映了数据的集中程度，该值越小，表示数据越集中于中位数附近，该值越大，表示数据越发散于两端。
  
  和中位数一样，不受极端值影响，稳健性好。
  $$
  R_1 = Q_3-Q_1
  $$
  

###### 2.3 分布特征
* 离散变量与连续变量
* 概率函数
* 分布函数（概率累积函数）
* 正态分布（高斯分布）

###### 2.6 偏度与峰度
* **偏度(skewness)**

  衡量实数随机变量概率分布的不对称性。

  偏度 > 0 $\Rightarrow$  右偏分布，正偏分布，函数曲线向左偏

  偏度 = 0 $\Rightarrow$  正态分布

  偏度 < 0 $\Rightarrow$  左偏分布，负偏分布，函数曲线向右偏
  $$
  Skew(X)=E[(\frac{X-\mu}s)^3] = \frac{m_3}{s^3}
  $$

  $$
  m_3 =  \frac{1} {n} {\sum_{i=1}^{n}(x_i-\overline x)^3}
  $$

  其他计算公式，例如
  $$
  Skew(X)=\frac{n^2m_3}{(n-1)(n-2)s^3}
  $$
  
* **峰度(kurtosis)**

  衡量实数随机变量概率分布的峰态。

  峰度越高，意味着分布越陡。

  正态分布的峰度 = 3
  $$
  Kurt(X)=E[(\frac{X-\mu}{s})^4] = \frac{m_4}{s^4}
  $$

  $$
  m_4 =  \frac{1} {n} {\sum_{i=1}^{n}(x_i-\overline x)^4}
  $$

  超值峰度（excess kurtosis）（正态分布峰度为0）
  $$
  Kurt(X)= \frac{m_4}{s^4}-3
  $$
  其他计算公式，例如
  $$
  Kurt(X)= \frac{n^2(n+1)m_4}{(n-1)(n-2)(n-3)s^4}-3\frac{(n-1)^2}{(n-2)(n-3)}
  $$
  

  

  

#### 练习部分
###### python实现数据各维度的描述性分析
参考资料

[numpy教程：统计函数Statistics](https://blog.csdn.net/pipisorry/article/details/48770785)

[使用Python进行描述性统计 ](https://www.cnblogs.com/jasonfreak/p/5441512.html)

* 平均数，中位数，众数，百分位数，频数，四分位差，极差

```python
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
a_m2 = ser.mode()
print("a的众数:",a_m2)
```



* 方差，标准差，变异系数

```python
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
```



* 生成标准正态分布，偏度系数，峰度系数

```python
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
```

