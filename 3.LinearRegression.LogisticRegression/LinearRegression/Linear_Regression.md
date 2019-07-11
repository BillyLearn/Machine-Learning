该文章使用简单线性回归的模型预测在某个高度的海拔，气温会是多少？

现在有一组数据,文件类型如下：

| height | temperature        |
| ------ | ------------------ |
| 0      | 12.8340440094051   |
| 500    | 10.1906489868843   |
| 1000   | 5.50022874963469   |
| 1500   | 2.85466514526368   |
| 2000   | -0.706488218365774 |
| 2500   | -4.06532281046241  |
| 3000   | -7.12747957724466  |
| 3500   | -10.0588785459139  |
| 4000   | -13.2064650515387  |

在这里编辑成  height.vs.temperature.csv  文件。

，运行jupyter notebook

### 数据获取

1. 导入需要的包

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   ```

2. 获取数据，分析可视化数据

   ```python
   # csv 放在当前文件夹的路径下
   data = pd.read_csv("height.vs.temperature.csv")
   plt.figure()
   plt.scatter(data['height'], data['temperature'])
   plt.xlabel("height")
   plt.ylabel("temp")
   plt.show()
   ```

   

![1561565329041](/home/bishi/.config/Typora/typora-user-images/1561565329041.png)		

#### 选择模型

现在，我们要在上面的离散点，找到最合适的线，这样我们就可以预测任何新特征值的数据，也就是（数据集）中不存在的  X (温度) 值。这条线就是线性回归。

选择预测回归的函数：
$$
h(x_i) = wx_i + b
$$
这里：

- $h(x_i)$ 为第 i 个温度（预测值）
- $x_i$ 为第 i 个海拔（样本值）
- $w$  回归线的y轴斜率。
- b  回归线的截距

这里已知的x , y  要创建模型，必须 "学习" 或者估计回归系数的 w  和 b 值，一旦得到估算 w 和 b 的值，就可以使用该模型来进行预测了。

在本文中，使用最小二乘法， 而最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧式距离之和最小。

### 损失函数

现在考虑：最大限度地减少误差error 。比如现在预测目标值 Y 
#### $Y_i = wx + b  + error_i = h(x_i) + error \Rightarrow error_i = Y_i - h(x_i)$

#### $\sum_{i=1}^{m}error^2 = \sum_{i=1}^{m}(Y_i - h(x_i))^2$

注意的是上面的是最小化误差平方的和，而不是平均误差，

平均误差需要除以m :  $argmin_{w, b} = \frac{1}{m}\sum_{i=1}^{m}(Y_i - h(x_i))^2$

而有些多除以2 是为 了方便求导  $argmin_{w, b} = \frac{1}{2m}\sum_{i=1}^{m}(Y_i - h(x_i))^2$ 

最小二乘估计的经典推导使用微积分来找到 w 和 b .

最小化误差平方和的参数估计， 这里这个推导不使用微积分，只使用一些冗长的代数

#### 线性回归推导

#### $\sum_{i=1}^{m}error^2 = \sum_{i=1}^{m} (Y_i - h(x_i))^2 = \sum_{i=1}^m(Y_i - wx_{i} - b)^2$          

加减 $\overline X ， \overline Y$  代入， 上式子等于

#### $ =  \sum_{i=1}^m[(Y_i + \overline Y - \overline Y) - w(x_i + \overline X - \overline X) - b]^2$

展开里面的式子

#### $= \sum_{i = 1}^{m} [Y_i + \overline Y  - \overline Y - wx_{i} - w \overline X + w\overline X - b]^2$

整理

#### $= \sum_{i=1}^m [(\overline Y - w\overline X - b) - (wx_i - w\overline X - Y_i + \overline Y)]^2$

#### $= \sum_{i=1}^{m}[(\overline Y - w\overline X - b)^2]  + \sum_{i=1}^{m}[(wx_i - w\overline X - Y_i + \overline Y)]^2$

#### $ = m(\overline Y - w\overline X - b)^2 + \sum_{i=1}^{m}[(wx_i - w\overline X - Y_i + \overline Y)]^2$

#### $= m(\overline Y - w\overline X - b)^2  + \sum_{i =1}^{m}[w(x_i - \overline X)- (Y_i - \overline Y)]^2$

把右边的平方乘进去

#### $= m(\overline Y - w\overline X - b)^2  + \sum_{i =1}^{m}[w^2(x_i - \overline X)^2- 2w(x_i-\overline X)(Y_i - \overline Y) + (Y_i - \overline Y)^2]$

把右边的拆分开

#### $= m(\overline Y - w\overline X - b)^2 + (w^2  \sum_{i=1}^{m}(x_i - \overline X)^2 - 2  w \sum_{i=1}^{m}(x_i - \overline X) (Y_i - \overline Y) + \sum_{i=1}^{m}(Y_i - \overline Y)^2)$

右边拿出 $\sum_{i=1}^{m}(x_i-\overline X)^2$

#### $= m(\overline Y - w\overline X - b)^2 \\  \ \ \ \  \ + \Bigg (w^2 - \frac{2w\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)} {\sum_{i=1}^{m}(x_i-\overline X)^2} + \frac{\sum_{i=1}^{m}(Y_i - \overline Y)^2}{\sum_{i=1}^{m}(x_i-\overline X)^2}\Bigg ) \ \sum_{i=1}^{m}(x_i-\overline X)^2    \ \ \ \ \ \ \ \ \ \ \ \ (1)$                      

先不看左边的式子$m(\overline Y - w \overline X - b)^2$

这里主要处理右边的式子括号里面的

#### $\bigg (w^2 - \frac{2w\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)} {\sum_{i=1}^{m}(x_i-\overline X)^2} + \frac{\sum_{i=1}^{m}(Y_i - \overline Y)^2}{\sum_{i=1}^{m}(x_i-\overline X)^2} \bigg ) \ \sum_{i=1}^{m}(x_i-\overline X)^2 \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (2)$

加减 $[\frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2}]^2$ , 主要是为了化简类似 $(a - b)^2$ 右边的式子可得 :

#### $(2) = \Big ( w^2 - \frac{2w\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)} {\sum_{i=1}^{m}(x_i-\overline X)^2} + [\frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2}]^2  + \frac{\sum_{i=1}^{m}(Y_i - \overline Y)^2}{\sum_{i=1}^{m}(x_i-\overline X)^2} - [\frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2}]^2 \Big ) \\ \times  \sum_{i=1}^{m}(x_i-\overline X)^2$



最后

#### $\sum_{i=1}^{m}error^2 = (1) = m(\overline Y - w\overline X - b)^2 + \Bigg ( w - \frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2} \Bigg ) ^2  \\\  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  +\sum_{i=1}^{m}(Y_i - \overline Y)^2 \bigg (1 - \bigg [\frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sqrt {\sum_{i=1}^m}(x_i - \overline X)^2 \ \sum_{i=1}^{m}(Y_i - \overline Y)^2}\bigg ]^2 \bigg ) $

前两项都包含$w, b$ , 第三项不包含，可以认为是一个数，所以这已经是最小二乘法误差的最小值了，因为第三项不可能是 0 了，而我们又要使得误差接近于 0 。只有当上面等式的前两个项为 0 时，误差才能达到最小

即 

#### $m(\overline Y - w\overline X - b)^2 = 0 \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (3)$

#### $\Bigg ( w - \frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2} \Bigg ) ^2 = 0 \ \ \ \ \ \ \ \ \ \ \ (4)$



(3)  式求得 

#### $\ \ \ \ \ 0 = m(\overline Y - w\overline X - b)^2 \\ \Rightarrow \ \ \ \ \ \ 0 = \overline Y - w \overline X - b \\ \Rightarrow \ \ \ \ \ \ \  b = \overline Y - w\overline X$																		



(4) 式求得

$0 = \Bigg ( w - \frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2} \Bigg ) ^2 \\ \Rightarrow \ \ \ \ 0 = w -  \frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2}  \\ \Rightarrow \ \ \ \ w =  \frac{\sum_{i=1}^{m}(x_i - \overline X)(Y_i - \overline Y)}{\sum_{i=1}^{m}(x_i - \overline X)^2}$



#### 线性推理结论


#### $SS_{xy} = \sum_{i=1}^{m}(x_{i}-\overline{x})(y_{i}-\overline{y}) = \sum_{i=1}^{m}y_ix_i - m \overline x \ \ \overline y$

#### $SS_{xx} = \sum_{i=1}^{m}(x_{i}-\overline{x})^{2} = \sum_{i=1}^{m}x_i^2 - m(\overline x)^2$

#### $w = \frac{SS_{xy}}{SS_{xx}}$

#### $b = \overline Y - w \overline X$



损失函数代码实现：

```python
def LinearRegression(x, y):
    m = np.size(x)
    # x和y向量的平均值
    m_x, m_y = np.mean(x), np.mean(y) 
   
    SS_xy = np.sum(y * x) - m * m_y * m_x
    SS_xx = np.sum(x * x) - m * m_x * m_x
    
    # 计算回归系数
    w = SS_xy / SS_xx
    
    b = m_y - w*m_x 
    
    return w, b

# 预测 y = wx + b
def predict(w, b, x):
    return (w*x + b)
```

#### 训练模型

```python
X = data['height'].values
y = data['temperature'].values
w, b = LinearRegression(X, y)
```

打印系数 w、b 

```python
print('w = {:.5}'.format(w))
print('b = {:.5}'.format(b))
print("线性回归模型为: Y = {:.5} * X + {:.5} ".format(w, b))
```

> w = -0.0065695
> b = 12.719
> 线性回归模型为: Y = -0.0065695 * X + 12.719 

训练得到的线性回归模型，直线图形如下所示：

```python
pred_Y = predict(w, b, X)
plt.figure()
plt.scatter(data['height'], data['temperature'])
plt.plot(data['height'], pred_Y, c ='red', linewidth=2)
plt.xlabel("height")
plt.ylabel("temp")
plt.show()
```

![1561569332332](/home/bishi/.config/Typora/typora-user-images/1561569332332.png)



#### 预测在 8000米高的海拔，气温会是多少度。

```python
pred = predict(w, b, 8000)
print('在8000米的海拔, 预测气温是: {}'.format(pred))
```

> 在8000米的海拔, 预测气温是: -39.83776550281288



参考资料:

 <https://www.amherst.edu/system/files/media/1287/SLR_Leastsquares.pdf>

<https://github.com/GreedyAIAcademy/Machine-Learning/tree/master/1introduction>