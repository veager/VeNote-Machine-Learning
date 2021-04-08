# Statistical Method LOESS

局部加权回归（locally weighted scatterplot smoothing，LOWESS或LOESS）

# 1. 数学原理



# 2. 程序原理

使用`sklearn.base.BaseEstimator` 和`sklearn.base.RegressorMixin`作为基类



# 3. 参数说明

## 3.1. 基本情况

`LOESS(k=5, kernel='bisquare', distance='manhattan', p='1', istimeseries=True)`类

- 参数：

`k`：`int`类型，邻近点个数

`kernel`：`str`类型，核函数，可选`'bisquare','tricube','epanechnikov'`

`distance`：`str`类型，距离类型。当`istimeseries=False`，该参数才会生效，用于传入[`sklearn.neighbors.NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)类中的`metric`参数。

`p`：`int`类型，闵可夫斯基距离（Minkowski）指数。当`istimeseries=False`，该参数才会生效，用于传入[`sklearn.neighbors.NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)类中的`p`参数。

`istimeseries`：`bool`类型。是否使用时间序列类型评估

- 方法：

`fit(X, y)`：

`predict(X)`：

## 3.2. 其他说明：

`istimeseries`类型决定了具体的算法：

`istimeseries=True`是针对一维时间序列优化的算法。

要求：(1) `X`和`y`均为1维向量（即 `x.shape=y.shape=(n,)`）；(2) 时间变量 `x` 必须维等间隔的；(3) 预测时，输入的`x`必须存在于训练样本`x`中。

`istimeseries=False`是通过[`sklearn.neighbors.NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)类搜索邻近值，其要求输入数据`X`必须维2维或2维以上。如果`X`维1维，要转换成二维的形式，即`(n,)->(n,1)`。



# 参考文献

[1] The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition, Chapter 6 Kernel Smoothing Methods, [link](https://web.stanford.edu/~hastie/ElemStatLearn/).

[2] [1]虞乐,肖基毅. 数据挖掘中强局部加权回归算法实现[J]. 电脑知识与技术,2012,8(07):1493-1495. doi: