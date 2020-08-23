# 矩阵补全

## 1. 基于PPCA的矩阵补全算法（未完成）

概率主成分分析 Probabilistic Principal Component Analysis

假设有 ${N}$ 个样本，每个样本 ${\mathbf{y}_n = \left[y_n(1), y_n(2), \cdots, y_n(D) \right]^\mathbf{T} \in \R^D}$ 由潜变量（latent variable）${x_n \in \R^Q}$ 决定：

$$
\mathbf{y}_n = \mathbf{W} \mathbf{x}_n + \boldsymbol{\mu} + \boldsymbol{\epsilon}_n
$$

其中 ${\mathbf{W} \in \R^{D \times Q}}$ 表示潜变量到观测变量的映射矩阵，${\boldsymbol{\mu} \in \R^D}$, 为每个特征的平均值：

$$
\boldsymbol{\mu} = \left[
\frac{1}{N} \sum_{n=1}^N y_n(1), \frac{1}{N} \sum_{n=1}^N y_n(2), \cdots, \frac{1}{N} \sum_{n=1}^N y_n(D)
\right]^\mathbf{T}
$$

${\boldsymbol{\epsilon}_n}$ 为高斯误差，${\boldsymbol{\epsilon}_n \sim \mathcal{N}_D(0, \sigma^2 \mathbf{I})}$ 。

则 log-likelihood 函数为：

$$
\arg \max_{\boldsymbol{\mu}, \mathbf{W}, \sigma^2} L_{c0} = 
\arg \max_{\boldsymbol{\mu}, \mathbf{W}, \sigma^2} \ln \left\{\prod_{n=1}^{N} p(\mathbf{y}_n|\boldsymbol{\mu}, \mathbf{W}, \sigma^2) \right\}
$$

其中：

$$
\begin{align}{}
p(\mathbf{y}_i| \boldsymbol{\mu}, \mathbf{W}, \sigma^2) &= \int 
p(\mathbf{y}_i| \mathbf{x}_i, \boldsymbol{\mu}, \mathbf{W}, \sigma^2) 
p(\mathbf{x}_i) \text{d} \mathbf{x} \\

p(\mathbf{y}_i|\mathbf{x}_i, \boldsymbol{\mu}, \mathbf{W}, \sigma^2) &= 
\mathcal{N}(\mathbf{y}_i|\mathbf{W} \mathbf{x}_n + \boldsymbol{\mu}, \sigma^2 \mathbf{I}) \\

p(\mathbf{x}_i) &= \mathcal{N}(\mathbf{x}_i|0, \mathbf{I})
\end{align}{}
$$



## 2. 基于KNN的缺失数据补全方法

>参考文献：
>
>[1] nan_euclidean_distances, sklearn Tutorials, [链接](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html)
>
>[2] KNNImputer, , sklearn Tutorials, [链接](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)

### 2.1. 带虑缺失值的 euclidean distances

给定两个样本 ${x_1=[3, \text{na}, \text{na}, 6]^{\mathbf{T}}}$ 和  ${x_2=[1, \text{na}, 4, 5]^{\mathbf{T}}}$  ，${x_1}$ 和 ${x_2}$ 的 euclidean 计算公式为：

$$
d(x_1, x_2) = \sqrt{\frac{4}{2} \left((3-1)^2 + (6-5)^2 \right)}
$$


### 2.2. 基于KNN的缺失数据补全

根据计算的 distance，以最近的 ${K}$ 个样本的平均值作为缺失值的补全值。 

### 2.3. 代码实现

```python
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
# 生成初始缺失矩阵
np.random.seed(42)
X = np.random.randint(0, 100, size=(20,5)).astype(float)
X[X % 3==0] = np.nan
print(X)
# 
# X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])

n_neighbors = 7
(n_samples, n_features) = X.shape


# --- 方法一开始 ------------------
Y = X.copy()
# 计算带缺失数据的欧拉距离
dist = nan_euclidean_distances(X)
# 查找每个样本距离最近K个的样本的索引
dist_index = np.argsort(dist, axis=1)[:, 1:n_neighbors+1]

for i in range(n_samples):
    # 找到样本i X[i, :]的最近的K个样本
    k_neighsamples = X[dist_index[i, :]]
    for j in range(n_features):
        if np.isnan(Y[i, j]):
            Y[i, j] = np.nanmean(k_neighsamples[:, j])
# --- 方法一结束 ------------------

# --- 方法二开始：KNNImputer实现 ---
imputer = KNNImputer(n_neighbors=n_neighbors)
Z = imputer.fit_transform(X)
# --- 方法二结束 ------------------
print(Z - Y)
```

**方法一**与**方法二**部分缺失值补全结果不太一样，有一些细节方面尚待考虑。

## 3. 基于低秩分解的矩阵补全算法

> 参考文献
>
> [1] 矩阵补全（Matrix Completion）和缺失值预处理, [连接](https://www.cnblogs.com/wuliytTaotao/p/10814770.html)

低秩矩阵分解 Low-Rank Matrix Factorization 

### 3.1. 预备知识

#### 3.1.1. 缺失值正交映射

设 ${\mathbf{Y} \in \R^{M \times N}}$ 为观测矩阵，将被观测元素的索引合集记作 ${\Omega}$ 。令 ${\mathcal{P}_\Omega(\cdot)}$ 表示一个作用于 ${\Omega}$ 的正交映射 (orthogonal projection)，其定义：
$$
[\mathcal{P}_\Omega(\mathbf{Y})]_{m,n} = \left\{
\begin{array}{left}
y_{mn}, & \text{if} \space (m, n) \in \Omega\\ 
0, & \text{otherwise}
\end{array}
\right.
$$
比如：

#### 3.1.2. 矩阵低秩分解

矩阵低秩分解（Low-Rank Matrix Factorization）

对于一个完整的矩阵 ${\mathbf{X} \in \R^{M \times N}}$ 可以分解为两个矩阵 ${\mathbf{U} \in \R^{M \times K}}$ 和 ${\mathbf{V} \in \R^{N \times K}}$ 相乘，其中 ${K < \min\{M,N\}}$，即：
$$
\mathbf{X} = \mathbf{U}  \mathbf{V} ^{\mathbf{T}}
$$

### 3.2. 基本原理

假定原始矩阵可以低秩分解为 ${\mathbf{U}  \mathbf{V} ^{\mathbf{T}}}$，通过寻找矩阵 ${\mathbf{U}}$ 和 ${\mathbf{V}}$ 使得重构的矩阵 ${\mathbf{X}}$ 与原始缺失矩阵 $\mathbf{Y}$ 中未缺失的值的差距最小。

定义损失函数 ${L}$：
$$
\begin{align}{}
L &= \left\| \mathcal{P}_\Omega \left( \mathbf{Y} - \mathbf{X} \right) \right\| ^2_F \\
&= \left\| \mathcal{P}_\Omega \left( \mathbf{Y} - \mathbf{U}  \mathbf{V} ^{\mathbf{T}} \right) \right\|^2_F \\
&= \sum_{m, n, x_{mn} \not = \text{nan}} \left(x_{mn} - \sum_{k=1}^K u_{mk}v_{nk} \right)^2 
\end{align}{}
$$

引入正则化，

$$
\begin{align}
J &= \left\| \mathcal{P}_\Omega \left( \mathbf{Y} - \mathbf{U}  \mathbf{V} ^{\mathbf{T}}\right) \right\|^2 + 
\frac{\beta}{2} \left( \|\mathbf{U}\|^2 + \|\mathbf{V}\|^2 + \|\mathbf{b}_u\|^2  + \|\mathbf{b}_v\|^2 \right)

\\ &=\sum_{m,n,x_{mn} \not = nan} 
    \left(
    	x_{mn} - 
    	\sum_{k=1}^K u_{mk} v_{nk} - 
    	b - 
    	b_{u_m} - 
    	b_{v_n} 
    \right)^2 + 
\frac{\beta}{2} 
	\left( 
		\sum_{m,k}(u_{mk})^2 + 
        \sum_{n,k}(v_{mk})^2 + 
        \sum_{m}(b_{u_m}) ^2 +
        \sum_{n}(b_{v_n})^2 
    \right)
\end{align}
$$

其中：${x_{mn} = \sum_{k=1}^{K} u_{mk} v_{nk} + (b + b_{u_m} + b_{v_n})}$；${b = \sum_{m, n, x_{mn} \not = text{nan}} x_{mn}/{Q}}$，${Q}$ 为（未缺失）样本总数；

### 3.3. 编程实现

基于`PyTorch`实现：

## 4. 奇异值软阈值算法（未完成）

Singular Value Thresholding Algorithm for Matrix Completion

> 参考文献
>
> [1] 机器学习 | 矩阵补全和奇异值软阈值算法, 知乎, [链接](https://zhuanlan.zhihu.com/p/93400890)
>
> [2] 矩阵补全（Matrix Completion）和缺失值预处理, [连接](https://www.cnblogs.com/wuliytTaotao/p/10814770.html)

### 4.1. 基本原理

核范数（unclear norm）
$$
||\mathbf{X}_*|| = tr \left( \sqrt{\mathbf{X}^\mathbf{T} \mathbf{X}} \right)
$$

### 4.2. 编程实现

基于`PyTorch`实现：