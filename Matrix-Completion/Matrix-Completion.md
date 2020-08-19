# 低秩矩阵补全

## 1. Matrix Factorization

> 参考文献
>
> [1] 矩阵补全（Matrix Completion）和缺失值预处理, [连接](https://www.cnblogs.com/wuliytTaotao/p/10814770.html)

利用矩阵分解，进行缺失数据补全。

### 1.1. 缺失值正交映射

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

### 1.2. 矩阵低秩分解

矩阵低秩分解（Low-Rank Matrix Factorization）

对于一个完整的矩阵 ${\mathbf{X} \in \R^{M \times N}}$ 可以分解为两个矩阵 ${\mathbf{U} \in \R^{M \times K}}$ 和 ${\mathbf{V} \in \R^{N \times K}}$ 相乘，其中 ${K < \min\{M,N\}}$，即：
$$
\mathbf{X} = \mathbf{U}  \mathbf{V} ^{\mathbf{T}}
$$

### 1.3. 目标函数

矩阵补全（Matrix Completion）

定义损失函数 ${L}$
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

### 1.4. 编程实现

基于`PyTorch`实现：

## 2. SVT

Singular Value Thresholding Algorithm for Matrix Completion

> 参考文献
>
> [1] 机器学习 | 矩阵补全和奇异值软阈值算法, 知乎, [链接](https://zhuanlan.zhihu.com/p/93400890)
>
> [2] 矩阵补全（Matrix Completion）和缺失值预处理, [连接](https://www.cnblogs.com/wuliytTaotao/p/10814770.html)

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



核范数（unclear norm）
$$
||\mathbf{X}_*|| = tr \left( \sqrt{\mathbf{X}^\mathbf{T} \mathbf{X}} \right)
$$
