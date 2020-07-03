# 支持向量机

支持向量机（Support Vector Machine, SVM）

## 1. 支持向量机

支持向量机（Support Vector Machine, SVM）

## 2. 支持向量回归

支持向量回归（Support Vector Regression, SVR）

### 2.1. 基本原理



### 2.2. `sklearn`实现



## 3. 最小二乘支持向量机

最小二乘支持向量机（Least Square Support Vector Machine, LSSVM）

## 4. 最小二乘支持向回归

最小二乘支持向量回归（Least Square Support Vector Regression, LSSVR）

## 5. 核方法

### 5.1. 核函数的定义和基本性质



### 5.2. 常用核函数

#### 5.2.1. 常用核函数

#### 5.2.2. wavelet kernel

设 ${\mathbf{x} = \left[x_1,x_2,\cdots,x_D \right]^\mathrm{T} \in \mathbb{R}^D}$ 和 ${\mathbf{x}^{\prime} = [x_1^{\prime}, x_2^{\prime},\cdots, x_D^{\prime} ]^\mathrm{T} \in \mathbb{R}^D}$ 分别为两个样本，${d=1,2,...,D}$ 为样本的维度；${a(a \in \mathbb{R})}$ 表示尺度因子（dilation factor）； ${c(c \in \mathbb{R})}$ 表示位移因子（translation factor）。

正定小波核具有以下形式：
$$
K\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\prod_{d=1}^{D} h\left(\frac{x_{d}-c_{d}}{a}\right) h\left(\frac{x_{d}^{\prime}-c_{d}^{\prime}}{a}\right)
$$
具有平移不变形（ translation-invariant）的小波核：
$$
K\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\prod_{d=1}^{D} h\left(\frac{x_{d}-x_{d}^{\prime}}{a}\right)
$$

其中${h(\cdot)}$ 为小波母函数（wavelet motherfunction），文献[1,2,3]使用以下形式的小波核函数：

$$
h(x)=\cos \left(k \cdot \frac{x}{a}\right) \cdot \exp \left(-\frac{x^{2}}{a^{2}}\right)
$$

根据文献[1,2]的推荐，

$$
h(x)=\cos \left(1.75 x \right) \cdot \exp \left(-\frac{x^{2}}{2}\right)
$$

所以用于SVM的小波核函数可以写作：

$$
\begin{aligned} 
	K\left(\mathbf{x}, \mathbf{x}^{\prime}\right) 
	&= \prod_{d=1}^{D} h\left(\frac{x_{i}-x_{i}^{\prime}}{a}\right) 
	\\ &=\prod_{i}^{N}\left(\cos \left(1.75 \times \frac{\left(x_{d}-x_{d}^{\prime}\right)}{a}\right) \exp \left(-\frac{\left(x_{d}-x_{d}^{\prime}\right)^{2}}{2 a^{2}}\right)\right) 
\end{aligned}
$$

> 参考文献：<br/>[1] Harold H. Szu, Brian A. Telfer, Shubha L. Kadambe, "*Neural network adaptive wavelets for signal representation and classification*," Opt. Eng. 31(9).<br/>[2] Li Zhang, Weida Zhou and Licheng Jiao, "Wavelet support vector machine," in *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, vol. 34, no. 1, pp. 34-39.<br/>[3] Jianwu Zeng and Wei Qiao, "Short-term wind power prediction using a wavelet support vector machine," *2013 IEEE Power & Energy Society General Meeting*, Vancouver, BC, 2013, pp. 1-1.


### 5.3. `sklearn`自定义核函数

在`sklearn.svm.SVR`类中，可以通过`kernel`参数（`kernel=my_kernel`）指定定义的核函数，或者`kernel='precomputed'`使用自定义的Gama矩阵（使用`kernel='precomputed'`时，在`fit()`和`predict()`过程传入的`X`为计算好的Gama矩阵。详细参考`sklearn.svm.SVR`的[说明](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)）

- 自定义kernel

```python
def my_kernel(X, Y):
	return np.dot(X,Y.T)
```

需要注意的是，在`SVR`在计算的训练（`fit()`）过程中，传入`X`和`Y`为一样的两个矩阵，矩阵的形式为`(n_samples, n_features)`。在预测的过程中（`predict()`），传入的`X`为训练集（形式为`(n_samples_train, n_features)`），传入的`Y`为测试集（形式为`(n_samples_test, n_features)`）。

- 自定义wavelet kernel
```python
def WaveletKernel(X, Y, a=0.05): 
    n_train = X.shape[0]
    n_test = Y.shape[0]
    kernel = np.zeros((n_train, n_test))
    for i in range(n_train):
        for j in range(n_test):
            delta = X[i, :]-Y[j, :]
            kernel[i, j] = np.prod(np.cos(1.75*delta/a) * np.exp(-delta*delta)/(2*a*a))
    return kernel
```
### 5.4. Multi-Kernel

