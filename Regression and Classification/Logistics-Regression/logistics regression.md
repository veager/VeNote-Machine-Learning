# Logistics Regression

## 1. Logistics Regression



## 2. Softmax Regression

### 2.1. 基本原理

> 参考文献
>
> [1] [机器学习--Softmax回归](https://www.cnblogs.com/whiterwater/p/11415650.html). 博客园
>
> 

$$
p(y=k|\mathbf{x}) = \frac{\text{exp}({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})}{\sum_{k=1}^{K}\text{exp}({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})}
$$

### 2.2. 极大似然估计

$$
L(\theta) =\prod_{m=1}^M p(y_m|\mathbf{x}_m;\theta)
$$

其中${y_n \in \{1,2,...,K\}}$ 表示第 ${m}$ 个样本的正确分类， ${p(y_m|\mathbf{x}_m;\theta)}$ 表示第 ${m}$ 个属于 ${y_m}$ （正确分类）的概率。

通过最大化似然函数求 ${\theta}$ ：

$$
\theta = \text{argmax} \space \log L(\theta)
$$

等价于：

$$
\begin{align}
\theta &= \text{argmin} \left( - \text{log} (L(\theta)) \right) \\
&= \text{argmin} \left( - \sum_{m=1}^{M} \log p(y_m|\mathbf{x}_m;\theta) \right)
\end{align}
$$

所以，Softmax回归的损失函数可以定义为：

$$
\begin{align}
L(\hat{y}_m,y_m) &= \sum_{k=1}^K \Iota(\hat{y}_m = k) \log p(y=k|\mathbf{x}_m;\theta) \\
&= \sum_{k=1}^K \Iota(\hat{y}_m = k) \cdot \log \left( \frac{\text{exp}({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})}{\sum_{k=1}^{K}\text{exp}({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})} \right)
\end{align}
$$

当 ${\log(x)}$ 是以自然对数 ${e}$ 底的对数函数 ${\ln(x)}$ 时，  ${\ln \left( \frac{\text{exp}({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})}{\sum_{k=1}^{K}\text{exp}({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})} \right)}$ 可进一步写作：

$$
\begin{align}
\ln \left( \frac{\exp({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})}{\sum_{k=1}^{K}\text{exp}({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b})} \right) = (\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b) - \ln \left( \sum_{k=1}^{K} \exp ({\mathbf{w}_{k}^{\mathbf{T}}\mathbf{x}+b}) \right)
\end{align}
$$
该形式为 pytorch 中 tutorial 中所采用的 [log_softmax()](https://pytorch.org/tutorials/beginner/nn_tutorial.html#neural-net-from-scratch-no-torch-nn) 的函数形式

### 2.3. Cross Entropy 交叉熵

交叉熵损失函数用于多分类，定义为：

$$
\begin{align}
L(\hat{y}_m,y_m) &= - \sum_{k=1}^K p(y=k|\mathbf{x}_m) \log q(y=k|\mathbf{x}_m) \\
&= - \sum_{k=1}^K p_{k} \log q_{k}
\end{align}
$$
其中 ${p_k}$ 表示 ${\mathbf{x}_m}$ **实际**属于第 ${k}$ 类的概率，${q_k}$ 表示 ${\mathbf{x}_m}$ **预测**属于第 ${k}$ 类的概率。

`pytorch` 中 [`cross_entropy`](https://pytorch.org/docs/stable/nn.functional.html#cross-entropy)

