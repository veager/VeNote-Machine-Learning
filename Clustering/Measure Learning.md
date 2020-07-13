## 度量学习

参考文献：

> [1] 豪斯多夫（Hausdorff）距离, CSDN博客, [连接](https://blog.csdn.net/maizousidemao/article/details/105030333)
> [2] Hausdorff distance, 博客园, [连接](https://www.cnblogs.com/yhlx125/p/5478147.html)

### 1. 常用的度量指标

#### 1.1.  豪斯多夫（Hausdorff）距离

Hausdorff距离常用于轨迹之间的相似性度量

给定欧氏空间中的两点集 ${A= \{a_1, a_2, \cdots\} }$ 和 ${B= \{b_1, b_2, \cdots \} }$，豪斯多夫（Hausdorff）距离就是用来衡量这两个点集间的距离：

$$
H(A,B) = \max [h(A,B), h(B,A)]
$$

其中：
$$
h(A,B) = \max_{a \in A} \min_{b \in B} d(a,b)
$$

$$
h(B,A) = \max_{b \in B} \min_{a \in A} d(b,a)
$$

上式中，${H(A,B)}$  称为双向 Hausdorff 距离， ${h(A,B)}$ 称为从点集 ${A}$ 到点集 ${B}$ 的单向 Hausdorff 距离，${h(B,A)}$ 称为从点集B到点集A的单向 Hausdorff 距离。${d(a,b)}$ 表示点 ${a}$ 与点 ${b}$ 的几何距离，通常选择欧氏（Euclidean）距离，即 ${d(a,b) = ||a-b||}$。

