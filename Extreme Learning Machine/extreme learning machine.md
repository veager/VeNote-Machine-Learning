## Extreme Learning Machine

极限学习机

### 1. 极限学习机的一般形式

极限学习机是对传统单隐藏层前馈神经网络的一种改进，在模型训练的过程中，输入权重由初始随机生成并且不再改变，只需要学习输出权重，这样可以大大减少模型的训练时间。

假设有 ${N}$ 个样本 ${\left\{({{\mathbf{x}}_{i}},{{\mathbf{y}}_{i}}) \right\}_{i=1}^{N}}$，其中输入向量 ${{\mathbf{x}}_{i}} = {{\left[{{x}_{i,1}},{{x}_{i,2}},...,{{x}_{i,P}} \right]}^{\mathbf{T}} \in {{\mathbb{R}}^{P\times1}}}$，输出向量 ${{\mathbf{y}}_{i}}\text{=}{{\left[{{y}_{i,1}},{{y}_{i,2}},...,{{y}_{i,Q}} \right]}^{\mathbf{T}}}\in {{\mathbb{R}}^{Q\times
1}}$，则具有 ${H}$ 个隐含神经元的ELM输出可以表示为：
$$
f({{\mathbf{x}}_{i}})=\sum\limits_{h=1}^{R}{{{\mathbf{\beta }}_{h}}}g({{\mathbf{w}}_{h}}{{\mathbf{x}}_{i}}+{{b}_{h}}),\space i=1,2,...N
$$


### 2. 加权极限学习机



### 3. 核极限学习机

