---
title: A breif view of discrete laplacian
subtitle: 本文从简单介绍了 discrete laplacian 和几种具体的实现
# If this post should be treated as a blog post
is_blog_item: true
# Language used in this blog
blog_lang: cn
date: 2024-06-17 21:41:32
tags:
---

拉普拉斯算子（laplacian）在几何处理中是非常基础的概念，简单来说，这表示了一个标量场的梯度场的散度场。梯度场衡量了标量场变化的大小、方向；而这个散度场表示了这些梯度向量指向的特征，这个梯度场在哪些位置是“指向”内部的，哪些位置是“发散”向外面的。一个模型表面的拉普拉斯算子往往可以看成是这个模型表面的一种局部特征，在许多算法中都有运用。

我们从三角网格表面的 laplacian 举例如何计算，这部分更详细的推导可见 Polygon Mesh Processing 这本经典教材。

<center>
<img src="/images/laplacian/1.png" style="width:20%; height:auto;"/>
</center>

上图左边是一个三角形网格表面的一部分，$\text{x}_i$ 和 $\text{x}_j$ 都是模型的顶点，这些顶点通过三角形面组合成整个模型的表面。

我们的输入是这些顶点的三维坐标和他们的连接关系，以及每个顶点对应的一个标量值 $f(\text{x}_i)$​ （也可以理解成一个数组，通过每个顶点作为索引）。

我们的输出是另一个每个顶点对应的标量值 $\Delta f(\text{x}_i)$ ，也就是标量场 $f$ 的散度场 $\nabla f$ 的散度场 $\Delta f$ 。这其中从原本的标量场 $f$  变换到散度场 $\Delta f$  的变换 $\Delta$ 称为拉普拉斯算子（laplacian）。

在先前的描述中，我们说明了散度描述的是“梯度”在一个位置是“发散的”还是“聚集的”，我们可以通过 $f_i-f_j$ 表示从 $\text{x}_j$ 到 $\text{x}_j$ 的梯度，自然可以通过下面的公式表示散度场：

$$
\Delta f(\text{x}_i)=\frac{1}{|N(\text{x}_i)|} \sum_{\text{x}_j\in N(\text{x}_i)}(f_j-f_i)
$$

其中 $N(\text{x}_i)$ 表示顶点 $\text{x}_i$​ 的邻居集合，上面的公式表达的就是，从一个顶点到他的所有邻居的梯度加权平均之和。

不过我们之前说过，我们希望 laplacian 能够表达模型表面的局部特征，我们可以把三角形的面积，夹角放到散度的计算中，也就是把这些“局部特征”放到加权平均的考量中，如下图所示：

<center>
<img src="/images/laplacian/2.png" style="width:30%; height:auto;"/>
</center>

我们按照上图的符号，可以将 laplacian 进一步表示成：

$$
\Delta f(\text{x}_i)=\frac{1}{2A_i}\sum_{\text{x}_j\in N(\text{x}_i)}(\cot \alpha_{ij}+\cot \beta_{ij})(f_j-f_i)
$$

其中 $A_i$ 是图中蓝色区域面积的大小，一般来说，这个蓝色的区域是顶点 $\text{x}_i$ 周边三角形重心的连接而成，我们可以用 $\partial A_i$ 表示这个蓝色区域的边界，这个表示和大物以及数分中是一致的；$\alpha_{ij}$ 和 $\beta_{ij}$ 是两个三角形的夹角，我们通过这些夹角作为梯度模长 $f_j-f_i$ 的加权平均权重。

上面就是经典的 cotangent laplacian，其实按照我们的定义，也能知道在别的表示下 laplacian 应该是什么样的，比如说在 grid （方形网格） 上面：

<center>
<img src="/images/laplacian/3.png" style="width:40%; height:auto;"/>
</center>

laplacian 在应用中，主要用于参入到微分方程求解中，因为梯度 $\nabla$ 相当于是一阶导数，散度 $\Delta$ 相当于是二阶导数，自然可能出现在很多的微分方程里面，比如说非常经典的热传导方程：

$$
\Delta u=\nabla_t u
$$

上面的 $u$ 是热量场，这个场 $u$ 是一个四维函数 $u(x,y,z,t)$ ，与空间和时间有关，$\nabla_t$ 表示了热量关于时间的导数，$\Delta$ 表示了热量关于空间的散度，按照上面的公式，我们可以把 laplacian 看成是一个作用与向量 $f(\text{x})$ 的矩阵 $\Delta$ ，对于每一个给定的显示表达，这个矩阵都是固定的，所以上式左边相当于是一个矩阵左乘一个未知的向量；上式右边可以写成 $\frac{u_t-u_0}{t}$ ，其中初始的热量分布 $u_0$​ 是给定的初始条件，所以这个方程可以通过解线性方程组求出来。实际我们可能采用更精确的隐式积分方法求解，但是我们需要了解 laplacian 在这个过程的作用是让我们把一个方程看成是一个可解的线性方程组。
