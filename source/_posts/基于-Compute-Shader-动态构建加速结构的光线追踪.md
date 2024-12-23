---
title: 基于 Compute Shader 动态构建加速结构的光线追踪
subtitle: 本文介绍了我通过 OpenGL 4.3 的 Compute Shader 实现每帧重新构建场景中 mesh 的 LBVH，并利用该加速结构做简单的阴影光线追踪。
is_blog_item: true
blog_lang: cn
date: 2024-12-14 22:28:19
tags:
mathjax: true
---

关于光追有非常多资源，例如 [RayTracingInOneWeekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) 系列，[Games202](https://games-cn.org/games202/)，[GLSL-PathTracer](https://github.com/knightcrawler25/GLSL-PathTracer)。关于实时硬件光追也有很多可以参考的资源 [Vulkan RayTracing](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)，[DXR RayTracing](http://cwyman.org/code/dxrTutors/dxr_tutors.md.html)。

不过我现在的代码库都是基于 OpenGL 的，转到 Vulkan 和 Direct12 这样带硬件光追支持的 api 比较困难，而且我也希望编写一个能够在 gpu 上构建，遍历的加速结构，既能够用于实验光追，也能给之后的碰撞检测和物理模拟模块打下一个基础。所以我选择用 OpenGL 4.3 引入的 Compute Shader 实现动态的 BVH 构建。在简单的调研后，我发现了一种能够非常快速构建 BVH 的[算法@Tero Karras](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)。关于这个算法 nvidia 也有一系列博客详细介绍其步骤 [blog](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)，我也看到了很多对这篇文章的复现代码和博客，[VkLBVH](https://github.com/MircoWerner/VkLBVH)，[lbvh](https://github.com/ToruNiina/lbvh)， [基于GPU的LBVH算法](https://zhuanlan.zhihu.com/p/673723218?utm_psn=1848892512913203200)，[CUDA学习笔记：并行构造BVH](https://zhuanlan.zhihu.com/p/423351818)，[并行构建BVH](https://zhuanlan.zhihu.com/p/97056642)。但是很可惜，大部分代码复现都是使用 cuda 或者 vulkan，看起来比较困难，复现的博客并没有给出很多的代码示例，对于一些细节的问题也不是很清楚。

为此，我把我对原论文的理解，和这些已有项目的信息做了一个整合，补全了一些细节问题，并会给出足够的代码示例。首先我们可以看看构建出来的 BVH：

<center>
<img src="/images/lbvh/1.png" style="width:60%; height:auto;"/>
</center>

这是通过上面的加速结构，计算光追阴影的结果。光追阴影不存在 shadow map 容易出现的 shadow acne 和 Peter panning 问题，不需要调整很多参数就能够达到很真实的效果，有了加速结构和很强大的显卡（测试平台 RTX 2050 laptop）也能够实现实时运行：

<center>
<img src="/images/lbvh/rt_hard_shadow.png" style="width:60%; height:auto;"/>
</center>

本文将主要分为下面的几个部分，涉及到算法的主要步骤均在 GPU 进行：

1. LBVH 算法步骤，算法动机
2. 从 Vertex Buffer 和 Index Buffer 计算 Morton Code
3. GPU 上 Radix Sort 排序算法的实现
4. 通过排序的 Morton Code 构建 Binary Radix Tree
5. 通过 Binary Radix Tree，自底向上计算每个节点的 Bounding Box
6. 每一帧重新构建加速结构，遍历该结构用于简单的阴影光线追踪
7. 一个简易的总结

## LBVH 算法的步骤，动机

这里提到的 LBVH 算法来自论文 Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees，这篇论文于 2012 年发表，提出通过排序后的 Morton Code，并行构建适合多种空间数据结构的层次结构。用 Morton Code 构建 BVH 的思路并不是这篇文章首先提出的，但是这篇文章首先提出了通过 Morton Code 并行地构建树状结构的算法。

具体来说，我们算法的输入是这个模型所有三角形重心的 Morton Code 排序后的结果 `gCode`，其大小为 $N_t$，即模型三角形的个数。算法的输出是这个层次结构的所有节点 `gNode`，每个节点保留其左右子节点和父亲节点，由于这个算法构建的树是一颗满二叉树，输出的大小为 $2\times N_t - 1$，这棵树有 $N_t$ 个叶子节点，$N_t - 1$ 个中间节点。

我们有了这个层次结构的所有节点后，可以从叶子节点出发，计算叶子节点对应三角形的 Bounding Box，再逐步合并 Bounding Box 直到根节点，得到整个 BVH。

### Morton Code

在详细介绍怎么构建这个层次结构之前，我们必须知道什么是 Morton Code，这里从 [nvidia 的博客](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)借一张图：

<center>
<img src="/images/lbvh/morton_code.png" style="width:60%; height:auto;"/>
</center>

我们可以把 Morton Code 理解成对于空间中点的坐标 $(x,y,z)$ 进行编码，给定一个点在三维空间中的坐标，我们将坐标编码成 Morton Code 之后，沿着 Morton Code 单调递增的方向给空间中对应的点连线，最终得到的连线呈现 `zig-zag` 的形状，也就是上图中右侧黑线的形状，这意味着，邻近的 Morton Code 对应的点，在空间中的位置也是邻近的，如果我们在排好序的 Morton Code 里面取一个连续的区间 $[i,j]$，这个区间对应的所有点 $(p_i,\dots,p_j)$ 大概率在空间中也都是邻近的。如果我们能够**让树的每一个节点对应排好序的 Morton Code 中的一个区间**，每一个节点就大致对应了空间中**邻近的一些点**。

这样就非常好了，我只需要找到每一个节点所对应的区间是什么，并且找到一种办法在这个区间 $[i,j]$ 中分出一个左右来，左边的子区间 $[i,\gamma]$ 给左子节点，右边的子区间 $[\gamma+1,j]$ 给右子节点，就可以构建出这个树状结构了，并且每个节点还都大致对应了空间中的一块聚在一起的点，这种结构就很符合用于加速的空间数据结构的思路了。

那么具体怎么算一个坐标 $(x,y,z)$ 的 Morton Code？博客中给了一段非常清晰的代码：

```cpp
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}
```

首先我们找一个办法，把点的坐标 $(x,y,z)$ 的每一维映射到 $[0,1]$ 区间里面，然后我们取小数位的前 10 位二进制数，通过在这 10 位二进制数的每一位数后面填充两个 `0`，我们把一个 10 位的二进制数填充成了一个 30 位的二进制数，接着我们按照 x，y，z 的顺序把这三个 30 位的二进制数插空填充成一个 30 位的二进制数（实际中我们直接用一个 4 字节的无符号整数）。从计算的过程中我们不难理解为什么排好序的 Morton Code 在空间中是这样的形状。毕竟对两个 Morton Code 做比较，等价于按照 x，y，z 的顺序逐位比较两个点的坐标。

值得一提的是，如果有两个点的 Morton Code 是相同的，而且很可能出现这种情况，我们必须要对后面的算法做特殊处理。

### 构建 Binary Radix Tree

这里还是从 nvidia 的同一个博客里面借一张图：

<center>
<img src="/images/lbvh/build_brt_1.png" style="width:50%; height:auto;"/>
</center>

我们先前已经提到了，我们希望让每一个节点对应到排好序的 Morton Code 的一个区间上，这样每一个节点都能够形象的对应到空间中差不多邻近的一块点（一些三角形的重心）上。很自然的，我们希望从一个节点的区间 $[i,j]$ 里面找到一个合理的分界点 $\gamma$ ，使得当前节点对应的区间分成了两个子区间 $[i,\gamma]$ 和 $[\gamma+1,j]$。问题来了，我们怎么找到这个分界点 $\gamma$ ，确保我们对于当前节点的划分是合理的？

提到对于一个节点的划分，我们很容易想到正常在 CPU 上构建 BVH 时，我们会用一种叫做 [Surface Area Heuristics](https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/) 的方法把一个节点分成两个子节点，简单来说，SAH 计算当前节点的 Bounding Box 表面积有多大，然后对于每一个可能划分的位置，计算在这里划分之后，左右子节点 Bounding Box 的表面积，用这些表面积作为一个分数，判断当前应该在哪里分割节点，还是干脆不分割节点。实际中 SAH 有一些优化能够让他更快，而且基于我们正在提的这篇工作 LBVH 之后也有把 SAH 的想法和 LBVH 的办法结合起来的构建算法，但是我们这里关注 LBVH 的做法。

最朴素的做法当然是按照数量划分，直接让 $\gamma=\frac{i+j}{2}$，但是这样等于是忽略了 Morton Code 对应的三角形在空间中的分布情况，作者的办法是观察当前节点对应的区间 $[i,j]$ 的 Morton Code $[c_i,\cdots,c_j]$ 里面公共前缀的下一位。

想象我们有很多排好序的二进制数 $[c_i,\cdots,c_j]$ ，这个问题中就是我们的 Morton Code，所有这些二进制数一定有一个最长的公共前缀（我们也允许这个公共前缀的长度是 0），那么对于这个公共前缀的下一位，他的分布一定是这样的：

$$
[0,\cdots,0,0,1,1,\cdots,1]
$$

毕竟这些二进制数是排好序的，所以我们如果找到这些数中 $0$ 位和 $1$ 位的分界点，把这个分界点当做是 $\gamma$，听起来是非常有道理的。毕竟类似的 Morton Code 在空间中的位置也是大致相近的，公共前缀更长的 Morton Code 当然就是更“类似”的。这样我们确保两个子节点都是内部更类似，但是互相不太类似的。虽然实际情况并不会这么美好，但是道理上他很对。

道理说完了，我们怎么才能让这个想法变成代码，而且是能够在 GPU 上并行的代码？

实际上，基于我们上面说到的，我们想要让分界点 $\gamma$ 是当前节点公共前缀的下一位中 $0$ 位和 $1$ 位的分界点，对于每一个节点，他所在的区间 $[i,j]$ 对应的 Morton Code $[c_i,\cdots,c_j]$ 其公共前缀的长度一定是大于区间 $[0,i]$ 和区间 $[j,N_t-1]$ 的，毕竟按照我们的划分方式，$c_i$ 与 $c_{i+1}$ 的公共前缀长于 $c_i$ 和 $c_{i-1}$ 的公共前缀长度，单就区间 $[i-1, i]$ 的对应的公共前缀长度已经小于 $[i,j]$ 了。如果我们能够利用这个性质，独立地确定每一个区间的两个端点，再在这两个端点之间找到一个分界点 $\gamma$ 划分出这个节点的两个子节点，就能为每一个中间节点分配一个 thread，并行地算出整个树状结构。

我们首先设定，节点 $n_i$ 所对应的区间其中一个端点一定是 $i$，这样我们只需要找到另一个端点是什么，对于 $i$ 邻近的两个点 $i-1$ 和 $i+1$，我们知道，对应的 Morton Code $c_{i-1}$ 和 $c_{i+1}$ 中与当前端点的 Morton Code 公共前缀长的那一个一定是属于当前区间的，毕竟每一个区间内部公共前缀的长度一定大于区间的外部。这样我们至少知道了另一个端点应该在哪一个方向上。接下来，我们沿着这个方向一直计算公共前缀的长度，直到公共前缀的长度变得不大于区间外侧的公共前缀长度了，也就是对于我们搜索的点 $\phi$，$c_\phi$ 和 $c_i$ 的公共前缀长度不大于 $c_i$ 和 $c_{i-d}$ 的公共前缀长度，其中 $d$ 为我们搜索的方向（+1或者-1）。此时我们取 $\phi-d$ 作为这个区间的另外一个端点。文章的作者给出了下面的伪代码，我们在搜索的时候当然可以用二分搜索，但是目的是一样的：

<center>
<img src="/images/lbvh/code.png" style="width:500px; height:auto;"/>
</center>

在我们找到了当前节点 $n_i$ 对应区间的两个端点 $i$ 和 $j$，之后，我们就该找这个区间的分界点 $\gamma$ 了，很自然的，我们按照上面的思路找到分界点 $\gamma$ 之后，可以让左子节点为 $n_{\gamma}$ ，右子节点为 $n_{\gamma+1}$。毕竟这样分配左右子节点，对于两个子节点，也是满足我们想要的“区间内部公共前缀长度大于区间外部”。

这个算法非常巧妙，我们可以这样理解它的正确性，我们定义公共前缀长度函数为 $\text{pfl}$。我们不难知道，对于当前节点 $[i,j]$ 的两个子区间 $[i,\gamma]$ 和 $[\gamma+1,j]$ ，其中 $[i,\gamma]$ 在搜索的时候方向一定是-1，毕竟他从 $\gamma$ 出发， $\text{pfl}(c_\gamma, c_{\gamma-1}) > \text{pfl}(c_\gamma,c_{\gamma+1})$ ，它搜索的终止点一定是 $i$，因为 $\text{pfl}(c_k,c_\gamma)>\text{pfl}(c_{i-1},c_\gamma),\forall k\in [i,\gamma]$ 成立，所以它独立搜索出来的区间和它父亲节点划分取出来的区间是一样的。同理，每一个节点自己独立划分出来的区间和他的父亲节点独立帮它划分出来的区间是完全一样的，父亲节点可以被每个子节点完整的分成了两个部分，并行下是正确的。

在讨论 Morton Code 时，我们提到了如果两个 Morton Code 相等，我们需要特殊处理，毕竟如果我们在第一步找区间方向时如果 $c_{i-1},c_i,c_{i+1}$ 全都相等就不好了，作者指出的办法是，如果两个 Morton Code 是一样的，我们转而比较他们的下标，也就是如果 $c_{i-1}=c_i$ ，我们还要计算 $i-1$ 和 $i$ 的公共前缀长度，把这个额外的长度加上，这样得到的结果也是正确的。

至此我们已经明确了这个算法的理论部分，接下来该开始写代码复现了。

## 从 Vertex Buffer 和 Index Buffer 计算 Morton Code

由于我用到的 api 是古老的 OpenGL，我下面的代码也都是按照方便 OpenGL 的写法组织的。我们在 OpenGL 中定义一个 mesh 时，我们需要指定一个 `Vertex Array Object (VAO)`，每一个 `VAO` 需要绑定一个 `Vertex Buffer Object (VBO)` 和一个可选的 `Element Array Buffer Object (EBO)`。其中 `VBO` 保存了每个顶点的数据，比如说顶点的位置，法向，颜色，纹理坐标。`EBO` 则是保存了每个三角形索引的三个顶点下标，他们的长度分别是 $N_{vertices}$ 和 $3\times N_{triangles}$。

首先我们注意到，计算 Morton Code 需要所有点的三个维度都是在 $[0,1]$ 之间的。我们因此可以首先计算整个模型的 Bounding Box，然后把每个三角形重心的坐标缩放到这个 Bounding Box 的范围里面，这样就都是 $[0,1]$ 之间了。

计算所有顶点的 Bounding Box 是一个 Reduce 操作，不过我采用了 Scan 的方法，通过第一个 Pass 把每个三角形的 Bounding Box 都先算出来，接着计算出每一个 Work Group 里面的 Bounding Box，把这些 Bounding Box 写到一个大小为 Work Group 数量的数组里，然后递归操作，直到得到整体的 Bounding Box：

```glsl
#version 430
#define WORK_GROUP_SIZE 256
layout(local_size_x = WORK_GROUP_SIZE) in;

struct AABB {
  vec3 boxMin;
  float padding1;
  vec3 boxMax;
  float padding2;
};
void mergeAABB(inout AABB a, AABB b) {
  a.boxMin = min(a.boxMin, b.boxMin);
  a.boxMax = max(a.boxMax, b.boxMax);
}
layout(std430, binding = 0) buffer WorldVertexBuffer {
  AABB gGlobalBoundingBoxes[];
};
layout(std430, binding = 1) buffer WorkGroupBuffer {
  AABB gGroupBoundingBoxes[];
};

uniform int gActualSize;
shared AABB sharedData[WORK_GROUP_SIZE];

// exclusive scan
void LocalScan(uint lid) {
  //up sweep
  uint d = 0;
  uint i = 0;
  uint offset = 1;
  uint totalNum = WORK_GROUP_SIZE;
  for (d = totalNum>>1; d > 0; d >>= 1) {
    barrier();
    if (lid < d) {
      uint ai = offset * (2 * lid + 1) - 1;
      uint bi = offset * (2 * lid + 2) - 1;
      
      mergeAABB(sharedData[bi], sharedData[ai]);
    }
    offset *= 2;
  }

  //clear the last element
  if (lid == 0) {
    sharedData[totalNum-1].boxMin = vec3(1e30);
    sharedData[totalNum-1].boxMax = vec3(-1e30);
  }
  barrier();

  //Down-sweep
  for (d = 1; d < totalNum; d *= 2) {
    offset >>= 1;
    barrier();

    if (lid < d) {
      uint ai = offset * (2 * lid + 1) - 1;
      uint bi = offset * (2 * lid + 2) - 1;

      AABB tmp = sharedData[ai];
      sharedData[ai] = sharedData[bi];
      mergeAABB(sharedData[bi], tmp);
    }
  }
  barrier();
}

void main() {
  uint gid = gl_GlobalInvocationID.x;
  uint lid = gl_LocalInvocationID.x;
  uint groupId = gl_WorkGroupID.x;

  if (gid < gActualSize) {
    sharedData[lid] = gGlobalBoundingBoxes[gid];
  } else {
    sharedData[lid].boxMin = vec3(1e30);
    sharedData[lid].boxMax = vec3(-1e30);
  }
  barrier();

  LocalScan(lid);

  if (lid == WORK_GROUP_SIZE - 1)
    gGroupBoundingBoxes[groupId] = sharedData[lid];
}
```

接下来我们对于每个三角形创建一个 thread，传入整个模型的 Bounding Box 缩放他，计算这个三角形重心的 Morton Code：

```glsl
#version 430
#define WORK_GROUP_SIZE 256
layout(local_size_x = WORK_GROUP_SIZE) in;

struct _packed_vertex {
  vec4 position;
  vec4 normal;
};
layout(std430, binding = 0) buffer SceneVertexBuffer {
  _packed_vertex gSceneVertexBuffer[];
};
layout(std430, binding = 1) buffer SceneIndexBuffer {
  uint gSceneIndexBuffer[];
};
layout(std430, binding = 2) buffer MortonCodeBuffer {
  uint gMortonCode[];
};
layout(std430, binding = 3) buffer PrimitiveIndexBuffer {
  uint gPrimIndex[];
};
uniform int gNumTriangles;
uniform vec3 gBoundingBoxMin;
uniform vec3 gBoundingBoxMax;
uniform int gVertexOffset;
uniform int gIndexOffset;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
uint expandBits(uint v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton3(float x, float y, float z) {
  x = min(max(x * 1024.0, 0.0), 1023.0);
  y = min(max(y * 1024.0, 0.0), 1023.0);
  z = min(max(z * 1024.0, 0.0), 1023.0);
  uint xx = expandBits(uint(x));
  uint yy = expandBits(uint(y));
  uint zz = expandBits(uint(z));
  return xx * 4 + yy * 2 + zz;
}

void main() {
  uint gid = gl_GlobalInvocationID.x;
  uint lid = gl_LocalInvocationID.x;
  uint groupId = gl_WorkGroupID.x;

  if (gid >= gNumTriangles) return;

  vec3 v0 = gSceneVertexBuffer[gSceneIndexBuffer[3 * gid + 0 + gIndexOffset]].position.xyz;
  vec3 v1 = gSceneVertexBuffer[gSceneIndexBuffer[3 * gid + 1 + gIndexOffset]].position.xyz;
  vec3 v2 = gSceneVertexBuffer[gSceneIndexBuffer[3 * gid + 2 + gIndexOffset]].position.xyz;

  vec3 barycenter = (v0 + v1 + v2) / 3.0;
  barycenter -= gBoundingBoxMin;
  // scale each axis to [0.0, 1.0]
  vec3 extent = gBoundingBoxMax - gBoundingBoxMin;
  barycenter = vec3(barycenter.x / extent.x, barycenter.y / extent.y, barycenter.z / extent.z);

  gMortonCode[gid] = morton3(barycenter.x, barycenter.y, barycenter.z);
  gPrimIndex[gid] = gid;
}
```

如果细心的话，可以发现我实际上是用一个 `offset` 从 `gSceneVertexBuffer` 和 `gSceneIndexBuffer` 里面取的顶点位置，由于我的引擎还需要处理 Blend Shape 和 Skinned Mesh，我预先将这些该变换顶点位置的操作的执行了，保留一份额外的，形变过后的，整个场景中所有 mesh 的 Vertex Buffer 和 Index Buffer，这样做对内存不太友好，但是我目前没有学到更合适的办法。

## GPU 上 Radix Sort 排序算法的实现

有了所有三角形重心的 Morton Code 之后，我们需要对他们排序，由于目前我们的数据都在显存里面，如果把数据读回 CPU 排序，再送回去对于带宽非常不友好，特别是当我们有很多很多三角形时，我们需要找到一种快速在 GPU 上排序的算法。

同样的 [nvidia 博客](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/) 提到了他们采用了 Radix Sort （基数排序）的 GPU 实现。经过一定的调研，很可惜我并没有找到用 OpenGL 实现的 Radix Sort （严格来说有一个 [gl-radix-sort](https://github.com/loryruta/gl-radix-sort)，可惜用到的版本是 4.6，而且用到了一个拓展，只在我的 RTX 2050 laptop 上能跑起来，并不适用于 intel 的集显，虽然用集显跑排序很奇怪，但我希望我的实现有很强的跨平台性），于是我打算参照资料自己写一个基于 OpenGL 4.3，不用到 OpenGL 拓展，有良好跨平台性同时保证效率的 GPU 排序算法。

Radix Sort 在 CPU 上的实现很容易理解，输入一个未排序的整数数组，我们从十进制的最低位出发，按照所有数在这个位置上的数值 $R\in[0,9]$ 分成 10 个桶，把所有的数按照对应位的数值放到对应的桶里面，例如 $19$ 就应该先放到编号为 $9$ 的桶里面，然后我们遍历所有的桶，完成个位的排序，接着对十位做同样的操作，直到对所有数位操作完，整个序列就排好序了。算法的正确性在 [oi-wiki](https://oi-wiki.org/basic/radix-sort/#%E6%AD%A3%E7%A1%AE%E6%80%A7) 对应章节有非常清晰的介绍。

我们在 GPU 上实现算法时，由于需要把所有的 threads 分成多个 work group，每个 work group 之间可以理解成并行，除了执行完当前的 kernal 和调用 `atomicXXX` 函数计数在 OpenGL 中没有简单的办法确保 work group 之间数据的同步，但是我们能够很方便地通过 `barrier` 内置函数确保一个 work group 内部所有 threads 的同步，也能通过 `shared` 关键字为每一个 work group 申请一块组内共享的专有内存。

关于 Radix Sort 在 GPU 上的实现，我主要参考[这篇文章](https://www.researchgate.net/publication/267453578_Fast_Data_Parallel_Radix_Sort_Implementation_in_DirectX_11_Compute_Shader_to_Accelerate_Ray_Tracing_Algorithms)以及[这篇博客](https://www.cnblogs.com/Ligo-Z/p/16254628.html)。虽然文章和博客采用的语言都是 hlsl，但是他们的算法都很适合迁移到别的语言，对于算法的解释也非常到位。

按照 Radix Sort 在 CPU 上的步骤，我们能够理出其在 GPU 上的步骤大概应该是什么样的，假设我们的输入数据是 4 字节的无符号整数，我们每次对这 32 个 bit 中的 k 个 bit 排序，排序完之后对接下来的 k 个 bit 继续排序。比较朴素的做法是让 $k=1$，也就是一次只排序一个 bit，由于这样每个 bit 只有两种可能的数，我们可以很方便的创建两个桶，然后确保 0 数位的排在 1 数位之前，在把每个 work group 内部的数排好序之后通过 merge sort 的办法合并成一个大的有序数组。

但是我们这里关注让 $k>1$ 的算法，我的实现中让 $k=4$ 取得了比较好的效果，但是为了方便解释，我们让 $k=2$，首先从[这篇文章](https://www.sci.utah.edu/~csilva/papers/cgf.pdf)中借一张图：

<center>
<img src="/images/lbvh/radix_sort.png" style="width:80%; height:auto;"/>
</center>

上图中右侧部分的一大堆数字中，第一行是我们算法的输入，由于我们的 $k=2$，排序的基数可能有 0,1,2,3 四种情况，我们按照每个 work group 大小为 4，将整个数据分成了 5 个 work groups。

对于每个 work group，我们找到每个基数是当前 work group 中的第几号基数，例如上图左侧方框内第一列中红色的 0，他是这个 work group 中的第一个 0，所以号码是 0；第二列中有两个 0，第一个 0 的号码是 0，第二个 0 的号码是 1，如果有第三个 0 的话，它的号码就是 2，不管这些 0 在 work group 所有基数中序号是几。这个结果被称为 `Local Prefix Sum`，其实就是对每个基数的 Mask 做 Scan （前缀和），在 GPU 上比较容易实现。

有了每个 work group 的 `Local Prefix Sum`，我们记录一个长度为 work group 大小和可能的基数种类数量乘积的数组，这里该数组长度为 $5(组的数量)\times 4(基数种类的数量)$。这个数组记录了每一个 work group 中，有多少个对应的基数。上图右侧第四行 `Block Sum` 即该数组，这个数组头两个红色的 1 和 2 分别表示第一个 work group 中有 1 个基数 0，第二个 work group 中有 2 个基数 0。类似的，这个数组中绿色的元素代表了第四个 work group 中有 2 个基数 2。不难意识到，我们求 `Local Prefix Sum` 同时也可以把这个数组算出。

接着我们要对这整个 `Block Sum` 求前缀和，得到数组 `Prefix Block Sum`。有了这些信息后，如果我想知道输入中第 $i$ 个 work group 的第 $j$ 个基数 2 排序后的位置，我只需要找到基数二对应的 `Prefix Block Sum`，取出其中第 $i$ 个元素，并与我们输入中这个元素对应 `Local Prefix Sum` 的数值相加，就可以得到这个输入元素最终应该在输出的哪一个位置上。我们可以按照图中绿色数字过一遍这个流程。

这个算法的正确性是容易理解的，我们取输入元素对应 `Prefix Block Sum` 中的数，含义是在最终的输出中，相对于当前的 work group 有多少个小于当前基数的元素，在这个基础上加上这个输入元素是自己 work group 中的第几个该元素，就能够得到他的最终位置。

我们首先来看求 `Local Prefix Sum` 和 `Block Sum` 的代码：

```glsl
#version 430
#define RADIX_BITS 4
#define THREAD_NUM 256
layout(local_size_x = THREAD_NUM) in;

layout(std430, binding = 0) buffer KeysInput {
  uint gKeysInput[];
};
// shape: (1 << RADIX_BITS) * numWorkGroup
layout(std430, binding = 1) buffer RadixCountingBuffer {
  uint gRadixCountingBuffer[];
};
// shape: THREAD_NUM * numWorkGroup
layout(std430, binding = 2) buffer RadixPrefixBuffer {
  uint gRadixPrefixBuffer[];
};

uniform int gActualSize;
uniform int gNumWorkGroups;
uniform int gCurrentIteration;
shared uint localPrefix[THREAD_NUM];

// exclusive scan
void LocalScan(uint lid) {
  //up sweep
  uint d = 0;
  uint i = 0;
  uint offset = 1;
  uint totalNum = THREAD_NUM;
  for (d = totalNum>>1; d > 0; d >>= 1) {
    barrier();
    if (lid < d) {
      uint ai = offset * (2 * lid + 1) - 1;
      uint bi = offset * (2 * lid + 2) - 1;

      localPrefix[bi] += localPrefix[ai];
    }
    offset *= 2;
  }

  //clear the last element
  if (lid == 0) {
    localPrefix[totalNum-1] = 0;
  }
  barrier();

  //Down-sweep
  for (d = 1; d < totalNum; d *= 2) {
    offset >>= 1;
    barrier();

    if (lid < d) {
      uint ai = offset * (2 * lid + 1) - 1;
      uint bi = offset * (2 * lid + 2) - 1;

      uint tmp = localPrefix[ai];
      localPrefix[ai] = localPrefix[bi];
      localPrefix[bi] += tmp;
    }
  }
  barrier();
}

uint getDigit(uint gid) {
  uint data = gid >= gActualSize ? 4294967295u : gKeysInput[gid];
  return (data << ((32 - RADIX_BITS) - RADIX_BITS * gCurrentIteration)) >> (32 - RADIX_BITS);
}

void RadixSortCount(uint gid, uint lid, uint groupId) {
  uint digit = getDigit(gid);

  uint radixCategory = 1 << RADIX_BITS;
  for (uint r = 0; r < radixCategory; ++r) {
    //load to share memory
    localPrefix[lid] = (digit == r ? 1 : 0);
    barrier();

    // prefix sum according to r in this blocks
    LocalScan(lid);

    // ouput the total sum to global counter
    if (lid == THREAD_NUM - 1) {
      uint counterIndex = r * gNumWorkGroups + groupId;
      uint counter = localPrefix[lid];
      if (digit == r)
        counter++;
      gRadixCountingBuffer[counterIndex] = counter;
    }

    // output prefix sum according to r
    if(digit == r) {
      gRadixPrefixBuffer[gid] = localPrefix[lid];
    }

    barrier();
  }
}

void main() {
  uint gid = gl_GlobalInvocationID.x;
  uint lid = gl_LocalInvocationID.x;
  uint groupId = gl_WorkGroupID.x;

  RadixSortCount(gid, lid, groupId);
}
```

通过这一个 Pass 我们能算出 `Local Prefix Sum` 和 `Block Sum`。接着我们要对整个 `Block Sum` 求前缀和得到 `Prefix Block Sum`，这里我用到一个递归执行 local scan 的办法，大致代码如下：

```cpp
void prefixSumInternal(Buffer &input, Buffer &output, unsigned int size) {
  int workGroupNum = (size + m_workGroupSize - 1) / m_workGroupSize;
  Buffer workGroupChunkSum;
  workGroupScan.Use();
  workGroupChunkSum.SetDataSSBO(sizeof(unsigned int) * workGroupNum);
  workGroupScan.SetInt("actualSize", size);
  workGroupScan.BindBuffer(input, 0).BindBuffer(output, 1).BindBuffer(workGroupChunkSum, 2);
  workGroupScan.Dispatch(workGroupNum, 1, 1);
  workGroupScan.Barrier();
  if (size > m_workGroupSize) {
    // recursion
    Buffer workGroupChunkPreixSum;
    workGroupChunkPreixSum.SetDataSSBO(sizeof(unsigned int) * workGroupNum);
    prefixSumInternal(workGroupChunkSum, workGroupChunkPreixSum, workGroupNum);
    globalPrefixSum.Use();
    globalPrefixSum.SetInt("actualSize", size);
    globalPrefixSum.BindBuffer(output, 0).BindBuffer(workGroupChunkPreixSum, 1);
    globalPrefixSum.Dispatch(workGroupNum, 1, 1);
    globalPrefixSum.Barrier();
  }
}
```

大致的思路是，我们通过将整个数组分成多个 work group，每个 work group 内部求前缀和以及总和，将总和写到一个全局的数组里面，然后递归的对这个总和求前缀和，得到总和的前缀和之后，将总和的对应元素加到每个对应 work group 的元素上。递归的终止条件是，当前输入的大小小于一个设定的 work group 内部线程数量大小。虽然这样做效率可能不是很好，但是写起来很方便。

有了 `Prefix Block Sum` 之后，我们只需要按照之前的描述就能算出每个元素对应排序之后的位置了，这部分的核心代码如下：

```glsl
uint gid = gl_GlobalInvocationID.x;
uint lid = gl_LocalInvocationID.x;
uint groupId = gl_WorkGroupID.x;
if (gid >= gActualSize) return;

// get current radix 
uint radix = getRadix(gid, gNumInteration);
// global dispatch
uint counterIndex = radix * gNumWorkGroups + groupId;
uint globalPos = gRadixPrefixBuffer[gid] + gRadixCountingPrefixBuffer[counterIndex];
gKeysOutput[globalPos] = gKeysInput[gid];
gValsOutput[globalPos] = gValsInput[gid];
```

可以看到，我其实是对一个键值对排序，毕竟我们需要记录每一个 Morton Code 对应的三角形序号是什么。

经过了这些实现，我们可以看看这个排序的性能如何：

> 测试平台：
> 
> CPU：12th Gen Intel(R) Core(TM) i5-12500H  2.50 GHz
> 
> GPU：NVIDIA GeForce RTX 2050 laptop
> 
> 显卡驱动：561.17


> 数据量，时间
> 
> 1024: 0.69862 ms
> 
> 16384: 1.5495 ms
> 
> 65536: 3.1757 ms
> 
> 131072: 6.36876 ms

从数据上看，是大致符合 Radix Sort 线性的复杂度的，虽然不算特别快，但是用于实时的应用 （60fps+）应该已经够了。

## 通过排序的 Morton Code 构建 Binary Radix Tree

现在我们有了排好序的每个三角形重心的 Morton Code 和每一个 Morton Code 对应的三角形索引。我们可以按照本文第一部分的算法，构建这颗 Binary Radix Tree，核心代码如下所示：

```glsl
void handleInternalNode(int i) {
  // find the direction for this range
  int d = isign(delta(i, i + 1) - delta(i, i - 1));
  // find the upper bound for this range
  int sigmaMin = delta(i, i - d);
  int lMax = 2;
  while (delta(i, i + lMax * d) > sigmaMin)
    lMax *= 2;
  int l = 0;
  for (int t = lMax/2; t > 0; t /= 2)
    if (delta(i, i + (l + t) * d) > sigmaMin)
      l += t;
  int j = i + l * d;

  int lower = min(i, j), upper = max(i, j);
  // split
  int deltaNode = delta(i, j);
  int gamma = 0, p = lower, q = upper, mid;
  while (p <= q) {
    mid = p + (q-p)/2;
    if (delta(lower, mid) > deltaNode)
      p = mid + 1;
    else {
      gamma = mid-1;
      q = mid - 1;
    }
  }

  BVHNode node;
  node.firstPrim = lower;
  node.primCount = l + 1;
  if (lower == gamma) {
    node.left = gamma + gNumInternalNodes;
  } else {
    node.left = gamma;
  }
  if (upper == gamma + 1) {
    node.right = gamma + 1 + gNumInternalNodes;
  } else {
    node.right = gamma + 1;
  }

  gNodes[i + gBVHNodeOffset] = node;
  gInfo[node.left].parent = i;
  gInfo[node.right].parent = i;
}
```

上面代码中的 `delta` 函数计算两个 Morton Code 公共前缀的长度，如果这两个 Morton Code 相同，我们继续比较他们索引的公共前缀长度。为了减少构建出来的最终节点占据显存的大小，我对每个节点的定义如下：

```cpp
struct BVHNode {
  int left, right, fristPrim, primCount;
  Math::Vector4 aabbMin;
  Math::Vector4 aabbMax;
};
```

每个节点并不保存到父亲节点的索引，毕竟我们使用这个数据结构的时候并不需要从下往上遍历，构建过程中父亲节点和其他信息存储在 `gInfo` 里面。同时，我们还希望保留每一个节点对应的区间 $[i,j]$ 是什么，如果我们能够提前知道当前遍历到的节点对应到多少个三角形，我们可以设一个参数，决定是继续遍历 BVH 还是直接和该节点的所有三角形求交，当我们的树很深的时候，这一点可以带来比较大的性能提升。这要求我们通过排好序的 Morton Code 对应的三角形索引修改 Index Buffer 里面索引的顺序，这里不做赘述。

## 通过 Binary Radix Tree，自底向上计算每个节点的 Bounding Box

我们构建 BVH 现在还剩最后一步，我们需要利用构建好的树状结构，计算出每一个节点的 Bounding Box。在 LBVH 原文中作者提到了他们的构建方法。我们对于每个叶子节点分配一个 thread，对于每一个节点记录一个全局访问次数。

从叶子节点出发一直向上，如果我们当前遍历到的节点访问次数为 0，意味着另一个子节点还没有算好它的 Bounding Box，我们通过 `atomicAdd` 把访问次数加一，然后终止执行；如果当前遍历到的节点访问次数为 1，意味着当前节点的两个子节点 Bounding Box 都已经计算完成，我们合并当前节点的两个子节点的 Bounding Box，继续向上访问，核心代码如下：

```glsl
uint gid = gl_GlobalInvocationID.x;
if (gid >= gNumTriangles) return;
uint leafNodeOffset = gNumTriangles - 1;
int nodeIndex = gInfo[leafNodeOffset + gid].parent;

while (true) {
  int visits = atomicAdd(gInfo[nodeIndex].visits, 1);
  if (visits < 1) {
    // terminate current thread
    return;
  }
  // merge bounding boxes, keep upward
  BVHNode node = gNodes[nodeIndex + gBVHNodeOffset];
  BVHNode leftNode = gNodes[node.left + gBVHNodeOffset];
  BVHNode rightNode = gNodes[node.right + gBVHNodeOffset];
  node.aabbMin = min(leftNode.aabbMin, rightNode.aabbMin);
  node.aabbMax = max(leftNode.aabbMax, rightNode.aabbMax);
  gNodes[nodeIndex + gBVHNodeOffset] = node;

  // terminate at root node
  if (nodeIndex == 0)
    return;

  nodeIndex = gInfo[nodeIndex].parent;
}
```

这种构建 Bounding Box 的方法不需要 threads 之间有直接的同步，通过一个全局的计数器实现自下向上的访问，理论上有较高的效率。不过我们有几点需要注意：

1. glsl 中必须给一个 Buffer 添加关键字 `coherent` 才能确保对它的修改是实时全局可见的，否则别的节点不一定能及时接收到修改
2. 对于普通的全局 Buffer（SSBO），在代码中的修改并不会直接同步到全局内存中，我们不可以在上面的代码中计算每个叶子节点的 Bounding Box，毕竟这样别的 thread 不一定能同步到当前节点的 Bounding Box，导致错误结果，需要单独开一个 Pass 计算好这些三角形的 Bounding Box

其实我们容易观察到，这个重新构建的算法也很适合用来做 BVH 的 refit，如果我们构建好了一个质量比较高的 BVH，比如采用了 SAH，我们不希望从头构建整个数据结构，可以考虑保留树状结构不变，更新所有节点的 Bounding Box，但是这样得到的 BVH 没有质量的保证，可以作为后续改进的方向。

在这样一大堆的步骤过后，我们终于得到了 LBVH，这种算法就我的调研看，是目前构建速度最快的算法。假如我们有一些会发生形变的模型，例如 Blend Shape，Skinned Mesh，FEM，我们可以在每一帧重新构建出一个新的 BVH，确保光线求交，碰撞检测有足够的效率保证。

## 每一帧重新构建加速结构，遍历该结构用于简单的阴影光线追踪

最后，我们可以把场景中所有构建好的 BVH 拼在一起，在这些 BVH 的基础上构建一个更高层级的数据结构 TLAS，在[这篇文章](https://jacco.ompf2.com/2022/05/13/how-to-build-a-bvh-part-6-all-together-now/)中有比较详细的描述。我们只需要知道，构建的时候，我们可以直接通过一个 offset 写入全局的 BVH 数组，并且用一个额外的数组保存场景中所有 mesh 的这个 offset，做光线求加时遍历这个保存了 offset 的数组，就可以实现整个场景的光线求交，这个保留了一个 mesh 自己 offset 的结构我定义如下：

```cpp
struct SceneTLAS {
  unsigned int numNodes, nodeOffset;
  unsigned int vertexOffset, indexOffset;
};
```

光线与场景求交的代码有非常多可以参考，我在这里放我的实现：

```glsl
uniform int gTerminatePrimCount;
void RayIntersectScene(inout Ray ray) {
  // test with tlas
  for (int tlasIndex = 0; tlasIndex < gSceneTLAS.length(); tlasIndex++) {
    SceneTLAS tlas = gSceneTLAS[tlasIndex];
    BVHNode node = gNodes[tlas.nodeOffset];
    if (TestRayAABB(ray, node.aabbMin.xyz, node.aabbMax.xyz) == 1e30)
      continue;
    int stack[256], stackPtr = 0, depthStack[256], depth = 0;
    while (true) {
      if ((node.primCount != 0 && node.primCount <= gTerminatePrimCount) || (node.left == -1 && node.right == -1)) {
        // leaf node
        bool hitAny = false;
        for (int primOffset = 0; primOffset < node.primCount; primOffset++) {
          int indexBase = 3 * (primOffset + node.firstPrim) + int(tlas.indexOffset);
          vec3 v0 = gSceneVertexBuffer[gSceneIndexBuffer[indexBase + 0]].position.xyz;
          vec3 v1 = gSceneVertexBuffer[gSceneIndexBuffer[indexBase + 1]].position.xyz;
          vec3 v2 = gSceneVertexBuffer[gSceneIndexBuffer[indexBase + 2]].position.xyz;
          if (IntersectRayTriangle(ray, v0, v1, v2))
            hitAny = true;
        }
        if (stackPtr == 0 || hitAny)
          break;
        node = gNodes[stack[--stackPtr] + tlas.nodeOffset];
        depth = depthStack[stackPtr];
        continue;
      }
      // intermidiate node
      int c0 = node.left;
      int c1 = node.right;
      BVHNode c0Node = gNodes[c0 + tlas.nodeOffset];
      BVHNode c1Node = gNodes[c1 + tlas.nodeOffset];
      float dist0 = TestRayAABB(ray, c0Node.aabbMin.xyz, c0Node.aabbMax.xyz);
      float dist1 = TestRayAABB(ray, c1Node.aabbMin.xyz, c1Node.aabbMax.xyz);
      if (dist0 > dist1) {
        float tmp0 = dist0;
        dist0 = dist1;
        dist1 = tmp0;
        int tmp1 = c0;
        c0 = c1;
        c1 = tmp1;
        BVHNode tmp2 = c0Node;
        c0Node = c1Node;
        c1Node = tmp2;
      }
      if (dist0 == 1e30) {
        if (stackPtr == 0)
          break;
        node = gNodes[stack[--stackPtr] + tlas.nodeOffset];
        depth = depthStack[stackPtr];
      } else {
        node = c0Node;
        depth++;
        if (dist1 != 1e30) {
          depthStack[stackPtr] = depth;
          stack[stackPtr++] = c1;
        }
      }
    }
  }
}
```

其中 `TestRayAABB` 和 `IntersectRayTriangle` 分别是检测光线和 AABB 相交以及光线和三角形求交，光线和三角形求交的算法可以参考 [Realtime Rendering 4th 里面的实现](https://www.realtimerendering.com/#isect)，这里放上我的实现：

```glsl
uniform float gRayMinDist;
bool IntersectRayTriangle(inout Ray ray, vec3 v0, vec3 v1, vec3 v2) {
  vec3 e1 = v1 - v0;
  vec3 e2 = v2 - v0;
  vec3 q = cross(ray.d, e2);
  float a = dot(e1, q);
  if (abs(a) < 1e-8)
    return false;
  float f = 1 / a;
  vec3 s = ray.o - v0;
  float u = f * (dot(s, q));
  if (u < 0.0f)
    return false;
  vec3 r = cross(s, e1);
  float v = f * (dot(ray.d, r));
  if (v < 0.0 || u + v > 1.0)
    return false;
  float t = f * (dot(e2, r));
  if (t > gRayMinDist) {
    ray.t = ray.t < t ? ray.t : t;
    return true;
  }
  return false;
}
```

最后我们可以提一提阴影光线的追踪，我们只需要记录从屏幕空间中的每一个像素，自摄像头射出一条光线与场景求交，自这个交点向每一个光源发射一条光线，并检测这条管线是否和场景里面的任何物体相交即可。我的实现中复用了 defered rendering 得到的位置贴图 `posTex` 和法向贴图 `normalTex`，可以直接从到摄像头的最近点向所有光源发射阴影光线，实现大致如下：

```glsl
ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
if (pixelCoord.x >= viewport.x || pixelCoord.y >= viewport.y) return;
vec2 texCoords = (vec2(pixelCoord) + vec2(0.5)) / viewport;
vec3 worldNormal = normalize(2 * (texture(normalTex, texCoords).xyz - vec3(0.5)));
vec3 worldPos = texture(posTex, texCoords).xyz;

float shadow = 0.0;
int lightsNum = lights.length();
for (int l = 0; l < lightsNum; l++) {
  vec3 lightDir;
  float lightDist;
  if (lights[l].idata[0] == 0) { // dir light
    lightDir = -normalize(lights[l].direction.xyz);
    lightDist = dot(lightDir, lights[l].position.xyz-worldPos);
  } else if (lights[l].idata[0] == 1) { // point light
    lightDir = normalize(lights[l].position.xyz-worldPos);
    lightDist = length(lights[l].position.xyz-worldPos);
  } else continue;
  if (dot(lightDir, worldNormal) < 0.0) {
    // in shadow, do nothing
    continue;
  }
  // trace the scene
  Ray ray;
  ray.o = worldPos;
  ray.d = lightDir;
  ray.t = 1e30;
  RayIntersectScene(ray);
  if (ray.t > lightDist)
    shadow += 1.0;
}
imageStore(rtTex0, pixelCoord, vec4(clamp(shadow, 0.0, 1.0), 0.0, 0.0, 1.0));
```

最后展示一下添加了光追阴影的效果：

<center>
<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=113651679826135&bvid=BV1xVBEYhEmb&cid=27350535562&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" style="width:900px; height:500px;"></iframe>
</center>

## 一个简易的总结

至此我们完成了 LBVH 的复现工作，并且应用它作为加速结构做了简单的阴影光线追踪，取得了比较合适的帧数。但是上面的代码实现还有优化的空间，我对于部分算法的理解也可能有误，如果您发现问题，或者有什么疑问，请通过邮箱 1012872688@qq.com 及时联系我。
