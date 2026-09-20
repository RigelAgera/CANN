# BatchMatmulMaxSum 算子赛题

> 根据提供的 8 张赛题图片整理，保留原文章节编号。图片中未提供第六节内容，故本文从第五节直接接第七节。公式、代码与示例按原图转录。

## 一、赛题背景

在神经信息检索和 RAG 重排场景中，ColBERT 使用 Late Interaction 计算 query 与 document 的相关性：先计算 token 级点积相似度，再对每个 query token 执行 MaxSim，最后求和得到 query-document 分数。该流程由 BatchMatMul、ReduceMax 和 ReduceSum 组成，融合后可减少 Kernel Launch 与中间数据搬运。

本题要求基于 PyTorch 中 `torch.bmm`、`torch.amax` 和 `torch.sum` 的核心业务逻辑，采用 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款高性能的 BatchMatmulMaxSum 算子。

## 二、算子功能描述

实现的 BatchMatmulMaxSum 算子需完成以下核心计算：

1. **批量矩阵乘**：对 B 组一一对应的矩阵执行 BatchMatMul，计算 query token 与 document token 的点积相似度。
2. **MaxSim 归约**：沿 document-token 维 N 取最大值，得到每个 query token 的最高相似度。
3. **求和归约**：沿 query-token 维 M 求和，输出每个 query-document pair 的相关性分数。

输入 embedding 的 L2 归一化由上游网络完成，本算子不执行 L2 归一化，也不接收 padding、有效长度或 mask 输入。

算子的核心难点在于：在支持四种输入存储布局组合、非对齐尾块、较小 Batch 和最大可达 8192 的 M/N/K 维度时，协调 Cube 矩阵乘与 Vector 归约流水，降低相似度计算与两级归约之间的数据搬运开销。

## 三、核心定义与约束

### 3.1 参考算子

PyTorch 原生算子：`torch.bmm`、`torch.amax`、`torch.sum`。

等价 Python 实现：

```python
import torch


def batch_matmul_max_sum(x1_logical, x2_logical):
    # x1_logical: [B, M, K]
    # x2_logical: [B, K, N]
    similarity = torch.bmm(
        x1_logical.to(torch.float32),
        x2_logical.to(torch.float32),
    )
    max_sim = torch.amax(similarity, dim=-1)
    return torch.sum(max_sim, dim=-1, dtype=torch.float32)
```

上述代码描述固定逻辑 shape 下的计算语义。`transposeX1` 和 `transposeX2` 只声明输入的 storage shape，不参与数学计算，也不表示算子需要额外执行 transpose 操作。算子行为、计算结果需与上述 PyTorch 组合实现的数学语义对齐。标准 golden 使用输入实际存储值在 FP64 精度下计算，最后转换为 FP32。

### 3.2 数学公式

参与 BatchMatMul 的逻辑矩阵固定为：

$$
X_1 \in \mathbb{R}^{B \times M \times K},
\qquad
X_2 \in \mathbb{R}^{B \times K \times N}
$$

**阶段一——批量矩阵乘：**

$$
A[b,m,n] = \sum_{k=0}^{K-1} X_1[b,m,k] \times X_2[b,k,n],
\qquad
A \in \mathbb{R}^{B \times M \times N}
$$

**阶段二——MaxSim 归约：**

$$
R[b,m] = \max_{0 \le n < N} A[b,m,n],
\qquad
R \in \mathbb{R}^{B \times M}
$$

**阶段三——求和归约：**

$$
y[b] = \sum_{m=0}^{M-1} R[b,m]
= \sum_{m=0}^{M-1} \left(\max_{0 \le n < N} A[b,m,n]\right),
\qquad
y \in \mathbb{R}^{B}
$$

第 b 个 x1 仅与第 b 个 x2 配对，不进行 batch broadcast 或跨 batch 笛卡尔积计算。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 逻辑 shape / storage shape | 支持数据类型 | 数据格式 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| INPUT（必选） | x1 | tensor | 逻辑 shape 固定为 $(B,M,K)$；storage shape 在 `transposeX1=false` 时为 $(B,M,K)$，`true` 时为 $(B,K,M)$ | FLOAT16、BFLOAT16 | ND | Query token embedding |
| INPUT（必选） | x2 | tensor | 逻辑 shape 固定为 $(B,K,N)$；storage shape 在 `transposeX2=false` 时为 $(B,K,N)$，`true` 时为 $(B,N,K)$ | FLOAT16、BFLOAT16 | ND | Document token embedding 的矩阵乘布局 |
| OUTPUT（输出） | y | tensor | $(B,)$ | FLOAT32 | ND | 每个 query-document pair 的相关性分数 |
| ATTR（可选） | transposeX1 | bool | - | BOOL | - | 仅声明 x1 的 storage shape，默认 false |
| ATTR（可选） | transposeX2 | bool | - | BOOL | - | 仅声明 x2 的 storage shape，默认 false |

### 3.4 关键输入约束

**维度取值范围（均为正整数）：**

- $1 \le B \le 64$。
- $1 \le M \le 8192$。
- $1 \le N \le 8192$。
- $32 \le K \le 8192$，且 K 为 8 的整数倍。
- 输入规模同时满足 $B \times M \times K \le 2^{26}$ 和 $B \times N \times K \le 2^{26}$。

**形状约束：**

- x1 和 x2 均为 3 维 Tensor。
- x1 的逻辑形状固定为 $(B,M,K)$，x2 的逻辑形状固定为 $(B,K,N)$；`transposeX1` 和 `transposeX2` 仅用于确定对应输入的 storage shape。
- 两个输入的 B 和 K 必须分别相等，不支持 batch broadcast。
- M、N 建议为 16 的整数倍；非对齐场景必须正确处理尾块。

**数据类型约束：**

- x1 和 x2 的数据类型必须相同。
- x1 和 x2 仅支持 FLOAT16、BFLOAT16。
- K 维点积、MaxSim 和 Sum Reduction 采用 FLOAT32 累加或具有等效精度的实现。

**其他约束：**

- 不支持空 Tensor，仅支持连续 ND Tensor。
- 输入不包含 NaN 或正负 Inf，输入元素允许为负数。
- M 和 N 均表示参与计算的实际 token 数，不提供 padding、有效长度或 mask 输入。
- 算子不得修改输入。

### 3.5 核心属性说明

- **transposeX1**（bool，默认 false）：用于说明 x1 的 storage shape。false 时 storage shape 为 $(B,M,K)$；true 时 storage shape 为 $(B,K,M)$。该属性不改变 x1 的逻辑 shape。
- **transposeX2**（bool，默认 false）：用于说明 x2 的 storage shape。false 时 storage shape 为 $(B,K,N)$；true 时 storage shape 为 $(B,N,K)$。该属性不改变 x2 的逻辑 shape。

两个属性仅描述输入数据的存储布局，不改变逻辑 BatchMatMul 与输出结果的定义。

### 3.6 输出严格要求

- **形状约束**：输出 y 的 shape 固定为 $(B,)$。
- **类型约束**：输出 y 的数据类型固定为 FLOAT32。
- **计算顺序**：必须按照 `BatchMatMul → Max(N) → Sum(M)` 计算，Max 与 Sum 不可交换。
- **有限性**：所有输出均不得包含 NaN 或正负 Inf。

### 3.7 特殊值处理规则

- **全负相似度**：MaxSim 的初始值必须为负无穷或归约行的首个元素，不得初始化为 0；当某行全部为负数时仍需返回其中最大负数。
- **大 K 累加**：K 维点积采用 FLOAT32 累加或具有等效精度的实现，避免低精度长归约造成明显误差。
- **无效输入**：题目用例不包含 NaN、正负 Inf、空 Tensor 或形状不匹配输入。

## 四、规则要求

1. **数值一致性规则**：给定相同输入与属性，算子多次执行结果应保持一致。
2. **一一配对规则**：x1 与 x2 的 batch 维必须相同，第 b 组 x1 只与第 b 组 x2 计算，不允许 batch broadcast。
3. **归约顺序规则**：计算顺序固定为先沿 N 维取最大值，再沿 M 维求和，不得改写为先求和再取最大值。
4. **存储布局规则**：`transposeX1` 和 `transposeX2` 仅声明对应输入的 storage shape，不表示执行转置操作；四种 storage shape 组合均需正确支持。
5. **性能要求**：在保证数值精度和正确性的前提下，充分利用 NPU 硬件特性：
   - 合理利用 Cube 与 Vector 计算单元，设计 BatchMatMul 和两级归约的协同流水。
   - 优化 GM、L1、L0 和 UB 之间的数据搬运与复用。
   - 沿 B 维和/或 M 维合理分配多核任务，兼顾较小 Batch、大 M/N/K 和尾块负载均衡。
   - 鼓励使用 Double Buffer 或 Ping-Pong 机制重叠计算与搬运。
   - 性能基线为 `BatchMatMul + ReduceMax + ReduceSum` 拆分实现的总耗时，按融合实现相对基线的加速比评分。
   - 使用 `msprof` 采集 Task Duration、Cube 利用率、Vector 利用率及 MTE 带宽利用率。

## 五、精度判断规则

精度要求：计算结果需满足以下精度误差要求：

- float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）。
- float16、bfloat16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）。
- int32：要求计算结果完全准确，无误差。

## 七、示例说明

### 示例 1：基础计算

- x1：shape=(1,2,2)，值为 `[[[1,0],[0,1]]]`。
- x2：shape=(1,2,3)，值为 `[[[1,0,-1],[0,1,0]]]`。
- `transposeX1=false`，`transposeX2=false`。
- BatchMatMul 结果为 `[[[1,0,-1],[0,1,0]]]`。
- MaxSim 结果为 `[[1,1]]`，最终输出 `y=[2.0]`。

### 示例 2：转置存储布局

- 逻辑矩阵内容与示例 1 相同。
- x1 按 $(B,K,M)$ 物理布局输入，x2 按 $(B,N,K)$ 物理布局输入。
- `transposeX1=true`，`transposeX2=true`。
- 两个属性仅说明上述 storage shape；x1 和 x2 的逻辑 shape 仍分别为 $(B,M,K)$ 和 $(B,K,N)$。算子不额外交换维度，而是直接按照对应存储布局完成相同逻辑计算，输出仍为 `y=[2.0]`。

### 示例 3：全负相似度

- x1：shape=(1,1,2)，值为 `[[[1,0]]]`。
- x2：shape=(1,2,2)，值为 `[[[-1,-2],[0,0]]]`。
- BatchMatMul 结果为 `[[[-1,-2]]]`，MaxSim 结果为 `[[-1]]`。
- 最终输出 `y=[-1.0]`，不能因错误地将 MaxSim 初始值设为 0 而输出 0。
