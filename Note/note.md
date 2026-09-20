# note

## AI 基础

## NPU

1. 为什么需要
   1. 计算密集
   2. 数据并行
   3. 低精度可接受（FP16/BF16）
2. CPU versus GPU
   1. CPU：标量计算
   2. Vector：向量计算
   3. Cube：矩阵计算
3. 略
4. NPU和CPU协作 （Device and Host）
5. NPU 组件
6. NPU计算核心
   1. Calculation Unit
      1. Cube Unit
      2. Vector Unit
      3. Scalar Unit
   2. Storage System
   3. Control Unit
7. 并行
   1. Tiling：切分数据
   2. 多核并行：每个AI Core 独立处理自己的数据块
   3. 隐式同步：所有核完成后TS统一回收
8. 架构对比

## CANN

1. 作用：对上适配AI框架，对下驱动NPU硬件
   1. 把PyTorch/MindSpore的算子调用翻译成NPU能执行的命令
   2. 优化计算顺序，融合算子（？），分配内存，调度多核并行
   3. 提供预置算子，通信库，调试工具
2. 

## 4.NPU实践

### 关键API

- Host -> Device: `tensor.npu()`
- Device -> Host: `tensor.cpu()`
- NPU同步: `torch.npu.synchronize()`
- 矩阵乘法: `torch.matmul(a, b)`
- 卷积：
