# BatchMatmulMaxSum 初赛解答

提交文件是 `kernel.asc`，接口为赛题要求的 `run_kernel`。本目录保留了
`test1` 的独立构建入口、数据脚本和回归检查，便于复现。

## 对论文方法的取舍

参考论文 *FLASH-MAXSIM: IO-Aware Fused Kernels for Late-Interaction Retrieval*
（`../2605.29517v2.pdf`）第 4.1 节和 Algorithm 1：按 query 行块、document
列块计算相似度；对每个 query 行维护从负无穷开始的 running max；扫描完全部
document 列块后，将有效 query 行的最大值用 FP32 求和。

赛题的逻辑公式是

```
y[b] = sum(m=0..M-1) max(n=0..N-1)
       sum(k=0..K-1) x1[b,m,k] * x2[b,k,n]
```

论文的反向传播、变长序列与 INT8 路径不属于本赛题的输入输出范围，因此这里
只实现前向 MaxSum。代码与论文的 GPU 实现有一处重要差别：Ascend C 的 Matmul
先把一个 `ROWS × baseN` 的 FP32 小块写到每个 worker 专有的 GM 临时区，
Vector 随即读回并做行归约。它不存储完整的 `B × M × N` 相似度张量，
峰值临时区与单个 tile 和 worker 数有关；但每个 tile 仍有一次 GM 写入和读回，
因此不能把论文中完全在片上归约的带宽收益或速度数字直接套用到本实现。

## 实现对应关系

- 针对本赛题 `1<=B<=64`、`1<=M,N<=8192`、`32<=K<=8192` 且 `K` 为 8 的倍数的范围选块。
  当前保守版使用最多 `16×128` 的矩阵块。Matmul 的乘加结果为 FP32。
- 每个有效行用 `-inf` 初始化。每个列块只对 `validN` 列取最大，再更新 running max；
  因此全负输入和 `N` 尾块不会被零填充改变结果。
- 所有 `M` 行最大值存在 UB，最终对有效的 `M` 行做一次 FP32 `ReduceSum`。
- 原始输入以物理形状存放，通过 `SetOrgShape`、转置模板参数和 `SetTail`
  处理四种布局与 `M/N/K` 尾块。输出为 `(B,)` 的 FP32。
- 一个 AIV 拥有连续 16 个 batch 输出，写回起点按 64 字节对齐；各 worker
  不共享缓存行。每次 `run_kernel` 只有一次 mixed Cube/Vector kernel launch。
- FP16 与 BF16 输入分别走对应的 Matmul 类型，没有改变输入精度，也不做近似量化。

当前提交用的 `kernel.asc` 与 `test1/tests/baselines/kernel_passed_v1.asc`
完全一致。用户在 2026-09-16 的平台评测中确认它 **15/15 Pass**，
所有测试点输出错误占比均为 0.00%。测试点形状不公开。
此前的 `16×512` 宽块版本在平台收到 Runtime Error，日志为
“Expected 75 launches, got 50”；因缺少逐次启动和设备报错日志，
不能确定是提前返回、内存分配失败，还是设备错误后启动失败。
实验版与用途见 `experimental/README.md`；其中论文式片上版本已在平台得到
Wrong Answer，其他候选版的状态以该文件记录为准。

## 本地检查

在本目录运行：

```bash
python tests/test_bf16_compat.py
python tests/test_single_kernel.py
python tests/test_optimization.py
python tests/test_examples.py
python scripts/test_cases.py cpu-check
```

这些检查覆盖 BF16 数据编码、单次 launch 的源码结构、四种转置组合、
尾块、负相似度、零输入和极端维度。CPU 模型通过不等于 Ascend C 编译或 NPU
通过；源码检查也不能代替 profiler 的实际 launch 计数。

在装有 CANN 9.0 和 dav-2201 NPU 的环境运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build
cmake --build build -j4
cmake -S tests -B build-tests
cmake --build build-tests -j4
python scripts/test_cases.py run --runner ./build-tests/batch_matmul_max_sum_cases
```

赛题要求 FP16/BF16 输入、FP32 输出和 FP32 归约；对 FP16/BF16 的结果，
相对及绝对误差均须小于 `1e-3`。15 个测试点需全部通过精度检查才计分。
当前 `kernel.asc` 的隐藏样例精度已由平台 15/15 通过结果确认。
本地机器没有 CANN 编译器和 NPU，因此实验版仍需由平台验证实际
launch 数、精度和设备端耗时。
