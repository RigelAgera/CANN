# kernel_917 优化候选版

提交候选：`kernel_917_optimized.asc`。用户本轮确认 `kernel_917.asc` 已通过；原文件保持不变。
本轮没有 CANN 编译器或 NPU，候选版尚未通过设备编译、精度和性能评测。

## 改动

1. **256 列完整块的树形最大值折叠。** 原版依次将三个 64 列段合并到首段，
   需要三条 Max 指令和三次 PIPE_V 屏障。候选先用一条指令同时合并相邻段对，
   再合并两个段对，只需两条 Max 和两次屏障；执行的元素比较总量不变。
   首条指令每行两个 repeat，最多 128 个 repeat。其他宽度及 N 尾块保留原路径。
   后续 WholeReduceMax、全负值初始化和最终 FP32 ReduceSum 保持原逻辑。
2. **均衡 N 分片。** 原版先计算每片 tile 数，再反推分片数，可能减少活跃核心。
   候选保留目标分片数，以 `floor(part*nTiles/nParts)` 定义边界，任意两片最多相差一个 tile。
   例如 B=3、M=32、N=8192、32 个可用 Cube 核时，原版 16 个 N 分片、48 个任务、
   24 个 Cube 核；候选 22 个分片、66 个任务、32 个 Cube 核。
   增加分片也会增加 partial workspace 和汇总工作，因此不能只凭核心数判断实际提速。

没有改变输入精度、四种转置布局、Matmul 的 GM 输出路径、全局屏障、单次启动策略、
输出写回所有权及 M=N=1 点积特化。未引入原子累加或跨调用缓存。

向量指令参数按官方定义检查：
[BinaryRepeatParams](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0013.html)、
[repeat 参数](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html)。

## 本地检查

运行 `python answer/tests/test_917_optimized.py`。
脚本从候选源码抽取规划器、分片边界和 MergeRowMax，用 g++ 编译 CPU 指令模型，
核对实际辅助函数，而非另写一份优化算法。

每种开关组合检查：

- 139,264 种行数、有效列数、物理行跨度组合，逐个比较首次与后续 tile 的行最大值；
  覆盖全负值和正负混合，使用 NaN 污染无效尾列，使用哨兵保护输出尾行。
- 323,840 种规划与缓冲区组合，检查分片无遗漏、无重叠、无空片、核心数量不超限、
  分片最大长度，以及用户 UB 加 64 KiB Matmul 预算不超过 192 KiB。
- 两个新增开关的四种组合，以及 BMMS_FOLDED_MAX=0 的兼容路径。

CPU 检查不能验证设备流水、Matmul 输出布局、四种转置的真实设备执行、NPU 精度或速度。

## 平台对比

候选文件顶部的宏可直接修改，不需要评测器支持额外编译参数。

| BMMS_TREE_MAX | BMMS_BALANCED_N | 用途 |
| --- | --- | --- |
| 1 | 1 | 默认候选 |
| 1 | 0 | 只评估树形归约 |
| 0 | 1 | 只评估均衡 N 分片 |
| 0 | 0 | 恢复原版计算与调度路径 |

保持 BMMS_FOLDED_MAX=1、BMMS_VECTOR_DOT=1。各版本先跑全量精度，再以相同设备、
输入与计时口径比较多次耗时；不能将指令数量的减少直接报告为整体加速百分比。
