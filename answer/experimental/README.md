# 性能实验版

`../kernel.asc` 是平台 **15/15 Pass** 的保守版，保持不动。
测试点的 B/M/N/K、数据类型和转置标志均不公开，不能按测试点编号写特化分支。

| 文件 | 与通过版的关系 | 平台状态 |
| --- | --- | --- |
| `kernel_batched_reduction.asc` | 仍用 16×128 块和相同单次启动；每行最大值写入独立的 32 字节位置，把每个 C 块的 V→S 等待从最多 16 次合并为 1 次 | 平台 15/15 Pass，耗时与保守版接近 |
| `kernel_vecin_same_schedule.asc` | 从 15/15 通过版复制；只把 Matmul 的 C 小块从 GM 写回再读入，改为 `GetTensorC` 输出到 VECIN；16×128 分块、batch 分配、归约和 host 分配均保持原样 | **平台 11/15 Pass；第 2、3、6、8 项 Wrong Answer，错误占比均为 100%** |
| `kernel_vecin_full_tiles_only.asc` | 在上一版基础上，只有 M 为 16 的倍数且 N 为 128 的倍数时使用 VECIN；其余形状回退到 15/15 通过版的 GM 小块路径，检验尾块假设 | **平台 15/15 Pass；耗时与前版接近，没有观察到有效提速** |
| `kernel_flash_maxsim.asc` | 论文式片上分块：Matmul 的 C 小块进入 VECIN，按 `N` 在线更新行最大值；把 `M` 行块分给多个核，单次启动内同步并汇总 FP32 局部和 | **平台结果：Wrong Answer；尚无失败用例的实际值与期望值** |
| `kernel_parallel_rows_gm.asc` | 沿用 15/15 通过版的 GM 小块输出和 16×128 分块；实验按 M 行块分核、写独立局部和并在单次启动内汇总 | **平台 0/15：全部 Wrong Answer，错误占比 100%，用时为「-」** |
| `kernel_parallel_rows_gm_v2.asc` | 修正 C 队列分配长度、改动跨核同步参数并对齐系统 workspace | 未经平台验证；**不建议提交**，组内通过版仍使用 `SyncAll<true>` |
| `kernel_team_passed.asc` | 用户提供的组内通过且当前最高分源码的原样备份 | 用户报告平台通过；本机未复验 |
| `kernel_team_vecin_full_tiles.asc` | 基于组内版本，仅在完整 M/N 块时将 Matmul 的 C 输出改为同步 `GetTensorC` 到 VECIN；其他形状和 tiling 失败时仍走原 GM 路径 | **平台 15/15 Pass；用户报告与组内原版基本相同，未见有效提速** |
| `kernel_32x256_candidate.asc` | 改为 32×256 块，减少 Matmul 调用和临时块数 | 待验证 |
| `kernel_wide_16x512.asc` | 改为 16×512 块 | Runtime Error：Expected 75 launches, got 50；根因不明 |
| `kernel_tiling_fallback.asc` | 宽块 tiling 失败时回退到窄块 | 待验证 |

`kernel_batched_reduction.asc` 的平台结果说明：合并每块的标量等待
没有带来明显提速。按用户本轮要求，下面的片上版改动了数据路径和
任务划分；其结果仍需平台实测。

## 单变量 VECIN 候选版

`kernel_vecin_same_schedule.asc` 用于区分旧实验版的 Wrong Answer 是否来自
矩阵结果的数据路径。它保留通过版的 batch 所有权、`16×128` 块、逐行
`ReduceMax`、最终 `ReduceSum`、单次启动，以及 host 的 workspace 分配。
唯一的功能性改动是把 C 的逻辑输出位置设为 VECIN，再用同步的
`Iterate`/`GetTensorC` 取得小块；读行时仍按 `baseN` 跨度访问，
`enSequentialWrite=true` 要求输出连续的 `baseM×baseN` 小块。

平台结果为 11/15 Pass，失败编号 2、3、6、8，四项的输出错误占比均为
100%。隐藏用例的形状、类型及转置标志不公开，因此这只能证明 VECIN
路径在某些输入上有系统性错误，不能仅凭编号确定是 M/N 尾块。

`kernel_vecin_full_tiles_only.asc` 是一次黑盒诊断版。它仅在
`M % 16 == 0` 且 `N % 128 == 0` 的完整分块场景使用 VECIN；其他输入
使用通过版的 GM 路径。2026-09-17 用户截图显示平台 15/15 Pass。
这与“边界形状触发错误”的假设相容，但隐藏用例形状不公开，不能据此
确定原版的错误原因。此版仍使用 16×128 分块及按 batch 分组的任务划分。
截图中的耗时与前一次 11/15 版本的对应测试点基本相同，没有观察到
有效提速；该片上改动不应作为最终性能优化成果。

## 按行块并行的 GM 实验版

`kernel_parallel_rows_gm.asc` 保留已通过版的 `IterateAll` 到 GM、再拷入
VECIN 的路径，保留 16×128 块及每行最大值计算。它把不同的 M 行块
分给不同 AIV，每块把 FP32 局部和写入独立的 64 字节位置；所有工作者
完成后同步，再按 batch 汇总。这针对小 batch 时原代码只有极少数核
工作的瓶颈，同时避开已观察到错误的 VECIN Matmul 输出路径。
用户随后提交的首版平台结果为 0/15：所有测试点 Wrong Answer、错误占比
100%，用时栏均为「-」。这表明该版没有得到可接受的输出，但截图不能
区分 host 提前返回、设备执行失败与计算结果错误。源码中有一个确定的
缓冲区长度问题：`cq` 按 `tileRows` 分配，却按 `MAX_ROWS` 拷贝；当
`tileRows < MAX_ROWS` 时会越界。此前我按通用混合核同步文档，把
`SyncAll<true>` 判作风险并在 `v2` 中改为 `SyncAll<false>`；这个推断
证据不足。用户随后提供的组内通过版也使用高阶 Matmul API，且在
`REGIST_MATMUL_OBJ` 后的 AIV 汇总路径调用 `SyncAll<true>`。因此不能把
同步参数当作首版 0/15 的已知根因，也不能把 `v2` 当作修复版提交。
组内通过版先写完整的行最大值片段到 GM，同步后用 `DataCopy` 读回 UB
做最终归约；首版写单个 FP32 局部和并用 `GlobalTensor::GetValue` 跨核
读取，这些差异更值得核查，但仍需设备证据确认根因。原版保留用于对照。

## 组内通过版的片上 C 输出实验

`kernel_team_passed.asc` 与用户提供的文件 SHA-256 一致，作为回退版本。
`kernel_team_vecin_full_tiles.asc` 保留其 B/M/N 任务划分、32/64 行规划、
最大值归约、跨核汇总和 `M=N=1` 向量点积。只有 `M` 能被当前行块整除且
`N` 能被当前列块整除时，才尝试把 Matmul C 输出位置设为 VECIN，使用同步的
`Iterate`/`GetTensorC(..., enSequentialWrite=true)` 取得连续小块；否则使用
原 `IterateAll` 到 GM 再读入 UB 的路径。若片上 tiling 失败，会在启动前
回退到原 GM tiling。此回退不能处理设备执行后的 Wrong Answer。

此前 16×128 的 VECIN 实验虽在完整块限制后通过 15/15，耗时却与 GM
路径接近。因此本实验只用于判断在组内更强的并行与向量归约实现中，
去掉 C 小块的 GM 往返是否有可测收益。用户本轮截图显示平台 15/15
Pass，测试点 6、7 分别为 17.58、19.62 微秒，8 为 132.89 微秒；
用户随后说明组内原版的用时基本相同，因此片上改动未见有效提速。
与更早的 16×128 按 batch 分配版本相比，
本版快很多，但这主要还包含组内原版的分块、并行和向量归约改进，
不能归功于 VECIN。隐藏测试点没有公开形状，当前也缺少组内原版的
逐项用时，暂时无法给片上路径制定有证据支持的切换阈值。

## 论文式片上实验版

`kernel_flash_maxsim.asc` 保留原输入、四种物理转置布局、FP16/BF16、
FP32 点积与输出。它把每个 `M` 行块作为任务分配到不同 AIV，沿 `N`
扫描时在片上取每行最大值；每个任务只写一个 FP32 局部和到独占的
64 字节 workspace 槽。AIV 核间 `SyncAll<true>` 后由各输出组的所有者汇总。
因此，它不再把整个 `ROWS × baseN` 相似度块写到 GM，仍只有一个
`<<<...>>>` 启动表达式。

这是**单独的、目前未通过正确性评测的候选版**。2026-09-17 用户确认平台结果为
Wrong Answer；在取得至少一个失败用例的实际输出、期望输出、形状、数据类型与转置标志前，
不能确定错误来自 VECIN 输出布局、多核同步汇总还是数值处理。优先检查的是：
`GetTensorC(..., enSequentialWrite=true)` 对 M/N 尾块的实际排布，
Matmul 高阶接口与混合 AIC/AIV 启动下的 AIV 核间同步，以及宽范围
K、四种转置下的 tiling。若平台提示启动次数不足，首先检查 host 侧
`GetTiling` 和 `aclrtMalloc` 的提前返回；若卡住，优先检查混合核同步。
本机无 CANN 编译器与 NPU，不能把源码检查当作设备验证。

评测方法：仅把该文件内容复制到官方模板的 `kernel.asc` 提交；
`../kernel.asc` 是已通过的回退版本。先定位并修复 Wrong Answer，重新通过
15 个测试点后再比较耗时，不要用论文的 A100/H100 加速比估计本题得分。
