# BatchMatmulMaxSum verification

`../kernel.asc` preserves the contest `run_kernel` ABI. The original main,
CMakeLists, generation script and verifier are unchanged.

The implementation launches one mixed Cube/Vector kernel per valid invocation.
It reads the original input storage without preprocessing. Matmul SetOrgShape
preserves physical strides and SetTail handles M/N/K edges. Each AIV owns groups
of 16 consecutive batch outputs, computes all query rows of each owned batch,
reduces only valid N columns, and sums its M row maxima in UB. Output groups
start on 64-byte boundaries; no global reduction or output atomics are needed.
Matmul writes each tile to a private GM scratch block with explicitly configured
C leading dimension (SetOrgShape's fifth argument), then Vector copies it to UB.
This replaces the previous version's dependence on packed GetTensorC tail layout.
GM scratch contains one 16*baseN FP32 tile per AIV plus the Matmul system workspace.
It is synchronized and freed before returning. This conservative baseline uses limited
parallelism (at most two Cube groups for B<=64); performance needs later tuning.

## Local CPU check (NumPy only)

`scripts/bf16.py` carries bfloat16 in float32 arrays with integer bit
manipulation, so these checks need no `ml_dtypes` extension and run on an
offline machine with a plain NumPy install:

```shell
python -m pip install numpy
python tests/test_bf16_compat.py
python tests/test_single_kernel.py
python tests/test_optimization.py
python tests/test_examples.py
python scripts/test_cases.py cpu-check
```

`tests/test_bf16_compat.py` checks the bfloat16 rounding against an exact
rational-arithmetic oracle (ties-to-even, subnormals, overflow, NaN) and
verifies that `generate` writes real 2-byte bfloat16 files. That layout matters:
`tests/main_cases.asc` reads each input as `batch * m * k * 2` bytes, so a
float16-encoded file would silently feed the kernel wrong numbers.

The source guard checks one device entry point and one launch expression; it
cannot count profiler events. The CPU check compares a NumPy model to FP64 golden,
including physical offsets, tile ownership, four layouts, both dtypes, tails,
negative similarities, zeros and dimension extremes. It does not compile or
execute Ascend C and is not evidence that the NPU kernel passes.

## Actual NPU verification (CANN 9.0, dav-2201)

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build
cmake --build build -j4
cmake -S tests -B build-tests
cmake --build build-tests -j4
python scripts/test_cases.py run --runner ./build-tests/batch_matmul_max_sum_cases
```

The additional runner initializes output with NaNs to catch missing writes.
Its verifier rejects wrong output lengths and non-finite values and uses
`atol=rtol=1e-4`. Generated data goes to `test_data`, separate from the template's
original one-case test. These 60 public regression cases are not the 15 hidden
judge cases. The old four-launch implementation was rejected by the profiler
(expected 75 launches, got 300). The following packed-output single-launch version
was reported as all Wrong Answer; no actual-versus-expected output was supplied.
The explicit-stride revision addresses the C transport layout dependency but
has not yet
been compiled or executed on NPU. Require both profiling acceptance and all 15
precision cases before treating it as a passing submission.

API reference: Huawei CANN asc-devkit official
`examples/01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_advanced_api`
for mixed direct invocation and Matmul registration; `Matmul_Kernel/GetTensorC`,
`Matmul_Tiling/SetFixSplit` for the streamed tile layout and split constraints.

Current verification limitation: the development PC has no CANN compiler or NPU.
Compilation, runtime synchronization correctness, precision on hardware and
judge scoring still require the commands above or a CANNJudge test run.

Do not interpret the CPU checks as proof that all Wrong Answer cases are fixed.
If the explicit-stride version still fails, collect one case's actual output,
expected output, shape, dtype and transpose flags before another implementation change.
