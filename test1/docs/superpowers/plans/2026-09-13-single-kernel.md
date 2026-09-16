# Single-kernel BatchMatmulMaxSum Implementation Plan

> For agentic workers: execute inline with executing-plans; review each task before proceeding.

**Goal:** Replace the four launches per valid run_kernel call with exactly one mixed Cube/Vector launch.

**Architecture:** Preserve the mixed Matmul API. Read original contiguous ND input storage directly, use SetOrgShape and SetTail for strides and tails, reduce each N tile to row maxima, and reduce all M rows inside the same kernel. Each AIV owns contiguous groups of 16 batch outputs (64 bytes); no output block is shared between workers. The original entry point and eight dtype/layout specializations are retained.

**Tech Stack:** Ascend C, CANN 9.0.0, dav-2201, Python/NumPy CPU model.

## Global Constraints

- One dynamic kernel launch per valid invocation; retain Cube matmul.
- FP16/BF16 input, FP32 dot products and output; four storage layouts.
- Max starts at negative infinity; exclude invalid M/N lanes.
- No separate padding, memset, row-sum or output-initialization kernel.
- Do not mutate input or platform runner files. Do not claim CANN/NPU validation locally.

## Task 1: Regression guard

- [x] Add tests/test_single_kernel.py: strip comments, assert one __global__ definition and one <<< launch site; require SetTail and original input strides; reject PadInput and SumRows entry points.
- [x] Run `python tests/test_single_kernel.py`; old source failed four checks as expected.

## Task 2: Implementation

- [x] Replace kernel.asc with one FusedMaxSum<T,TA,TB> entry point.
- [x] Use a templated LaunchTyped helper containing the only <<<...>>> expression; dispatch eight specializations through ordinary function calls.
- [x] For each owned batch, loop M by 16 and set `mm.SetTail(validM, n, k)`; allocate baseM*baseN C capacity. Review found compact N-tail output uses validN stride: full blocks use ReduceMax; compact tails use synchronized UB scalar reads.
- [x] Store row maxima in UB, ReduceSum over actual M, then write at most 16 contiguous FP32 outputs per group with DataCopyPad.
- [x] Allocate only system workspace on host; synchronize before releasing it.

## Task 3: Validation and handoff

- [x] Update scripts/test_cases.py tiled_cpu to use original storage and batch ownership. Add group-boundary B=17 and B=33 cases with both dtypes and all transpose combinations.
- [x] Six static checks and 60 CPU mathematical/storage checks passed; neither compiles or executes ASC. Python syntax checks passed.
- [x] Review against local official SetTail/GetTensorC/SetOrgShape documentation and mixed Matmul source. Keep original runner files unchanged.
- [ ] User submits kernel.asc; required platform evidence is 75 launches, compilation, and all 15 precision cases. Performance is a later optimization stage.
