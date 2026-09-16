# Optimization tracking

## Correctness baseline

`baselines/kernel_passed_v1.asc` is an exact copy of the explicit-C-stride
implementation reported as passing by the user. No profiler timings were supplied.
Keep this file for rollback and timing comparisons; only root `kernel.asc` is submitted.

## Candidate 1: larger tiles and batched scalar synchronization

Root `kernel.asc` now uses 32x256 instead of 16x128 maximum C tiles.
Row reductions write distinct 32-byte-aligned destinations, with a Vector pipeline
barrier between reductions sharing scratch and a single V-to-scalar event per tile.
The GM output stride, batch ownership, accumulation order over M, and one-launch
entry point are retained. Matmul UB budget is 64 KiB; user buffers at the maximum
shape total 99,392 bytes, below the combined 192 KiB limit.

For a full M=N=512 batch, source-level counts change as follows:

| Operation | Baseline | Candidate 1 |
|---|---:|---:|
| Matmul tile calls | 128 | 32 |
| Row ReduceMax calls | 2048 | 1024 |
| V-to-scalar waits for row maxima | 2048 | 32 |

These are operation counts, NOT measured speedups.

Local checks: nine unittest methods (including 12 example subcases), 60 CPU model
cases. CANN compilation, NPU precision and performance for this candidate are pending.

## Remaining optimization scope

The batch-group policy still limits active AIVs to four for B<=64; small batch
cases use one AIV. Splitting M across workers can address the largest remaining
parallelism limit but needs a single-kernel-safe final reduction and correct
cross-core synchronization. Do not call performance optimization complete based
on candidate 1 or CPU tests. Obtain case timings and preserve 15/15 precision plus
exactly one kernel per iteration before deciding which candidate to keep.
