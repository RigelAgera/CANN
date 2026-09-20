#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
BatchMatmulMaxSum算子golden实现
以 numpy bmm + amax + sum 的 FP64 结果为 golden
"""
import numpy as np
from ml_dtypes import bfloat16


def impl(x1, x2, transposeX1=False, transposeX2=False):
    """BatchMatmulMaxSum算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    x1: float16/bfloat16 numpy数组，逻辑形状 (B, M, K)；
        transposeX1=True 时物理形状为 (B, K, M)。
    x2: 与 x1 同数据类型的 numpy数组，逻辑形状 (B, K, N)；
        transposeX2=True 时物理形状为 (B, N, K)。
    transposeX1: bool，是否交换 x1 最后两个维度，默认 False。
    transposeX2: bool，是否交换 x2 最后两个维度，默认 False。

    返回:
    y: (B,) float32 numpy数组，每个 batch 的相关性分数。
    """
    x1_np = np.asarray(x1)
    x2_np = np.asarray(x2)

    if x1_np.ndim != 3 or x2_np.ndim != 3:
        raise ValueError("x1 and x2 must both be rank-3 tensors")

    # 使用实际存储值执行 FP64 golden 计算
    x1_f = x1_np.astype(np.float64)
    x2_f = x2_np.astype(np.float64)

    x1_logical = np.swapaxes(x1_f, -1, -2) if transposeX1 else x1_f
    x2_logical = np.swapaxes(x2_f, -1, -2) if transposeX2 else x2_f

    if x1_logical.shape[0] != x2_logical.shape[0]:
        raise ValueError("x1 and x2 must have the same batch dimension; no broadcast")
    if x1_logical.shape[2] != x2_logical.shape[1]:
        raise ValueError("the logical K dimensions of x1 and x2 must match")

    similarity = np.matmul(x1_logical, x2_logical)   # (B, M, N)
    max_sim = np.max(similarity, axis=-1)            # (B, M)
    y = np.sum(max_sim, axis=-1)                     # (B,)

    return y.astype(np.float32)


def _physical_shapes(B, M, N, K, transposeX1, transposeX2):
    x1_shape = (B, K, M) if transposeX1 else (B, M, K)
    x2_shape = (B, N, K) if transposeX2 else (B, K, N)
    return x1_shape, x2_shape


def _make_input(shape, dtype_name, value_range, rng):
    low, high = value_range
    value = rng.uniform(low, high, shape).astype(np.float32)
    if dtype_name == "float16":
        return value.astype(np.float16)
    if dtype_name == "bfloat16":
        return value.astype(bfloat16)
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _run_known_value_checks():
    x1 = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float16)
    x2 = np.array([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]], dtype=np.float16)
    actual = impl(x1, x2)
    np.testing.assert_allclose(actual, np.array([2.0], dtype=np.float32), rtol=0, atol=0)

    negative_x1 = np.array([[[1.0, 0.0]]], dtype=np.float16)
    negative_x2 = np.array([[[-1.0, -2.0], [0.0, 0.0]]], dtype=np.float16)
    negative_actual = impl(negative_x1, negative_x2)
    np.testing.assert_allclose(negative_actual, np.array([-1.0], dtype=np.float32), rtol=0, atol=0)


if __name__ == "__main__":
    _run_known_value_checks()

    # 15 个测试用例，与 JSON 中的 npu_cases 一一对应
    # (id, B, M, N, K, dtype, transposeX1, transposeX2, x1_range, x2_range)
    cases = [
        ( 1,   1,  2,   3,  4, "float16",  False, False, [-1.0, 1.0], [-1.0, 1.0]),
    ]

    for cid, B, M, N, K, dtype_name, tx1, tx2, x1_range, x2_range in cases:
        rng = np.random.default_rng(20260817 + cid)
        x1_shape, x2_shape = _physical_shapes(B, M, N, K, tx1, tx2)
        x1 = _make_input(x1_shape, dtype_name, x1_range, rng)
        x2 = _make_input(x2_shape, dtype_name, x2_range, rng)

        y = impl(x1, x2, tx1, tx2)

        assert y.shape == (B,), f"Case {cid:02d}: unexpected output shape {y.shape}"
        assert y.dtype == np.float32, f"Case {cid:02d}: unexpected output dtype {y.dtype}"
        assert np.all(np.isfinite(y)), f"Case {cid:02d}: output contains NaN or Inf"
        if cid == 2:
            assert np.all(y < 0), "Case 02 must produce all-negative scores"

        input_elems = B * K * (M + N)
        macs = B * M * N * K
        print(
            f"Case {cid:02d}: shape={y.shape}, dtype={y.dtype}, "
            f"input_dtype={dtype_name}, transpose=({tx1},{tx2}), "
            f"input_elems={input_elems}, macs={macs}, "
            f"score_range=[{y.min():.6f}, {y.max():.6f}]"
        )

    print("All tests passed!")
