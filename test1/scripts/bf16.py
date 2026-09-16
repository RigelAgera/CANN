#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-NumPy bfloat16 support, so the CPU-side checks need NumPy only.

NumPy has no native bfloat16 dtype, and this offline development box cannot
install the ``ml_dtypes`` extension (no outbound network).  The CPU golden
model, the regression driver and the verifiers need only four operations, all
reproduced here with integer bit manipulation on top of NumPy:

* ``round_to_bf16_bits`` -- float32 -> bfloat16 bit pattern, round-half-to-even
* ``bf16_float32``       -- bfloat16 values carried in a float32 array
* ``write_bf16`` / ``read_bf16`` -- real 2-byte little-endian bfloat16 files
* ``bfloat16``           -- a dtype marker standing in for ``ml_dtypes.bfloat16``

The NPU runner needs genuine 2-byte bfloat16 files, and ``write_tensor`` emits
exactly those.  Rounding is applied to the float32 nearest to the input, so a
float64 input can differ from a direct float64 -> bfloat16 conversion by one
bfloat16 ulp in roughly one element in 2**29; that is far below the 1e-4
tolerances used here.  ``tests/test_bf16_compat.py`` verifies the rounding
against an exact rational-arithmetic oracle.
"""
from __future__ import annotations

import numpy as np

_U32 = np.uint32
_EXP_MASK = _U32(0x7F800000)
_MANT_MASK = _U32(0x007FFFFF)
_QUIET_NAN16 = np.uint16(0x7FC0)
_SIGN16 = np.uint16(0x8000)


class _Bfloat16DType:
    """Stand-in for the ``ml_dtypes.bfloat16`` dtype object.

    NumPy refuses to build an ndarray with a bfloat16 dtype, so this marker is
    only ever compared (``is_bf16``) or used for its ``__name__``; the actual
    storage is a float32 array holding bfloat16-exact values.
    """

    name = "bfloat16"
    __name__ = "bfloat16"
    itemsize = 2

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "bfloat16"

    def __eq__(self, other) -> bool:
        return isinstance(other, _Bfloat16DType)

    def __hash__(self) -> int:
        return hash("bfloat16")


bfloat16 = _Bfloat16DType()


def is_bf16(dtype) -> bool:
    """True when ``dtype`` selects the bfloat16 marker."""
    return dtype is bfloat16 or isinstance(dtype, _Bfloat16DType)


def round_to_bf16_bits(values) -> np.ndarray:
    """Return uint16 bfloat16 bit patterns for ``values`` (little-endian).

    Round-half-to-even, matching hardware and ``ml_dtypes``: overflow to
    infinity, bfloat16 subnormals, signed zero and NaN are handled explicitly.
    """
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(_U32)
    # Ties-to-even: add 0x7FFF plus bit 16 of the input, the bit being kept.
    kept_lsb = (bits >> _U32(16)) & _U32(1)
    rounded = (bits + _U32(0x7FFF) + kept_lsb) >> _U32(16)
    result = rounded.astype(np.uint16)
    # 0x7FFFFFFF would otherwise round into -0.0; keep NaN a NaN instead.
    nan_like = ((bits & _EXP_MASK) == _EXP_MASK) & ((bits & _MANT_MASK) != 0)
    if nan_like.any():
        sign = (bits >> _U32(16)).astype(np.uint16) & _SIGN16
        result = np.where(nan_like, sign | _QUIET_NAN16, result).astype(np.uint16)
    return result


def bf16_float32(values) -> np.ndarray:
    """Carry bfloat16 values in a float32 array; every value stays exact."""
    bits = round_to_bf16_bits(values).astype(_U32) << _U32(16)
    return bits.view(np.float32)


def bits_to_float32(bits) -> np.ndarray:
    """Interpret uint16 bfloat16 *bit patterns* as float32 values.

    Unlike :func:`bf16_float32`, the input is raw storage, not numbers: the
    16-bit pattern 0x3F80 decodes to 1.0.  Reading a file needs this; passing
    the patterns through :func:`round_to_bf16_bits` would treat 0x3F80 as the
    number 16256.0 and corrupt the data.
    """
    packed = np.ascontiguousarray(bits, dtype=np.uint16).astype(_U32) << _U32(16)
    return packed.view(np.float32)


def write_bf16(path, values) -> None:
    """Write real 2-byte little-endian bfloat16 elements to ``path``."""
    round_to_bf16_bits(values).tofile(path)


def read_bf16(path) -> np.ndarray:
    """Read a 2-byte little-endian bfloat16 file into a float32 array."""
    return bits_to_float32(np.fromfile(path, dtype=np.uint16))


def as_dtype(values, dtype):
    """Cast ``values`` to ``dtype``: a real NumPy dtype, or ``bfloat16``."""
    if is_bf16(dtype):
        return bf16_float32(values)
    return np.asarray(values).astype(dtype)


def write_tensor(path, values, dtype_name: str) -> None:
    """Write one tensor in the layout the compiled NPU runner reads."""
    if dtype_name == "bfloat16":
        write_bf16(path, values)
    elif dtype_name == "float16":
        np.ascontiguousarray(values, dtype=np.float16).tofile(path)
    else:
        raise ValueError(f"unsupported dtype: {dtype_name}")


def read_tensor(path, dtype) -> np.ndarray:
    """Read a file previously written by :func:`write_tensor`."""
    if is_bf16(dtype):
        return read_bf16(path)
    return np.fromfile(path, dtype=dtype)
