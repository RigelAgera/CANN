"""Exact problem examples and K=32 equivalents; CPU model only, not NPU tests."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bf16 import as_dtype, bfloat16
from BatchMatmulMaxSum import impl
from test_cases import tiled_cpu


class Examples(unittest.TestCase):
    def test_all_examples(self):
        identity = [[[1, 0], [0, 1]]]
        document = [[[1, 0, -1], [0, 1, 0]]]
        examples = [(identity, document, False, 2),
                    (identity, document, True, 2),
                    ([[[1, 0]]], [[[-1, -2], [0, 0]]], False, -1)]
        for index, (a, b, trans, expected) in enumerate(examples, 1):
            for k in (2, 32):
                for dtype in (np.float16, bfloat16):
                    with self.subTest(example=index, k=k, dtype=dtype.__name__):
                        aa = as_dtype(np.pad(np.array(a), ((0, 0), (0, 0), (0, k - 2))), dtype)
                        bb = as_dtype(np.pad(np.array(b), ((0, 0), (0, k - 2), (0, 0))), dtype)
                        case = dict(shape=(1, aa.shape[1], bb.shape[2], k), ta=trans, tb=trans)
                        if trans:
                            aa, bb = aa.swapaxes(1, 2), bb.swapaxes(1, 2)
                        aa, bb = np.ascontiguousarray(aa), np.ascontiguousarray(bb)
                        np.testing.assert_array_equal(impl(aa, bb, trans, trans), [expected])
                        np.testing.assert_array_equal(tiled_cpu(aa, bb, case), [expected])


if __name__ == '__main__':
    unittest.main()
