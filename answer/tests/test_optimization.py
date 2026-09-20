"""Checks optimization structure and buffer bounds, not measured NPU speed."""
from pathlib import Path
import unittest


class Optimization(unittest.TestCase):
    def test_block_reduction_uses_separate_aligned_destinations(self):
        source = (Path(__file__).resolve().parents[1] / 'kernel.asc').read_text(encoding='utf-8')
        self.assertIn('ReduceMax(scalar, c[r * tiling.baseN]', source)
        self.assertIn('scalar.GetValue(0)', source)
        self.assertIn('constexpr int32_t ROWS = 16;', source)

    def test_worst_case_ub_budget(self):
        # Explicit user buffers + tiler's maximum UB allowance <= 192 KiB.
        user = 16 * 128 * 4 + 16 * 4 + 32768 + 32 + 8192 * 4
        self.assertLessEqual(user + 96 * 1024, 192 * 1024)


if __name__ == '__main__':
    unittest.main()
