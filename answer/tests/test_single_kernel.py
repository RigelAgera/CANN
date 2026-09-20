"""Source regression guard, NOT a substitute for CANN profiling or compilation."""
from pathlib import Path
import re
import unittest


class SingleKernelContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code = (Path(__file__).resolve().parents[1] / 'kernel.asc').read_text(encoding='utf-8')
        cls.code = re.sub(r'/\*.*?\*/|//[^\n]*', '', code, flags=re.S)

    def test_one_device_entry_point(self):
        self.assertEqual(len(re.findall(r'\b__global__\b', self.code)), 1)

    def test_one_launch_site(self):
        self.assertEqual(self.code.count('<<<'), 1)

    def test_no_separate_preprocess_or_postprocess(self):
        self.assertNotRegex(self.code, r'\b(PadInput|SumRows|aclrtMemset|aclrtMemsetAsync)\b')

    def test_original_storage_and_tails(self):
        self.assertIn('api.SetOrgShape(m, n, k)', self.code)
        self.assertIn('mm.SetTail(validM, validN, k)', self.code)

    def test_no_debug_output(self):
        self.assertNotRegex(self.code, r'\b(printf|fprintf|cstdio)\b')

    def test_explicit_c_stride(self):
        self.assertIn('mm.SetOrgShape(m, n, k, k, tiling.baseN)', self.code)
        self.assertIn('mm.template IterateAll<true>(cg, 0, false)', self.code)
        self.assertNotIn('GetTensorC', self.code)


if __name__ == '__main__':
    unittest.main()
