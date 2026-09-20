"""Exact verification of the ml_dtypes-free bfloat16 layer (CPU only, no NPU).

The vectorised rounding in ``scripts/bf16.py`` is checked against an oracle
that computes the nearest bfloat16 with exact rational arithmetic, so a bug in
the bit-trick cannot pass unnoticed.
"""
import shutil
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bf16  # noqa: E402


def f32_from_bits(pattern: int) -> np.float32:
    return np.array([pattern], dtype=np.uint32).view(np.float32)[0]


def _round_half_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator:
        return quotient + 1
    if twice == denominator:
        return quotient + (quotient & 1)
    return quotient


def encode_exact(pattern: int) -> int:
    """Exact round-half-to-even float32 bits -> bfloat16 bits."""
    sign = (pattern >> 31) & 1
    exp32 = (pattern >> 23) & 0xFF
    mant32 = pattern & 0x7FFFFF
    if exp32 == 0xFF:
        return (sign << 15) | (0x7FC0 if mant32 else 0x7F80)
    if mant32 == 0 and exp32 == 0:
        return sign << 15
    if exp32 == 0:
        exponent = mant32.bit_length() - 1 - 149
        magnitude = Fraction(mant32, 1 << 149)
    else:
        exponent = exp32 - 127
        magnitude = Fraction((1 << 23) | mant32, 1 << 23) * Fraction(2) ** exponent

    if exponent < -126:
        scaled = magnitude * Fraction(2) ** 133          # quantum is 2**-133
        kept = _round_half_even(scaled.numerator, scaled.denominator)
        if kept == 0:
            return sign << 15
        if kept >= 128:                                  # rounds up to 2**-126
            return (sign << 15) | 0x0080
        return (sign << 15) | kept

    scaled = magnitude * Fraction(2) ** (7 - exponent)    # quantum is 2**(e-7)
    kept = _round_half_even(scaled.numerator, scaled.denominator)
    if kept == 256:                                      # carry into exponent
        kept = 128
        exponent += 1
    if exponent > 127:
        return (sign << 15) | 0x7F80
    return (sign << 15) | ((exponent + 127) << 7) | (kept - 128)


# (float32 bit pattern, expected bfloat16 bits, description)
HAND_CHECKED = [
    (0x00000000, 0x0000, 'positive zero'),
    (0x80000000, 0x8000, 'negative zero'),
    (0x3F800000, 0x3F80, '1.0'),
    (0xBF800000, 0xBF80, '-1.0'),
    (0x40000000, 0x4000, '2.0'),
    (0x42C80000, 0x42C8, '100.0'),
    (0x3F810000, 0x3F81, '1+1/128, exact'),
    (0x3F808000, 0x3F80, '1+1/256, tie to even'),
    (0x3F818000, 0x3F82, '1+3/256, tie to even'),
    (0x3DCCCCCD, 0x3DCD, '0.1'),
    (0x7F7F0000, 0x7F7F, 'largest finite bfloat16'),
    (0x7F7FFFFF, 0x7F80, 'float32 max overflows to infinity'),
    (0xFF7FFFFF, 0xFF80, 'negative overflow'),
    (0x00000001, 0x0000, 'smallest float32 subnormal rounds to zero'),
    (0x00010000, 0x0001, '2**-133, smallest bfloat16 subnormal'),
    (0x00008000, 0x0000, '2**-134, tie to even zero'),
    (0x00018000, 0x0002, '3*2**-134, tie to even'),
    (0x7F800000, 0x7F80, 'infinity'),
    (0xFF800000, 0xFF80, 'negative infinity'),
    (0x7FC00000, 0x7FC0, 'quiet NaN'),
    (0x7FFFFFFF, 0x7FC0, 'all-ones NaN stays NaN'),
    (0xFFC00000, 0xFFC0, 'negative NaN keeps its sign'),
]


class Bfloat16Compat(unittest.TestCase):
    def test_hand_checked_patterns(self):
        values = np.array([pattern for pattern, _, _ in HAND_CHECKED], dtype=np.uint32)
        actual = bf16.round_to_bf16_bits(values.view(np.float32))
        for (pattern, expected, label), got in zip(HAND_CHECKED, actual):
            with self.subTest(case=label):
                self.assertEqual(int(got), expected,
                                 f'{label}: 0x{pattern:08X} -> 0x{int(got):04X}, want 0x{expected:04X}')

    def test_matches_exact_oracle_on_random_patterns(self):
        rng = np.random.default_rng(20260913)
        patterns = rng.integers(0, 1 << 32, size=8000, dtype=np.uint32)
        # Concentrate a slice on the subnormal/overflow edges of the range.
        edges = np.array([0x00000001, 0x00008000, 0x00010000, 0x00018000, 0x00800000,
                          0x0A7FFFFF, 0x0A800000, 0x7F7EFFFF, 0x7F7F0000, 0x7F7FFFFF,
                          0x7F800000, 0x7FFFFFFF, 0xFF7FFFFF], dtype=np.uint32)
        patterns = np.concatenate([patterns, edges])
        actual = bf16.round_to_bf16_bits(patterns.view(np.float32))
        mismatches = []
        for pattern, got in zip(patterns, actual):
            want = encode_exact(int(pattern))
            if int(got) != want:
                mismatches.append((int(pattern), int(got), want))
        self.assertEqual(mismatches[:5], [], f'{len(mismatches)} mismatches, first five shown')

    def test_every_bfloat16_value_is_a_fixed_point(self):
        all_bits = np.arange(1 << 16, dtype=np.uint16)
        carried = bf16.bits_to_float32(all_bits)
        again = bf16.round_to_bf16_bits(carried)
        is_nan = ((all_bits & np.uint16(0x7F80)) == np.uint16(0x7F80)) & \
                 ((all_bits & np.uint16(0x007F)) != 0)
        self.assertTrue(np.array_equal(again[~is_nan], all_bits[~is_nan]),
                        'a non-NaN bfloat16 pattern was changed by a round trip')
        expected_nan = (all_bits[is_nan] & np.uint16(0x8000)) | np.uint16(0x7FC0)
        self.assertTrue(np.array_equal(again[is_nan], expected_nan))

    def test_bit_patterns_decode_as_values_not_numbers(self):
        # 0x3F80 is 1.0 and 0x4049 is 3.140625; treating patterns as integers
        # would give 16256.0 and 16457.0 instead.
        decoded = bf16.bits_to_float32(np.array([0x3F80, 0xC049, 0x0000, 0x7F80], dtype=np.uint16))
        self.assertEqual(decoded.tolist(), [1.0, -3.140625, 0.0, float('inf')])

    def test_carried_values_are_exactly_representable(self):
        sample = bf16.bf16_float32(np.arange(0, 4096, dtype=np.uint16))
        self.assertEqual(sample.dtype, np.dtype(np.float32))
        self.assertTrue(np.array_equal(bf16.bf16_float32(sample).astype(np.float32), sample))

    def test_files_use_two_bytes_per_element(self):
        n = 1000
        values = np.linspace(-1.0, 1.0, n)
        # The sandbox denies system temp paths, so scratch inside the checkout.
        scratch = Path(__file__).resolve().parents[1] / 'build-tmp' / 'bf16-io'
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        try:
            bf16_path = scratch / 'bf16.bin'
            f16_path = scratch / 'f16.bin'
            bf16.write_tensor(bf16_path, values, 'bfloat16')
            bf16.write_tensor(f16_path, values, 'float16')
            self.assertEqual(bf16_path.stat().st_size, 2 * n)
            self.assertEqual(f16_path.stat().st_size, 2 * n)
            round_tripped = bf16.read_tensor(bf16_path, bf16.bfloat16)
            self.assertEqual(round_tripped.dtype, np.dtype(np.float32))
            self.assertTrue(np.array_equal(round_tripped, bf16.bf16_float32(values)),
                            'a written bfloat16 file did not read back as the same values')
            self.assertEqual(bf16_path.read_bytes()[:2],
                             bf16.round_to_bf16_bits(np.array([values[0]], dtype=np.float32)).tobytes())
            # bfloat16 is coarser than float16, so the two files differ.
            self.assertFalse(np.array_equal(
                round_tripped, np.fromfile(f16_path, dtype=np.float16).astype(np.float32)))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_marker_matches_numpy_dtype_protocol_usage(self):
        self.assertEqual(bf16.bfloat16.__name__, 'bfloat16')
        self.assertTrue(bf16.is_bf16(bf16.bfloat16))
        self.assertFalse(bf16.is_bf16(np.float16))
        self.assertFalse(bf16.is_bf16(np.dtype(np.float32)))
        self.assertEqual(bf16.bfloat16.itemsize, 2)


if __name__ == '__main__':
    unittest.main()
