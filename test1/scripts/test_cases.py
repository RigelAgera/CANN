"""Independent regression driver. CPU checks do NOT execute the Ascend C kernel."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np

from bf16 import as_dtype, bfloat16, write_tensor
from BatchMatmulMaxSum import impl


def cases():
    # Test every storage combination and dtype, including a K tail and M/N tails.
    for dtype, ta, tb in itertools.product(('float16', 'bfloat16'), (False, True), (False, True)):
        for shape, kind in [((2, 17, 131, 40), 'random'), ((3, 3, 7, 32), 'negative'),
                            ((17, 3, 17, 40), 'random'), ((33, 17, 129, 32), 'negative'),
                            ((1, 1, 1, 32), 'random'), ((1, 16, 128, 64), 'zero')]:
            yield dict(shape=shape, dtype=dtype, ta=ta, tb=tb, kind=kind)
    for shape in [(64, 1, 1, 32), (1, 3, 5, 8192), (1, 8192, 1, 32),
                  (1, 1, 8192, 32), (1, 33, 257, 72), (2, 31, 129, 136)]:
        for dtype in ('float16', 'bfloat16'):
            yield dict(shape=shape, dtype=dtype, ta=True, tb=True, kind='random')


def inputs(case, seed):
    b, m, n, k = case['shape']
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (b, m, k))
    x = rng.uniform(-1, 1, (b, k, n))
    if case['kind'] == 'negative':
        a = np.abs(a) + 0.1
        x = -np.abs(x) - 0.1
    elif case['kind'] == 'zero':
        a.fill(0)
    dtype = np.float16 if case['dtype'] == 'float16' else bfloat16
    a, x = as_dtype(a, dtype), as_dtype(x, dtype)
    if case['ta']:
        a = a.swapaxes(1, 2)
    if case['tb']:
        x = x.swapaxes(1, 2)
    return np.ascontiguousarray(a), np.ascontiguousarray(x)


def tiled_cpu(a, x, case):
    """Model original strides, single-kernel batch ownership and valid tails only."""
    b, m, n, k = case['shape']
    bn = 256 if n >= 256 else (n+15)//16*16
    flat_a, flat_x = a.astype(np.float32).reshape(-1), x.astype(np.float32).reshape(-1)
    result = np.full(b, np.nan, dtype=np.float32)
    writes = np.zeros(b, dtype=np.int32)
    groups = (b + 15) // 16
    workers = 2 * min(24, (groups + 1) // 2)
    output_blocks = {}
    for worker in range(workers):
        for group in range(worker, groups, workers):
            first_batch = group * 16
            assert first_batch * 4 % 64 == 0
            for bi in range(first_batch, min(b, first_batch + 16)):
                block = bi * 4 // 64
                assert output_blocks.setdefault(block, worker) == worker, 'shared output cache line'
                maxima = np.full(m, np.nan, dtype=np.float32)
                for mi in range(0, m, 32):
                    valid_m = min(32, m-mi)
                    ao = bi*m*k + (mi if case['ta'] else mi*k)
                    bo = bi*k*n
                    rr, kk = np.arange(valid_m)[:, None], np.arange(k)[None, :]
                    aa = flat_a[ao + (kk*m+rr if case['ta'] else rr*k+kk)]
                    running = np.full(valid_m, -np.inf, dtype=np.float32)
                    for ni in range(0, n, bn):
                        valid_n = min(bn, n-ni)
                        kk, nn = np.arange(k)[:, None], np.arange(ni, ni+valid_n)[None, :]
                        xx = flat_x[bo + (nn*k+kk if case['tb'] else kk*n+nn)]
                        sim = aa @ xx
                        # Explicit C stride is bn, including the last N tile.
                        # Poison unwritten lanes to catch accidental padding use.
                        c = np.full((32, bn), np.nan, dtype=np.float32)
                        c[:valid_m, :valid_n] = sim
                        running = np.maximum(running, c[:valid_m, :valid_n].max(axis=1))
                    maxima[mi:mi+valid_m] = running
                result[bi] = maxima.sum(dtype=np.float32)
                writes[bi] += 1
    assert np.all(writes == 1), 'missing or duplicate worker ownership'
    return result


def check(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError('Wrong shape or non-finite output')
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['cpu-check', 'generate', 'run', 'verify'])
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1]/'test_data')
    parser.add_argument('--runner', type=Path)
    args = parser.parse_args()
    if args.mode == 'run' and args.runner is None:
        parser.error('run requires --runner pointing to the compiled NPU executable')
    count = 0
    for index, case in enumerate(cases()):
        a, x = inputs(case, 20260912 + index)
        expected = impl(a, x, case['ta'], case['tb'])
        folder = args.directory / f'case{index:03d}'
        if args.mode == 'cpu-check':
            check(tiled_cpu(a, x, case), expected)
        if args.mode in ('generate', 'run'):
            folder.mkdir(parents=True, exist_ok=True)
            # write_tensor emits real 2-byte bfloat16 files; the runner needs them.
            write_tensor(folder/'x1.bin', a, case['dtype'])
            write_tensor(folder/'x2.bin', x, case['dtype'])
            expected.tofile(folder/'golden_y.bin')
            (folder/'case.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
        if args.mode == 'run':
            subprocess.run([str(args.runner.resolve()), str(folder.resolve()),
                *map(str, case['shape']), case['dtype'], str(int(case['ta'])), str(int(case['tb']))],
                check=True, timeout=180)
        if args.mode in ('run', 'verify'):
            actual = np.fromfile(folder/'actual_y.bin', dtype=np.float32)
            check(actual, expected)
        count += 1
    print(f'{args.mode}: {count} cases passed/completed.')
    if args.mode == 'cpu-check':
        print('CPU mathematical/storage model only; CANN compilation and NPU execution are NOT tested.')


if __name__ == '__main__':
    main()
