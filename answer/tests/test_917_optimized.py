"""Compile the candidate's actual planning/reduction helpers against CPU models.

This checks indexing and finite-value equivalence, not Ascend compilation,
pipeline synchronization, Matmul behavior, or NPU performance.
Run: python answer/tests/test_917_optimized.py
"""
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experimental/kernel_917_optimized.asc"
BASELINE = ROOT / "experimental/kernel_917.asc"


def region(source, name):
    return source.split("// BEGIN_" + name, 1)[1].split("\n", 1)[1].split(
        "// END_" + name, 1
    )[0]


MODEL = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#define __aicore__
constexpr int PIPE_V = 0;
namespace AscendC {
template<class T> struct LocalTensor {
    std::vector<T>* storage;
    size_t offset;
    LocalTensor(std::vector<T>* s, size_t o = 0) : storage(s), offset(o) {}
    LocalTensor operator[](size_t i) const { return {storage, offset + i}; }
    T& at(size_t i) const { return storage->at(offset + i); }
};
struct BinaryRepeatParams {
    uint8_t db, ab, bb, dr, ar, br;
};
enum class ReduceOrder { ORDER_ONLY_VALUE };
template<int> void PipeBarrier() {}
template<class T> void Max(LocalTensor<T> d, LocalTensor<T> a,
    LocalTensor<T> b, uint64_t width, uint8_t times, BinaryRepeatParams p) {
    assert(width <= 64 && times > 0);
    for (int r = 0; r < times; ++r)
        for (size_t i = 0; i < width; ++i) {
            const auto av = a.at(r*p.ar*8 + (i/8)*p.ab*8 + i%8);
            const auto bv = b.at(r*p.br*8 + (i/8)*p.bb*8 + i%8);
            assert(std::isfinite(av) && std::isfinite(bv));
            d.at(r*p.dr*8 + (i/8)*p.db*8 + i%8) = std::max(av, bv);
        }
}
template<class T> void Max(LocalTensor<T> d, LocalTensor<T> a,
    LocalTensor<T> b, int count) {
    for (int i = 0; i < count; ++i) d.at(i) = std::max(a.at(i), b.at(i));
}
template<class T> void WholeReduceMax(LocalTensor<T> d, LocalTensor<T> a,
    int width, int times, int dstStride, int blockStride, int repeatStride,
    ReduceOrder) {
    for (int r = 0; r < times; ++r) {
        T v = -std::numeric_limits<T>::infinity();
        for (int i = 0; i < width; ++i) {
            const T x = a.at(r*repeatStride*8 + (i/8)*blockStride*8 + i%8);
            assert(std::isfinite(x));
            v = std::max(v, x);
        }
        d.at(r*dstStride) = v;
    }
}
}
constexpr int32_t ROWS = 32, WIDE_ROWS = 64;
'''

CHECKS = r'''
int main() {
    std::mt19937 rng(917);
    int reductionCases = 0;
    // Every valid width and row count, every legal stride >= width.
    // NaNs poison invalid columns; sentinels protect output tail rows.
    for (int baseN = 16; baseN <= 256; baseN += 16)
    for (int n = 1; n <= baseN; ++n)
    for (int m = 1; m <= 64; ++m) {
        std::vector<float> c(m*baseN), running(72, 123456), scratch(72, 123456);
        std::vector<float> expected(m, -std::numeric_limits<float>::infinity());
        for (int pass = 0; pass < 2; ++pass) {
            std::fill(c.begin(), c.end(), std::numeric_limits<float>::quiet_NaN());
            for (int r = 0; r < m; ++r)
            for (int col = 0; col < n; ++col) {
                // All-negative first tile; mixed signs on the next tile.
                float value = -float(1 + rng()%10000) / 32;
                if (pass && (rng() & 1)) value = -value;
                c[r*baseN+col] = value;
                expected[r] = std::max(expected[r], value);
            }
            MergeRowMax({&running}, {&scratch}, {&c}, m, n, baseN, pass == 0);
            for (int r = 0; r < m; ++r) assert(running[r] == expected[r]);
            for (int r = m; r < 72; ++r) assert(running[r] == 123456);
        }
        ++reductionCases;
    }
    int planCases = 0;
    for (int batch = 1; batch <= 64; ++batch)
    for (int m : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128,
                  129, 255, 256, 257, 1023, 4097, 8191, 8192})
    for (int n : {1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256,
                  257, 511, 512, 513, 767, 768, 769, 2049, 4096, 8191, 8192})
    for (int cores : {1, 2, 3, 4, 8, 16, 20, 24, 32, 48, 64}) {
        int bn = std::min(256, (n+15)/16*16);
        auto p = MakePlan(batch, m, n, bn, cores);
        int nt = (n+bn-1)/bn, mt = (m+p.rows-1)/p.rows;
        assert(p.cores > 0 && p.cores <= cores && p.workers == 2*p.cores);
        assert(p.tasks == batch*mt*p.nParts);
        int end = 0, lo = nt, hi = 0;
        for (int part = 0; part < p.nParts; ++part) {
            int first = PartTileBoundary(part, nt, p.nParts, p.nTilesPerPart);
            int last = PartTileBoundary(part+1, nt, p.nParts, p.nTilesPerPart);
            assert(first == end && last > first && last <= nt);
            lo = std::min(lo, last-first); hi = std::max(hi, last-first);
            assert(last-first <= p.nTilesPerPart);
            end = last;
        }
        assert(end == nt);
#if BMMS_BALANCED_N
        assert(hi-lo <= 1);
#endif
        const int paddedM = mt*p.rows;
        const int userBytes = 4*std::max(p.rows*bn, paddedM) + 4*p.rows
            + 4096 + 4*std::max(p.rows*8, paddedM) + 32;
        assert(userBytes + 64*1024 <= 192*1024);
        ++planCases;
    }
    std::cout << "PASS: " << reductionCases << " reduction cases, "
              << planCases << " plan/buffer cases\n";
}
'''


def main():
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ is required for the extracted C++ helper checks")
    baseline_hash = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    source = SOURCE.read_text(encoding="utf-8")
    program = MODEL + region(source, "HOST_PLAN") + region(source, "ROW_MAX") + CHECKS
    # Keep generated files in the workspace; do not modify the passed source.
    with tempfile.TemporaryDirectory(prefix="check917_", dir=ROOT / "tests") as folder:
        path = Path(folder)
        cpp = path / "check.cpp"
        cpp.write_text(program, encoding="utf-8")
        for tree, balanced, folded in [(1, 1, 1), (1, 0, 1), (0, 1, 1),
                                       (0, 0, 1), (1, 1, 0)]:
            exe = path / "check.exe"
            subprocess.run([compiler, "-std=c++11", "-O2",
                            f"-DBMMS_TREE_MAX={tree}", f"-DBMMS_BALANCED_N={balanced}",
                            f"-DBMMS_FOLDED_MAX={folded}", str(cpp), "-o", str(exe)],
                           check=True)
            print(f"TREE={tree} BALANCED_N={balanced} FOLDED={folded}", flush=True)
            subprocess.run([str(exe)], check=True)
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == baseline_hash
    print("Original kernel_917.asc SHA-256:", baseline_hash)


if __name__ == "__main__":
    main()
