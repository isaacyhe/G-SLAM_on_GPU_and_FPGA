# G-SLAM CPU — Gauss-Jordan FP32

Single-threaded CPU implementation using Gauss-Jordan elimination in FP32.

## Algorithm

- Dense GJ elimination: N sequential pivot steps, each O(N²) row operations
- Information matrix Ω built in-place; augmented with identity for inversion
- FP32 throughout — diverges on large datasets where κ ≈ 10⁵

## Precision Note

FP32 GJ diverges on cityTrees800 (condition number κ ≈ 10⁵). Use `../cpu_gj_fp64/` for correctness on large problems, or `../cpu_pcg_fp32/` for sparse iterative solving.

## Build

```bash
make x86
# or:
gcc -O3 *.c -lm -o GSLAM
```

## Run

```bash
./GSLAM 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Related

- `../cpu_gj_fp64/` — FP64 version (accurate for all sizes)
- `../cpu_gj_mixed/` — FP32 inversion, FP64 elsewhere
- `../gpu_gj_fp32/` — same algorithm, CUDA
