# G-SLAM CPU — Gauss-Jordan FP64

Single-threaded CPU implementation using Gauss-Jordan elimination in FP64. This is the primary CPU reference: maximum numerical accuracy, no hardware dependencies.

## Algorithm

- Dense GJ elimination: N sequential pivot steps, each O(N²) row operations
- Information matrix Ω built in-place; augmented with identity for inversion
- FP64 throughout — accurate for all dataset sizes tested

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

- `../cpu_gj_fp32/` — FP32 version (faster, diverges on large data)
- `../cpu_gj_mixed/` — FP32 inversion, FP64 elsewhere
- `../gpu_gj_fp64/` — same algorithm, CUDA
