# G-SLAM GPU — Gauss-Jordan FP32 (CUDA)

CUDA implementation of Gauss-Jordan elimination in FP32.

## Algorithm

- Dense GJ elimination: N sequential pivot steps
- Each pivot: ratio kernel (N threads) + row-elimination kernel (N×N threads)
- FP32 throughout — diverges on large datasets where κ ≈ 10⁵
- `norm()` renamed to `vec_norm()` to avoid conflict with CUDA built-in

## Precision Note

FP32 GJ diverges on cityTrees800. Use `../gpu_gj_fp64/` for accuracy, or `../gpu_gj_mixed/` for a FP32-inversion / FP64-rest tradeoff.

## Build

```bash
make cuda
# or:
/usr/local/cuda/bin/nvcc -DNDEBUG -fmad=false -g -O3 -o GSLAM.cuda *.c *.cu \
    -I/usr/include/python3.12 -L/usr/lib/x86_64-linux-gnu \
    -lpython3.12 -lcrypt -lpthread -ldl -lutil -lm
```

## Run

```bash
./GSLAM.cuda 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Related

- `../gpu_gj_fp64/` — FP64 version (paper results)
- `../gpu_gj_mixed/` — FP32 inversion, FP64 elsewhere
- `../cpu_gj_fp32/` — same algorithm, CPU-only
- `../fpga_gj_fp32/` — same algorithm, SYCL
