# G-SLAM GPU — Gauss-Jordan FP64 (CUDA)

CUDA implementation of Gauss-Jordan elimination in FP64. This is the primary GPU implementation from the HPC Asia 2023 paper.

## Algorithm

- Dense GJ elimination: N sequential pivot steps
- Each pivot: ratio kernel (N threads) + row-elimination kernel (N×N threads)
- FP64 throughout — accurate for all dataset sizes tested
- `norm()` renamed to `vec_norm()` to avoid conflict with CUDA built-in

## Build

```bash
make cuda
# or:
/usr/local/cuda/bin/nvcc -DNDEBUG -g -O3 -o GSLAM.cuda *.c *.cu \
    -I/usr/include/python3.12 -L/usr/lib/x86_64-linux-gnu \
    -lpython3.12 -lcrypt -lpthread -ldl -lutil -lm
```

## Run

```bash
./GSLAM.cuda 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Related

- `../gpu_gj_fp32/` — FP32 version (faster, diverges on large data)
- `../gpu_gj_mixed/` — FP32 inversion, FP64 elsewhere
- `../cpu_gj_fp64/` — same algorithm, CPU-only
- `../fpga_gj_fp64/` — same algorithm, SYCL
