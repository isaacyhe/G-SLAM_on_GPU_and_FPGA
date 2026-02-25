# G-SLAM GPU — Gauss-Jordan Mixed Precision (CUDA)

Mixed precision CUDA variant: FP32 for GPU matrix inversion (`inv_cuda`), FP64 for all other operations.

## Motivation

Pure FP32 Gauss-Jordan diverges on cityTrees800 because the information matrix has condition number κ ≈ 10⁵. This mixed variant runs the fast FP32 CUDA inversion kernel while keeping FP64 accuracy for edge assembly and pose updates. Error ~3×10⁻⁴.

## How It Works

1. Build Ω (FP64 CPU), assemble ξ (FP64)
2. Cast Ω → float, call `inv_cuda()` on GPU (FP32 Gauss-Jordan)
3. Copy result back, cast to FP64, compute Δx = Ω⁻¹ · ξ (FP64 CPU dot)

## Build

```bash
make cuda
# or manually:
/usr/local/cuda/bin/nvcc -O3 -o GSLAM.cuda *.c *.cu \
    -I/usr/include/python3.12 -L/usr/lib/x86_64-linux-gnu \
    -lpython3.12 -lcrypt -lpthread -ldl -lutil -lm
```

## Run

```bash
./GSLAM.cuda 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Accuracy

The FP32 inversion introduces ~3×10⁻⁴ relative error in Δx per iteration — acceptable for robotics applications. For full accuracy, use `../gpu_gj_fp64/`.

## Related

- `../gpu_gj_fp64/` — FP64 throughout, paper results (HPC Asia 2023)
- `../gpu_gj_fp32/` — FP32 throughout (diverges on large data)
- `../cpu_gj_mixed/` — same strategy on CPU
- `../fpga_gj_mixed/` — same strategy on FPGA
