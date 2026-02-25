# G-SLAM CPU — Gauss-Jordan Mixed Precision

Mixed precision variant: FP32 for matrix inversion, FP64 for all other operations.

## Motivation

Pure FP32 Gauss-Jordan diverges on cityTrees800 because the information matrix has condition number κ ≈ 10⁵. This mixed variant keeps FP64 precision everywhere except the inversion step itself, reducing error to ~3×10⁻⁴ while cutting inversion cost by ~20–25%.

## How It Works

1. Build Ω (FP64), assemble ξ (FP64) — same as `cpu_gj_fp64`
2. Cast Ω → float, call `inv_float()` (Gauss-Jordan in FP32)
3. Cast result back to FP64, compute Δx = Ω⁻¹ · ξ (FP64 dot product)

## Build

```bash
make        # gcc -O3
make x86    # same
```

## Run

```bash
./GSLAM 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Accuracy vs cpu_gj_fp64

The FP32 inversion introduces ~3×10⁻⁴ relative error in Δx per iteration. For most robotics applications this is acceptable. For the highest accuracy, use `../cpu_gj_fp64/`.

## Related

- `../cpu_gj_fp64/` — FP64 throughout (reference)
- `../cpu_gj_fp32/` — FP32 throughout (diverges on large data)
- `../gpu_gj_mixed/` — same strategy on GPU
- `../fpga_gj_mixed/` — same strategy on FPGA
