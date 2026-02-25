# G-SLAM FPGA — Gauss-Jordan Mixed Precision (SYCL/oneAPI)

Mixed precision SYCL variant: FP32 for the FPGA matrix inversion kernel (`inv_parallel`), FP64 for all other operations.

## Motivation

Pure FP32 Gauss-Jordan diverges on cityTrees800 (κ ≈ 10⁵). This mixed variant runs the FP32 `inv_parallel` FPGA kernel (faster, lower DSP cost) while keeping FP64 for edge assembly and pose updates. Error ~3×10⁻⁴.

## How It Works

1. Build Ω (FP64 CPU), assemble ξ (FP64)
2. Cast Ω → float, call `inv_parallel()` on FPGA (FP32 Gauss-Jordan, SYCL)
3. Copy result back, cast to FP64, compute Δx = Ω⁻¹ · ξ (FP64 CPU dot)

## Build Targets

```bash
make cpu_test   # Test on CPU (icpx -fsycl, default selector)
make gpu_test   # Test on GPU (icpx -fsycl -DUSE_GPU)
make emu        # FPGA emulator
make s10        # Stratix 10 SX hardware synthesis
make a10        # Arria 10 GX hardware synthesis
make x86        # Pure C build (gcc, no SYCL)
```

## Run

```bash
./GSLAM.cpu_test 0 0.1 10 0 ../../data/cityTrees800.txt
./GSLAM.s10 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Related

- `../fpga_gj_fp32/` — FP32 throughout, paper results (MCSoC 2023)
- `../fpga_gj_fp64/` — FP64 throughout
- `../cpu_gj_mixed/` — same strategy on CPU
- `../gpu_gj_mixed/` — same strategy on GPU (CUDA)
