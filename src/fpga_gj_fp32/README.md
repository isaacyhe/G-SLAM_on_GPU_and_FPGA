# G-SLAM FPGA — Gauss-Jordan FP32 (SYCL/oneAPI)

SYCL/oneAPI implementation of Gauss-Jordan elimination in FP32, targeting Intel FPGAs. From the MCSoC 2023 paper.

## Algorithm

- Dense GJ elimination: N sequential pivot steps, each with a SYCL `parallel_for`
- FP32 throughout — diverges on large datasets where κ ≈ 10⁵
- Device selector: `FPGA_HARDWARE` / `FPGA_EMULATOR` / `USE_GPU` / CPU fallback

## Build

```bash
# CPU (no FPGA required)
make cpu_test
./GSLAM.cpu_test 0 0.1 10 0 ../../data/cityTrees800.txt

# FPGA emulator
make emu

# FPGA hardware synthesis (hours, requires board)
make s10    # Intel Stratix 10 SX
make a10    # Intel Arria 10 GX
```

## Run

```bash
./GSLAM.cpu_test 0 0.1 10 0 ../../data/cityTrees800.txt
```

Set `LD_LIBRARY_PATH` for the SYCL runtime:
```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/umf/1.0/lib
```

## Related

- `../fpga_gj_fp64/` — FP64 version
- `../fpga_gj_mixed/` — FP32 inversion, FP64 elsewhere
- `../gpu_gj_fp32/` — same algorithm, CUDA
- `../cpu_gj_fp32/` — same algorithm, CPU-only
