# G-SLAM FPGA — Gauss-Jordan FP64 (SYCL/oneAPI)

SYCL/oneAPI implementation of Gauss-Jordan elimination in FP64, targeting Intel FPGAs.

## Algorithm

- Dense GJ elimination: N sequential pivot steps, each with a SYCL `parallel_for`
- FP64 throughout — accurate for all dataset sizes tested
- Device selector: `FPGA_HARDWARE` / `FPGA_EMULATOR` / `USE_GPU` / CPU fallback

## Build

```bash
# CPU (no FPGA required)
make cpu_test

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

- `../fpga_gj_fp32/` — FP32 version (paper results)
- `../fpga_gj_mixed/` — FP32 inversion, FP64 elsewhere
- `../gpu_gj_fp64/` — same algorithm, CUDA
- `../cpu_gj_fp64/` — same algorithm, CPU-only
