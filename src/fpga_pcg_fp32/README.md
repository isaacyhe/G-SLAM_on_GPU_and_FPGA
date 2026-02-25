# G-SLAM FPGA — Sparse PCG FP32 (SYCL/oneAPI)

Sparse PCG solver in FP32 with the SpMV kernel offloaded to FPGA via SYCL/oneAPI.

## Algorithm

- **CSR matrix**: built on CPU from edges
- **SpMV** `q = Ω·p`: SYCL `parallel_for` over rows — each work-item computes one row of q using CSR. Annotated with `[[intel::scheduler_target_fmax_mhz(400)]]` for FPGA synthesis.
- **All other PCG ops** (dot, axpy, convergence check, preconditioner apply): on CPU host
- **Block-Jacobi preconditioner**: computed on CPU, applied per iteration
- **Convergence**: ‖r‖ < 10⁻⁶·‖b‖, max 500 iterations

## Build Targets

```bash
make cpu_test   # Test on CPU (icpx -fsycl, default selector)
make gpu_test   # Test on GPU (icpx -fsycl -DUSE_GPU)
make emu        # FPGA emulator (icpx -fintelfpga -DFPGA_EMULATOR)
make s10        # Stratix 10 SX hardware synthesis
make a10        # Arria 10 GX hardware synthesis
make x86        # Pure C build (gcc, no SYCL)
```

## Run

```bash
# After cpu_test build:
./GSLAM.cpu_test 0 0.1 10 0 ../../data/cityTrees800.txt

# After FPGA hardware build:
./GSLAM.s10 0 0.1 10 0 ../../data/cityTrees800.txt
```

## FPGA Design Notes

- The streaming SpMV kernel (`spmv_parallel`) is the only FPGA-offloaded kernel — PCG scalar ops on host avoid round-trip overhead for small vectors
- `[[intel::scheduler_target_fmax_mhz(400)]]` targets 400 MHz for Stratix 10 / Arria 10
- FP32 values: CSR `float *values`, `float *p`, `float *q`

## Related

- `../fpga_pcg_fp64/` — FP64 version
- `../cpu_pcg_fp32/` — CPU reference (same algorithm, no SYCL)
- `../gpu_pcg_fp32/` — GPU version using cuSPARSE
- `../fpga_gj_fp32/` — FPGA GJ (paper results, MCSoC 2023)
