# G-SLAM FPGA — Sparse PCG FP64 (SYCL/oneAPI)

Sparse PCG solver in FP64 with the SpMV kernel offloaded to FPGA via SYCL/oneAPI.

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

## Related

- `../fpga_pcg_fp32/` — FP32 version (~20% faster on FPGA)
- `../cpu_pcg_fp64/` — CPU reference (same algorithm, no SYCL)
- `../gpu_pcg_fp64/` — GPU version using cuSPARSE
- `../fpga_gj_fp64/` — FPGA GJ FP64
