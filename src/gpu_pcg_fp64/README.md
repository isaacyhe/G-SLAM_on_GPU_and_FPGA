# G-SLAM GPU — Sparse PCG FP64 (cuSPARSE + cuBLAS)

Sparse PCG solver in FP64 using CUDA cuSPARSE for SpMV and cuBLAS for vector operations.

## Algorithm

- **CSR matrix**: built on CPU from edges, transferred to GPU once per G-SLAM iteration
- **SpMV** `q = Ω·p`: `cusparseSpMV` with `CUDA_R_64F`
- **Vector ops** (dot, axpy, scal, nrm2): `cublasDdot`, `cublasDaxpy`, `cublasDscal`, `cublasDnrm2`
- **Block-Jacobi preconditioning**: custom CUDA kernel, 1 thread block per 3×3 diagonal block
- **Convergence**: ‖r‖ < 10⁻⁶·‖b‖, max 500 iterations

## Build

```bash
make cuda
# or manually:
/usr/local/cuda/bin/nvcc -O3 -o GSLAM.cuda *.c *.cu \
    -I/usr/include/python3.12 -L/usr/lib/x86_64-linux-gnu \
    -lpython3.12 -lcrypt -lpthread -ldl -lutil -lm -lcusparse -lcublas
```

## Run

```bash
./GSLAM.cuda 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Related

- `../gpu_pcg_fp32/` — FP32 version (~20% faster, same convergence with block-Jacobi)
- `../cpu_pcg_fp64/` — same algorithm, CPU-only
- `../fpga_pcg_fp64/` — same algorithm, FPGA SYCL
