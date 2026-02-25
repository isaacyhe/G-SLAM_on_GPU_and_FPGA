# G-SLAM CPU — Sparse PCG FP64

Sparse Preconditioned Conjugate Gradient solver in FP64, single-threaded C.

## Algorithm

Solves Ω·Δx = ξ iteratively without forming Ω⁻¹:

- **Ω in CSR format**: built directly from edges, never allocates the full 3N×3N dense matrix
- **Block-Jacobi preconditioner**: analytically inverts each 3×3 diagonal block of Ω
- **SpMV**: row-wise CSR dot product (CPU loop)
- **Convergence**: ‖r‖ < 10⁻⁶·‖b‖, max 500 iterations

## Build

```bash
make        # gcc -O3
```

## Run

```bash
./GSLAM 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Comparison with GJ

| | cpu_gj_fp64 | cpu_pcg_fp64 |
|--|--|--|
| Memory | O(N²) dense Ω | O(nnz) CSR |
| Per-iter work | O(N³) inversion | O(nnz·iters) |
| Parallelism | sequential pivots | each SpMV is independent |

For cityTrees800, PCG typically converges in ~20–50 iterations. Total work is comparable to GJ at this scale, but PCG scales much better for larger N.

## Related

- `../cpu_pcg_fp32/` — FP32 version
- `../gpu_pcg_fp64/` — same algorithm on GPU with cuSPARSE
- `../fpga_pcg_fp64/` — same algorithm on FPGA with SYCL
