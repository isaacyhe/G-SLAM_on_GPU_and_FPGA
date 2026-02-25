# G-SLAM CPU — Sparse PCG FP32

Sparse Preconditioned Conjugate Gradient solver in FP32, single-threaded C.

## Algorithm

Solves Ω·Δx = ξ iteratively without forming Ω⁻¹:

- **Ω in CSR format**: built directly from edges, never allocates the full 3N×3N dense matrix
- **Block-Jacobi preconditioner**: analytically inverts each 3×3 diagonal block of Ω, reducing κ from ~10⁵ to ~10³
- **SpMV**: row-wise CSR dot product (CPU loop)
- **Convergence**: ‖r‖ < 10⁻⁶·‖b‖, max 500 iterations

## Why FP32 Works Here

Unlike Gauss-Jordan, PCG with block-Jacobi preconditioning tolerates FP32 because the preconditioner reduces the condition number enough for CG to converge. ξ is accumulated in FP64 before casting to FP32 for the solve.

## Build

```bash
make        # gcc -O3
```

## Run

```bash
./GSLAM 0 0.1 10 0 ../../data/cityTrees800.txt
```

## Memory

Peak memory: O(nnz) for CSR + O(N) for preconditioner + O(N) for 4 work vectors.
For cityTrees800 (N=800, dim=2400): CSR ≈ 200–400 KB vs 44 MB for dense Ω.

## Related

- `../cpu_pcg_fp64/` — FP64 version (higher accuracy)
- `../gpu_pcg_fp32/` — same algorithm on GPU with cuSPARSE
- `../fpga_pcg_fp32/` — same algorithm on FPGA with SYCL
