#ifndef PCG_H
#define PCG_H

#ifdef __cplusplus
extern "C" {
#endif

// Sparse PCG solver using cuSPARSE + cuBLAS
// Solves: Omega * x = b  where Omega is stored in CSR format
// All arrays are on CPU; function handles GPU transfer internally
void pcg_solve_cuda(double *x,            // output: solution vector (CPU, dim)
                    int    *row_ptr,       // CSR row pointers (CPU, dim+1)
                    int    *col_idx,       // CSR column indices (CPU, nnz)
                    double *csr_vals,      // CSR values (CPU, nnz)
                    double *diag_inv,      // block-Jacobi preconditioner (CPU, N*9)
                    double *b,             // RHS vector (CPU, dim)
                    int     dim,
                    int     nnz,
                    int     max_iter);

// Device-resident variant: CSR (row_ptr_d, col_idx_d, csr_vals_d) and xi (xi_d)
// are already on device.  diag_inv_h is still on CPU and uploaded internally.
// x_h receives the solution on return (CPU).
void pcg_solve_cuda_d(double *x_h,
                      int    *row_ptr_d,
                      int    *col_idx_d,
                      double *csr_vals_d,
                      double *diag_inv_h,
                      double *xi_d,
                      int     dim,
                      int     nnz,
                      int     max_iter);

#ifdef __cplusplus
}
#endif
#endif
