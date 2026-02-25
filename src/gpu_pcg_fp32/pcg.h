#ifndef PCG_H
#define PCG_H
#ifdef __cplusplus
extern "C" {
#endif

/* Original: all arrays on CPU, function handles GPU transfer */
void pcg_solve_cuda(float *x, int *row_ptr, int *col_idx, float *csr_vals,
                    float *diag_inv, float *b, int dim, int nnz, int max_iter);

/* Device-resident variant: row_ptr_d, col_idx_d, csr_vals_d (float) and
   xi_d (double) are already on device.  diag_inv_h (float) is on CPU.
   x_h (float) receives the solution.
   xi_d is converted to float internally before the PCG solve. */
void pcg_solve_cuda_d(float  *x_h,
                      int    *row_ptr_d,
                      int    *col_idx_d,
                      float  *csr_vals_d,
                      float  *diag_inv_h,
                      double *xi_d,
                      int     dim,
                      int     nnz,
                      int     max_iter);

#ifdef __cplusplus
}
#endif
#endif
