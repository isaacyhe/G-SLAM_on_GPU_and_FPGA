#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "pcg.h"

/*
 * apply_block_jacobi:
 *
 * CUDA kernel that applies the block-Jacobi preconditioner: z = M_inv * r
 *
 * Each thread block handles one 3x3 diagonal block of the preconditioner.
 * Thread block blk processes DOFs [blk*3, blk*3+1, blk*3+2].
 *
 * diag_inv layout: diag_inv[blk*9 .. blk*9+8] stores the row-major 3x3
 * inverse of the blk-th diagonal block.
 */
__global__ void apply_block_jacobi(double *z, const double *r,
                                    const double *diag_inv, int n_blocks) {
    int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    int base = blk * 3;
    const double *inv = diag_inv + blk * 9;
    for (int row = 0; row < 3; row++) {
        double sum = 0.0;
        for (int col = 0; col < 3; col++)
            sum += inv[row*3+col] * r[base+col];
        z[base+row] = sum;
    }
}

/*
 * pcg_solve_cuda:
 *
 * Preconditioned Conjugate Gradient solver using cuSPARSE SpMV and cuBLAS
 * vector operations.  Solves: Omega * x = b.
 *
 * Algorithm (standard PCG):
 *   x = 0,  r = b
 *   z = M_inv * r  (block-Jacobi preconditioning)
 *   p = z
 *   rz_old = dot(r, z)
 *   for k = 0 .. max_iter-1:
 *     q = Omega * p          (sparse matrix-vector product via cuSPARSE)
 *     pq = dot(p, q)
 *     alpha = rz_old / pq
 *     x += alpha * p
 *     r -= alpha * q
 *     if ||r|| < tol: break
 *     z = M_inv * r
 *     rz_new = dot(r, z)
 *     beta = rz_new / rz_old
 *     p = z + beta * p
 *     rz_old = rz_new
 *
 * All CSR data and vectors are transferred to the GPU internally.
 * The solution is copied back to x_h on return.
 */
void pcg_solve_cuda(double *x_h, int *row_ptr_h, int *col_idx_h,
                     double *csr_vals_h, double *diag_inv_h, double *b_h,
                     int dim, int nnz, int max_iter) {

    int n_blocks = dim / 3;

    /* --- Allocate device memory --- */
    double *x_d, *r_d, *z_d, *p_d, *q_d, *b_d, *diag_inv_d;
    int    *row_ptr_d, *col_idx_d;
    double *csr_vals_d;

    cudaMalloc(&x_d,        dim * sizeof(double));
    cudaMalloc(&r_d,        dim * sizeof(double));
    cudaMalloc(&z_d,        dim * sizeof(double));
    cudaMalloc(&p_d,        dim * sizeof(double));
    cudaMalloc(&q_d,        dim * sizeof(double));
    cudaMalloc(&b_d,        dim * sizeof(double));
    cudaMalloc(&diag_inv_d, n_blocks * 9 * sizeof(double));
    cudaMalloc(&row_ptr_d,  (dim+1) * sizeof(int));
    cudaMalloc(&col_idx_d,  nnz * sizeof(int));
    cudaMalloc(&csr_vals_d, nnz * sizeof(double));

    /* --- Copy inputs to device --- */
    cudaMemcpy(b_d,        b_h,          dim * sizeof(double),      cudaMemcpyHostToDevice);
    cudaMemcpy(diag_inv_d, diag_inv_h,   n_blocks*9*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr_d,  row_ptr_h,    (dim+1)*sizeof(int),       cudaMemcpyHostToDevice);
    cudaMemcpy(col_idx_d,  col_idx_h,    nnz*sizeof(int),           cudaMemcpyHostToDevice);
    cudaMemcpy(csr_vals_d, csr_vals_h,   nnz*sizeof(double),        cudaMemcpyHostToDevice);

    /* --- Initialize x = 0, r = b --- */
    cudaMemset(x_d, 0, dim * sizeof(double));
    cudaMemcpy(r_d, b_d, dim * sizeof(double), cudaMemcpyDeviceToDevice);

    /* --- cuSPARSE setup --- */
    cusparseHandle_t sp_handle;
    cusparseCreate(&sp_handle);

    cusparseSpMatDescr_t mat_A;
    cusparseDnVecDescr_t vec_p, vec_q;

    cusparseCreateCsr(&mat_A, dim, dim, nnz,
                      row_ptr_d, col_idx_d, csr_vals_d,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vec_p, dim, p_d, CUDA_R_64F);
    cusparseCreateDnVec(&vec_q, dim, q_d, CUDA_R_64F);

    /* --- cuBLAS setup --- */
    cublasHandle_t bl_handle;
    cublasCreate(&bl_handle);

    /* --- Compute SpMV buffer size and allocate --- */
    size_t buf_size = 0;
    double alpha_spmv = 1.0, beta_spmv = 0.0;
    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_spmv, mat_A, vec_p, &beta_spmv, vec_q,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);
    void *buf_d = NULL;
    if (buf_size > 0) cudaMalloc(&buf_d, buf_size);

    /* --- Initial preconditioner application: z = M_inv * r --- */
    apply_block_jacobi<<<n_blocks, 1>>>(z_d, r_d, diag_inv_d, n_blocks);

    /* --- p = z --- */
    cudaMemcpy(p_d, z_d, dim*sizeof(double), cudaMemcpyDeviceToDevice);

    /* --- rz_old = dot(r, z) --- */
    double rz_old, rz_new, pq, alpha, beta, b_norm, r_norm;
    cublasDdot(bl_handle, dim, r_d, 1, z_d, 1, &rz_old);
    cublasDnrm2(bl_handle, dim, b_d, 1, &b_norm);
    double tol = 1e-6 * b_norm;

    /* --- PCG iteration --- */
    for (int k = 0; k < max_iter; k++) {

        /* q = Omega * p  (sparse matrix-vector product) */
        cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_spmv, mat_A, vec_p, &beta_spmv, vec_q,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_d);

        /* pq = dot(p, q) */
        cublasDdot(bl_handle, dim, p_d, 1, q_d, 1, &pq);
        if (pq == 0.0) break;
        alpha = rz_old / pq;

        /* x += alpha * p */
        cublasDaxpy(bl_handle, dim, &alpha, p_d, 1, x_d, 1);

        /* r -= alpha * q */
        double neg_alpha = -alpha;
        cublasDaxpy(bl_handle, dim, &neg_alpha, q_d, 1, r_d, 1);

        /* check convergence */
        cublasDnrm2(bl_handle, dim, r_d, 1, &r_norm);
        if (r_norm < tol) break;

        /* z = M_inv * r */
        apply_block_jacobi<<<n_blocks, 1>>>(z_d, r_d, diag_inv_d, n_blocks);

        /* rz_new = dot(r, z) */
        cublasDdot(bl_handle, dim, r_d, 1, z_d, 1, &rz_new);
        beta = rz_new / rz_old;

        /* p = z + beta * p  (implemented as: scale p by beta, then axpy z into p) */
        cublasDscal(bl_handle, dim, &beta, p_d, 1);
        cublasDaxpy(bl_handle, dim, &alpha_spmv, z_d, 1, p_d, 1);

        rz_old = rz_new;
    }

    /* --- Copy result back to host --- */
    cudaMemcpy(x_h, x_d, dim * sizeof(double), cudaMemcpyDeviceToHost);

    /* --- Cleanup --- */
    if (buf_d) cudaFree(buf_d);
    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_p);
    cusparseDestroyDnVec(vec_q);
    cusparseDestroy(sp_handle);
    cublasDestroy(bl_handle);

    cudaFree(x_d);
    cudaFree(r_d);
    cudaFree(z_d);
    cudaFree(p_d);
    cudaFree(q_d);
    cudaFree(b_d);
    cudaFree(diag_inv_d);
    cudaFree(row_ptr_d);
    cudaFree(col_idx_d);
    cudaFree(csr_vals_d);
}

/*
 * pcg_solve_cuda_d:
 *
 * Device-resident variant.  row_ptr_d, col_idx_d, csr_vals_d and xi_d are
 * already on device.  diag_inv_h is on CPU and uploaded internally.
 * The solution is copied back to x_h on return.
 *
 * This avoids the per-round CSR upload cost that pcg_solve_cuda incurs.
 */
void pcg_solve_cuda_d(double *x_h,
                      int    *row_ptr_d,
                      int    *col_idx_d,
                      double *csr_vals_d,
                      double *diag_inv_h,
                      double *xi_d,
                      int     dim,
                      int     nnz,
                      int     max_iter)
{
    int n_blocks = dim / 3;

    /* --- Allocate working vectors on device --- */
    double *x_d, *r_d, *z_d, *p_d, *q_d, *diag_inv_d;

    cudaMalloc(&x_d,        dim      * sizeof(double));
    cudaMalloc(&r_d,        dim      * sizeof(double));
    cudaMalloc(&z_d,        dim      * sizeof(double));
    cudaMalloc(&p_d,        dim      * sizeof(double));
    cudaMalloc(&q_d,        dim      * sizeof(double));
    cudaMalloc(&diag_inv_d, n_blocks * 9 * sizeof(double));

    /* Upload preconditioner */
    cudaMemcpy(diag_inv_d, diag_inv_h,
               n_blocks * 9 * sizeof(double), cudaMemcpyHostToDevice);

    /* x = 0,  r = xi */
    cudaMemset(x_d, 0, dim * sizeof(double));
    cudaMemcpy(r_d, xi_d, dim * sizeof(double), cudaMemcpyDeviceToDevice);

    /* --- cuSPARSE setup (using already-on-device CSR) --- */
    cusparseHandle_t sp_handle;
    cusparseCreate(&sp_handle);

    cusparseSpMatDescr_t mat_A;
    cusparseDnVecDescr_t vec_p, vec_q;

    cusparseCreateCsr(&mat_A, dim, dim, nnz,
                      row_ptr_d, col_idx_d, csr_vals_d,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vec_p, dim, p_d, CUDA_R_64F);
    cusparseCreateDnVec(&vec_q, dim, q_d, CUDA_R_64F);

    /* --- cuBLAS setup --- */
    cublasHandle_t bl_handle;
    cublasCreate(&bl_handle);

    /* --- SpMV buffer --- */
    size_t buf_size = 0;
    double alpha_spmv = 1.0, beta_spmv = 0.0;
    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_spmv, mat_A, vec_p, &beta_spmv, vec_q,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);
    void *buf_d = NULL;
    if (buf_size > 0) cudaMalloc(&buf_d, buf_size);

    /* --- Initial preconditioner: z = M_inv * r --- */
    apply_block_jacobi<<<n_blocks, 1>>>(z_d, r_d, diag_inv_d, n_blocks);

    /* p = z */
    cudaMemcpy(p_d, z_d, dim * sizeof(double), cudaMemcpyDeviceToDevice);

    /* rz_old = dot(r, z) */
    double rz_old, rz_new, pq, alpha, beta, b_norm, r_norm;
    cublasDdot(bl_handle, dim, r_d, 1, z_d, 1, &rz_old);
    cublasDnrm2(bl_handle, dim, xi_d, 1, &b_norm);
    double tol = 1e-6 * b_norm;

    /* --- PCG iterations --- */
    for (int k = 0; k < max_iter; k++) {
        cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_spmv, mat_A, vec_p, &beta_spmv, vec_q,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_d);

        cublasDdot(bl_handle, dim, p_d, 1, q_d, 1, &pq);
        if (pq == 0.0) break;
        alpha = rz_old / pq;

        cublasDaxpy(bl_handle, dim, &alpha, p_d, 1, x_d, 1);

        double neg_alpha = -alpha;
        cublasDaxpy(bl_handle, dim, &neg_alpha, q_d, 1, r_d, 1);

        cublasDnrm2(bl_handle, dim, r_d, 1, &r_norm);
        if (r_norm < tol) break;

        apply_block_jacobi<<<n_blocks, 1>>>(z_d, r_d, diag_inv_d, n_blocks);

        cublasDdot(bl_handle, dim, r_d, 1, z_d, 1, &rz_new);
        beta = rz_new / rz_old;

        cublasDscal(bl_handle, dim, &beta, p_d, 1);
        cublasDaxpy(bl_handle, dim, &alpha_spmv, z_d, 1, p_d, 1);

        rz_old = rz_new;
    }

    /* Copy result to host */
    cudaMemcpy(x_h, x_d, dim * sizeof(double), cudaMemcpyDeviceToHost);

    /* Cleanup */
    if (buf_d) cudaFree(buf_d);
    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_p);
    cusparseDestroyDnVec(vec_q);
    cusparseDestroy(sp_handle);
    cublasDestroy(bl_handle);

    cudaFree(x_d);
    cudaFree(r_d);
    cudaFree(z_d);
    cudaFree(p_d);
    cudaFree(q_d);
    cudaFree(diag_inv_d);
}
