#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include "pcg.h"

/* Cast double -> float element-wise on device */
__global__ void double_to_float_kernel(const double *src, float *dst, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = (float)src[tid];
}

/* Block-Jacobi preconditioner apply kernel (FP32).
   diag_inv: (dim/3) * 9 floats — the 3x3 block inverses, stored row-major.
   r:        dim floats — input residual vector.
   z:        dim floats — output preconditioned vector. */
__global__ void block_jacobi_apply(const float *diag_inv, const float *r,
                                   float *z, int n_blocks)
{
    int blk = blockDim.x * blockIdx.x + threadIdx.x;
    if (blk >= n_blocks) return;

    int base = blk * 3;
    const float *inv = diag_inv + blk * 9;

    z[base+0] = inv[0]*r[base+0] + inv[1]*r[base+1] + inv[2]*r[base+2];
    z[base+1] = inv[3]*r[base+0] + inv[4]*r[base+1] + inv[5]*r[base+2];
    z[base+2] = inv[6]*r[base+0] + inv[7]*r[base+1] + inv[8]*r[base+2];
}

void pcg_solve_cuda(float *x, int *row_ptr, int *col_idx, float *csr_vals,
                    float *diag_inv, float *b, int dim, int nnz, int max_iter)
{
    cusparseHandle_t sp_handle;
    cublasHandle_t   bl_handle;
    cusparseCreate(&sp_handle);
    cublasCreate(&bl_handle);

    /* ---- Allocate device memory ---- */
    float *d_x, *d_r, *d_z, *d_p, *d_q;
    float *d_vals, *d_b, *d_diag_inv;
    int   *d_row_ptr, *d_col_idx;

    cudaMalloc((void**)&d_x,        dim * sizeof(float));
    cudaMalloc((void**)&d_r,        dim * sizeof(float));
    cudaMalloc((void**)&d_z,        dim * sizeof(float));
    cudaMalloc((void**)&d_p,        dim * sizeof(float));
    cudaMalloc((void**)&d_q,        dim * sizeof(float));
    cudaMalloc((void**)&d_b,        dim * sizeof(float));
    cudaMalloc((void**)&d_vals,     nnz * sizeof(float));
    cudaMalloc((void**)&d_row_ptr,  (dim + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx,  nnz * sizeof(int));
    cudaMalloc((void**)&d_diag_inv, (dim / 3) * 9 * sizeof(float));

    /* ---- Copy inputs to device ---- */
    cudaMemcpy(d_b,        b,        dim * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals,     csr_vals, nnz * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr,  row_ptr,  (dim + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx,  col_idx,  nnz * sizeof(int),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_diag_inv, diag_inv, (dim / 3) * 9 * sizeof(float), cudaMemcpyHostToDevice);

    /* ---- Initialize x = 0, r = b ---- */
    float zero_f = 0.0f, one_f = 1.0f, neg_one_f = -1.0f;
    cudaMemset(d_x, 0, dim * sizeof(float));
    cudaMemcpy(d_r, d_b, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    /* ---- Set up cuSPARSE SpMV descriptor ---- */
    cusparseSpMatDescr_t mat_A;
    cusparseDnVecDescr_t vec_p, vec_q;

    cusparseCreateCsr(&mat_A, dim, dim, nnz,
                      d_row_ptr, d_col_idx, d_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vec_p, dim, d_p, CUDA_R_32F);
    cusparseCreateDnVec(&vec_q, dim, d_q, CUDA_R_32F);

    /* Workspace for SpMV */
    size_t spmv_buffer_size = 0;
    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &one_f, mat_A, vec_p, &zero_f, vec_q,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                            &spmv_buffer_size);
    void *spmv_buffer = NULL;
    if (spmv_buffer_size > 0)
        cudaMalloc(&spmv_buffer, spmv_buffer_size);

    /* ---- Preconditioner launch config ---- */
    int n_blocks = dim / 3;
    int threads = 128;
    int blocks  = (n_blocks + threads - 1) / threads;

    /* ---- z = M_inv * r ---- */
    block_jacobi_apply<<<blocks, threads>>>(d_diag_inv, d_r, d_z, n_blocks);

    /* ---- p = z ---- */
    cudaMemcpy(d_p, d_z, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    /* ---- rz_old = r^T z ---- */
    float rz_old = 0.0f;
    cublasSdot(bl_handle, dim, d_r, 1, d_z, 1, &rz_old);

    /* ---- b_norm for relative tolerance ---- */
    float b_norm = 0.0f;
    cublasSnrm2(bl_handle, dim, d_b, 1, &b_norm);
    float tol = 1e-6f * b_norm;

    /* ---- PCG main loop ---- */
    for (int k = 0; k < max_iter; k++) {
        /* q = A * p */
        cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one_f, mat_A, vec_p, &zero_f, vec_q,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer);

        /* pq = p^T q */
        float pq = 0.0f;
        cublasSdot(bl_handle, dim, d_p, 1, d_q, 1, &pq);
        if (pq == 0.0f) break;

        float alpha = rz_old / pq;

        /* x += alpha * p */
        cublasSaxpy(bl_handle, dim, &alpha, d_p, 1, d_x, 1);

        /* r -= alpha * q */
        float neg_alpha = -alpha;
        cublasSaxpy(bl_handle, dim, &neg_alpha, d_q, 1, d_r, 1);

        /* Check convergence: ||r|| < tol */
        float r_norm = 0.0f;
        cublasSnrm2(bl_handle, dim, d_r, 1, &r_norm);
        if (r_norm < tol) break;

        /* z = M_inv * r */
        block_jacobi_apply<<<blocks, threads>>>(d_diag_inv, d_r, d_z, n_blocks);

        /* rz_new = r^T z */
        float rz_new = 0.0f;
        cublasSdot(bl_handle, dim, d_r, 1, d_z, 1, &rz_new);

        float beta = rz_new / rz_old;

        /* p = z + beta * p  (scal then axpy: p = beta*p, p += z) */
        cublasSscal(bl_handle, dim, &beta, d_p, 1);
        cublasSaxpy(bl_handle, dim, &one_f, d_z, 1, d_p, 1);

        rz_old = rz_new;
    }

    /* ---- Copy result back to host ---- */
    cudaMemcpy(x, d_x, dim * sizeof(float), cudaMemcpyDeviceToHost);

    /* ---- Cleanup ---- */
    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_p);
    cusparseDestroyDnVec(vec_q);

    if (spmv_buffer) cudaFree(spmv_buffer);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_z);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_vals);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_diag_inv);

    cusparseDestroy(sp_handle);
    cublasDestroy(bl_handle);
}

/*
 * pcg_solve_cuda_d (FP32 variant):
 *
 * Device-resident variant.  row_ptr_d, col_idx_d, csr_vals_d (float) and
 * xi_d (double) are already on device.  diag_inv_h (float) is on CPU.
 * xi_d is converted to float on-device before the PCG solve.
 * The float solution is written back to x_h.
 */
void pcg_solve_cuda_d(float  *x_h,
                      int    *row_ptr_d,
                      int    *col_idx_d,
                      float  *csr_vals_d,
                      float  *diag_inv_h,
                      double *xi_d,
                      int     dim,
                      int     nnz,
                      int     max_iter)
{
    int n_blocks = dim / 3;
    int threads  = 128;
    int grid_b   = (n_blocks + threads - 1) / threads;
    int grid_dim = (dim + threads - 1) / threads;

    /* Allocate working vectors */
    float *x_d, *r_d, *z_d, *p_d, *q_d, *b_d, *diag_inv_d;

    cudaMalloc((void**)&x_d,        dim      * sizeof(float));
    cudaMalloc((void**)&r_d,        dim      * sizeof(float));
    cudaMalloc((void**)&z_d,        dim      * sizeof(float));
    cudaMalloc((void**)&p_d,        dim      * sizeof(float));
    cudaMalloc((void**)&q_d,        dim      * sizeof(float));
    cudaMalloc((void**)&b_d,        dim      * sizeof(float));
    cudaMalloc((void**)&diag_inv_d, n_blocks * 9 * sizeof(float));

    /* Upload preconditioner */
    cudaMemcpy(diag_inv_d, diag_inv_h,
               n_blocks * 9 * sizeof(float), cudaMemcpyHostToDevice);

    /* Convert xi (double on device) -> b_d (float on device) */
    double_to_float_kernel<<<grid_dim, threads>>>(xi_d, b_d, dim);

    /* x = 0, r = b */
    cudaMemset(x_d, 0, dim * sizeof(float));
    cudaMemcpy(r_d, b_d, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    /* cuSPARSE and cuBLAS handles */
    cusparseHandle_t sp_handle;
    cublasHandle_t   bl_handle;
    cusparseCreate(&sp_handle);
    cublasCreate(&bl_handle);

    cusparseSpMatDescr_t mat_A;
    cusparseDnVecDescr_t vec_p, vec_q;

    cusparseCreateCsr(&mat_A, dim, dim, nnz,
                      row_ptr_d, col_idx_d, csr_vals_d,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vec_p, dim, p_d, CUDA_R_32F);
    cusparseCreateDnVec(&vec_q, dim, q_d, CUDA_R_32F);

    float one_f = 1.0f, zero_f = 0.0f;
    size_t spmv_buf_size = 0;
    cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &one_f, mat_A, vec_p, &zero_f, vec_q,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                            &spmv_buf_size);
    void *spmv_buf = NULL;
    if (spmv_buf_size > 0) cudaMalloc(&spmv_buf, spmv_buf_size);

    /* z = M_inv * r */
    block_jacobi_apply<<<grid_b, threads>>>(diag_inv_d, r_d, z_d, n_blocks);

    /* p = z */
    cudaMemcpy(p_d, z_d, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    /* rz_old = dot(r, z) */
    float rz_old = 0.0f;
    cublasSdot(bl_handle, dim, r_d, 1, z_d, 1, &rz_old);

    float b_norm = 0.0f;
    cublasSnrm2(bl_handle, dim, b_d, 1, &b_norm);
    float tol = 1e-6f * b_norm;

    for (int k = 0; k < max_iter; k++) {
        cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one_f, mat_A, vec_p, &zero_f, vec_q,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf);

        float pq = 0.0f;
        cublasSdot(bl_handle, dim, p_d, 1, q_d, 1, &pq);
        if (pq == 0.0f) break;
        float alpha = rz_old / pq;

        cublasSaxpy(bl_handle, dim, &alpha, p_d, 1, x_d, 1);

        float neg_alpha = -alpha;
        cublasSaxpy(bl_handle, dim, &neg_alpha, q_d, 1, r_d, 1);

        float r_norm = 0.0f;
        cublasSnrm2(bl_handle, dim, r_d, 1, &r_norm);
        if (r_norm < tol) break;

        block_jacobi_apply<<<grid_b, threads>>>(diag_inv_d, r_d, z_d, n_blocks);

        float rz_new = 0.0f;
        cublasSdot(bl_handle, dim, r_d, 1, z_d, 1, &rz_new);
        float beta = rz_new / rz_old;

        cublasSscal(bl_handle, dim, &beta, p_d, 1);
        cublasSaxpy(bl_handle, dim, &one_f, z_d, 1, p_d, 1);

        rz_old = rz_new;
    }

    cudaMemcpy(x_h, x_d, dim * sizeof(float), cudaMemcpyDeviceToHost);

    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_p);
    cusparseDestroyDnVec(vec_q);
    if (spmv_buf) cudaFree(spmv_buf);
    cudaFree(x_d); cudaFree(r_d); cudaFree(z_d);
    cudaFree(p_d); cudaFree(q_d); cudaFree(b_d);
    cudaFree(diag_inv_d);
    cusparseDestroy(sp_handle);
    cublasDestroy(bl_handle);
}
