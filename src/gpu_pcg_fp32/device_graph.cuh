#pragma once
#include <cuda_runtime.h>
#include "Z.h"
#include "HAT_X.h"
#include "U.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * device_graph_init:
 *
 * Called once after read_data.  Builds CSR sparsity structure (row_ptr,
 * col_idx) covering all ObsEdges and MotionEdges, uploads fixed device
 * arrays (sorted obs indices, per-edge j/k/lm arrays, obs data,
 * us nu/omega for motion model).
 *
 * CSR values are float; xi accumulation uses double atomicAdd.
 *
 * Outputs:
 *   *nnz_out  — total non-zeros in CSR
 */
void device_graph_init(
    struct Z     *zlist,
    unsigned int  n_obs,
    struct U     *us,
    unsigned int  n_poses,
    unsigned int  dim,
    double        delta,
    double        lambda,
    const double *mns,   /* mns[4] = {0.19, 0.001, 0.13, 0.2} */
    int          *nnz_out
);

/*
 * device_graph_update:
 *
 * Called each round.
 *   1. Uploads hat_xs to device.
 *   2. Zeroes d_csr_vals (float) and d_xi (double) on device.
 *   3. Launches compute_motion_edges_kernel_f (anchor + MotionEdges).
 *   4. Launches compute_and_scatter_edges_kernel_f (ObsEdges).
 *   5. Synchronizes.
 */
void device_graph_update(
    struct HAT_X *hat_xs,
    unsigned int  n_poses,
    float        *d_csr_vals,  /* device float[nnz]  */
    double       *d_xi         /* device double[dim] */
);

/* Accessors for persistent device CSR structure pointers */
int *device_graph_get_row_ptr(void);
int *device_graph_get_col_idx(void);

/*
 * device_pcg_solve (fp32):
 *
 * Fully device-resident PCG.  d_csr_vals (float) and d_xi (double) must
 * be the buffers filled by device_graph_update().
 * x_h (float[dim]) receives the solution on return (CPU pointer).
 * No cudaMalloc/cudaFree per call; uses persistent buffers from init.
 */
void device_pcg_solve(float  *x_h,
                      float  *d_csr_vals,
                      double *d_xi,
                      int     dim,
                      int     nnz,
                      int     max_iter);

/* Free all persistent device memory */
void device_graph_free(void);

#ifdef __cplusplus
}
#endif
