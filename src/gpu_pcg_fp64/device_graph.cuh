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
 * Called once after read_data.  Analyses zlist to enumerate all unique
 * (landmark, obs_j, obs_k) edge pairs, builds the CSR sparsity structure
 * (row_ptr, col_idx) that covers both ObsEdges and MotionEdges, uploads
 * all fixed device arrays including motion model parameters.
 *
 * Outputs:
 *   *nnz_out       - total number of non-zeros in the CSR matrix
 *   (device pointers are stored internally in static variables)
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
 *   2. Zeroes d_csr_vals and d_xi on device.
 *   3. Launches compute_motion_edges_kernel (anchor + all MotionEdges).
 *   4. Launches compute_and_scatter_edges_kernel (ObsEdges).
 *   5. Synchronizes.
 *
 * After this call d_csr_vals and d_xi are ready for device_pcg_solve.
 */
void device_graph_update(
    struct HAT_X *hat_xs,
    unsigned int  n_poses,
    double       *d_csr_vals,  /* device [nnz]  */
    double       *d_xi         /* device [dim]  */
);

/*
 * device_graph_get_row_ptr / device_graph_get_col_idx:
 *
 * Accessors for the persistent device CSR structure pointers.
 */
int *device_graph_get_row_ptr(void);
int *device_graph_get_col_idx(void);

/*
 * device_build_diag_inv:
 *
 * Computes the block-Jacobi preconditioner entirely on device.
 * Reads d_csr_vals from the last device_graph_update() call.
 * dim must equal the value passed to device_graph_init().
 */
void device_build_diag_inv(int dim);

/*
 * device_pcg_solve:
 *
 * Fully device-resident PCG solve.  Uses persistent buffers allocated
 * in device_graph_init() — no cudaMalloc/cudaFree per call.
 * Reads d_csr_vals and d_xi from the last device_graph_update() call
 * and the diag_inv from the last device_build_diag_inv() call.
 * Writes the solution to x_h (CPU) via a single cudaMemcpy at the end.
 *
 * d_csr_vals and d_xi pointers passed in must be the same buffers
 * that device_graph_update() wrote into.
 */
void device_pcg_solve(double *x_h, double *d_csr_vals, double *d_xi,
                      int dim, int nnz, int max_iter);

/*
 * device_graph_free:
 *
 * Frees all persistent device allocations.
 */
void device_graph_free(void);

#ifdef __cplusplus
}
#endif
