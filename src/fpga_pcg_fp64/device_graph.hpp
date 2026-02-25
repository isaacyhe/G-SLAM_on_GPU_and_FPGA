#pragma once
/*
 * device_graph.hpp  —  Device-resident edge computation and CSR assembly
 *                      for fpga_pcg_fp64 (SYCL/oneAPI, FP64)
 *
 * DeviceGraph encapsulates:
 *   - Fixed CSR structure (row_ptr, col_idx), built once from input data.
 *   - Sorted observation index arrays, landmark-position arrays, and edge
 *     (j,k) pair arrays that let the device kernel address each ObsEdge.
 *   - Persistent SYCL buffers for all fixed data.
 *   - Per-round buffers for hat_x, csr_vals, and xi that are overwritten
 *     each iteration.
 *
 * Usage:
 *   DeviceGraph dg;
 *   dg.init(zlist, n_obs, n_poses, dim, device_queue);
 *
 *   for each round {
 *       dg.update(hat_xs, us, delta, lambda, mns, device_queue);
 *       // dg.csr_vals_host and dg.xi_host now hold current values
 *       // use them for build_diag_inv and PCG
 *   }
 *
 *   dg.free_all();
 */

#include <sycl/sycl.hpp>
#include "Z.h"
#include "HAT_X.h"
#include "U.h"

struct DeviceGraph {
    /* Dimensions (set by init) */
    int n_poses;        /* number of poses */
    int n_obs;          /* total number of observations in zlist */
    int n_lm;           /* number of distinct active landmarks */
    int n_obs_edges;    /* total number of ObsEdge (j,k) pairs */
    int dim;            /* 3 * n_poses */
    int nnz;            /* number of non-zeros in CSR */

    /* Host arrays — built once in init(), freed in free_all() */
    int    *lm_pos;       /* [n_lm]  start index in sorted_obs for each landmark */
    int    *lm_count;     /* [n_lm]  number of obs per landmark */
    int    *edge_lm;      /* [n_obs_edges]  landmark index for edge i */
    int    *edge_j;       /* [n_obs_edges]  j index within landmark's obs */
    int    *edge_k;       /* [n_obs_edges]  k index within landmark's obs (k>j) */
    int    *sorted_obs;   /* [n_obs]  indices into zlist[], sorted by landmark */
    int    *row_ptr;      /* [dim+1] CSR row pointer array */
    int    *col_idx;      /* [nnz]   CSR column indices (sorted per row) */
    double *z_range;      /* [n_obs]  zlist[i].z[0] */
    double *z_bearing;    /* [n_obs]  zlist[i].z[1] */
    int    *z_step;       /* [n_obs]  zlist[i].step */

    /* Per-round CPU staging arrays (overwritten in update) */
    double *motion_csr_vals; /* [nnz]  MotionEdge + anchor CSR contributions */
    double *motion_xi;       /* [dim]  MotionEdge xi contributions */

    /* Output host arrays — populated by update() after kernel completes */
    double *csr_vals_host;   /* [nnz]  full CSR values (motion + obs) */
    double *xi_host;         /* [dim]  full xi vector (motion + obs) */

    /* Persistent SYCL buffers (fixed data) */
    sycl::buffer<int,1>    *buf_sorted_obs;
    sycl::buffer<int,1>    *buf_edge_lm;
    sycl::buffer<int,1>    *buf_edge_j;
    sycl::buffer<int,1>    *buf_edge_k;
    sycl::buffer<int,1>    *buf_lm_pos;
    sycl::buffer<int,1>    *buf_lm_count;
    sycl::buffer<double,1> *buf_z_range;
    sycl::buffer<double,1> *buf_z_bearing;
    sycl::buffer<int,1>    *buf_z_step;
    sycl::buffer<int,1>    *buf_row_ptr;
    sycl::buffer<int,1>    *buf_col_idx;

    /* Per-round SYCL buffers (overwritten each round) */
    sycl::buffer<double,1> *buf_hat_x;    /* [3*n_poses] — current poses */
    sycl::buffer<double,1> *buf_csr_vals; /* [nnz]       — accumulated Omega */
    sycl::buffer<double,1> *buf_xi;       /* [dim]       — accumulated xi */

    /* Persistent PCG working buffers (allocated in init, reused each round) */
    sycl::buffer<double,1> *buf_diag_inv; /* [n_blocks * 9] */
    sycl::buffer<double,1> *buf_pcg_x;   /* [dim] */
    sycl::buffer<double,1> *buf_pcg_r;   /* [dim] */
    sycl::buffer<double,1> *buf_pcg_z;   /* [dim] */
    sycl::buffer<double,1> *buf_pcg_p;   /* [dim] */
    sycl::buffer<double,1> *buf_pcg_q;   /* [dim] */

    /*
     * init — build all fixed data structures and allocate SYCL buffers.
     */
    void init(struct Z *zlist, unsigned int n_obs_in,
              unsigned int n_poses_in, unsigned int dim_in,
              sycl::queue &q);

    /*
     * update — compute MotionEdge contributions on CPU, launch fused
     *          ObsEdge SYCL kernel, then build diag_inv on device.
     *
     * After this call buf_csr_vals and buf_xi are current on device and
     * buf_diag_inv holds the block-Jacobi preconditioner.
     * csr_vals_host / xi_host are NO LONGER populated by update().
     */
    void update(struct HAT_X *hat_xs,
                struct U *us, double delta, double lambda,
                const double *mns,
                sycl::queue &q);

    /*
     * pcg_solve — run the full PCG loop on device.
     * Uses buf_csr_vals, buf_xi, buf_diag_inv from the last update() call.
     * Writes delta_xs_out (CPU pointer, length dim) on return.
     */
    void pcg_solve(sycl::queue &q, int max_iter, double *delta_xs_out);

    /* free_all — release all host and SYCL allocations */
    void free_all();
};
