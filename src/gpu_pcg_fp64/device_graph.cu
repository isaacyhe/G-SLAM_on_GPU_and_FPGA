/*
 * device_graph.cu  —  device-resident edge computation and CSR assembly
 *                     for gpu_pcg_fp64 (FP64 double precision)
 *
 * Architecture:
 *  - build_csr_structure(): one-time CPU function to derive row_ptr/col_idx
 *    from the complete set of (ObsEdge + MotionEdge) (t1,t2) pairs.
 *  - device_graph_init(): uploads fixed arrays (row_ptr, col_idx, sorted
 *    obs indices, per-edge j/k/lm arrays, obs data, us nu/omega).
 *  - compute_motion_edges_kernel(): one thread per MotionEdge (i=0..n-2).
 *    Computes full motion model inline and atomicAdds into d_csr_vals/d_xi.
 *    Thread 0 also adds the anchor (1e6 on diagonal rows 0,1,2).
 *  - compute_and_scatter_edges_kernel(): fused CUDA kernel — one thread per
 *    ObsEdge.  Computes full ObsEdge math inline and atomicAdds into
 *    d_csr_vals and d_xi.
 *  - device_graph_update(): per-round entry point: uploads hat_xs, zeros
 *    d_csr_vals/d_xi, launches both kernels, syncs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "device_graph.cuh"
#include "Z.h"
#include "HAT_X.h"

/* =========================================================================
 * Compile-time constants
 * ========================================================================= */
#define SNR0 0.14
#define SNR1 0.05

/* =========================================================================
 * Static (persistent across rounds) device and host buffers
 * ========================================================================= */

/* CSR structure (fixed) */
static int *s_d_row_ptr  = NULL;
static int *s_d_col_idx  = NULL;
static int  s_nnz        = 0;
static int  s_dim        = 0;

/* hat_x on device (updated each round) */
static double *s_d_hat_x = NULL;

/* Sorted observation index array (fixed) — zlist indices sorted by landmark */
static int *s_d_sorted_obs = NULL;

/* Per-edge arrays (fixed) */
static int *s_d_edge_j  = NULL;   /* j-index within lm's obs list */
static int *s_d_edge_k  = NULL;   /* k-index within lm's obs list */
static int *s_d_edge_lm = NULL;   /* which landmark this edge belongs to */

/* Per-landmark arrays (fixed) */
static int *s_d_lm_pos   = NULL;  /* start of this lm's obs in sorted_obs */
static int *s_d_lm_count = NULL;  /* #obs for this lm                      */

/* Flat observation data (fixed) */
static double       *s_d_z_range   = NULL;  /* z[0] = range   */
static double       *s_d_z_bearing = NULL;  /* z[1] = bearing */
static int          *s_d_z_step    = NULL;  /* pose step index */

/* Motion model data (fixed, uploaded once) */
static double *s_d_us_nu    = NULL;  /* [n_poses] us[i].nu   */
static double *s_d_us_omega = NULL;  /* [n_poses] us[i].omega */
static double  s_delta;
static double  s_lambda;
static double  s_mns[4];            /* {0.19, 0.001, 0.13, 0.2} */

/* Total edges count */
static int s_n_edges = 0;
static int s_n_poses = 0;

/* =========================================================================
 * Persistent PCG state (allocated once in device_graph_init)
 * ========================================================================= */
static double *s_pcg_x        = NULL;  /* [dim] solution vector            */
static double *s_pcg_r        = NULL;  /* [dim] residual                   */
static double *s_pcg_z        = NULL;  /* [dim] preconditioned residual    */
static double *s_pcg_p        = NULL;  /* [dim] search direction           */
static double *s_pcg_q        = NULL;  /* [dim] A*p                        */
static double *s_pcg_diag_inv = NULL;  /* [n_blocks * 9] block-Jacobi inv  */

static cublasHandle_t       s_bl_handle  = NULL;
static cusparseHandle_t     s_sp_handle  = NULL;
static cusparseSpMatDescr_t s_mat_A      = NULL;
static cusparseDnVecDescr_t s_vec_p      = NULL;
static cusparseDnVecDescr_t s_vec_q      = NULL;
static void                *s_spmv_buf   = NULL;

/* =========================================================================
 * Helper: CUDA error check
 * ========================================================================= */
static void cuda_check(cudaError_t err, const char *where)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where,
                cudaGetErrorString(err));
        exit(1);
    }
}

/* =========================================================================
 * CPU helper: combination C(n,2)
 * ========================================================================= */
static unsigned int comb2(unsigned int n)
{
    return n * (n - 1) / 2;
}

/* =========================================================================
 * CUDA kernel: build_diag_inv_kernel
 *
 * One thread per 3x3 diagonal block.  Reads the 3x3 block from CSR,
 * computes the analytical 3x3 inverse, writes to diag_inv.
 * ========================================================================= */
__global__ void build_diag_inv_kernel(
    const int    *row_ptr,   /* [dim+1]     */
    const int    *col_idx,   /* [nnz]       */
    const double *csr_vals,  /* [nnz]       */
    double       *diag_inv,  /* [n_blocks*9] output */
    int           n_blocks
)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;

    int base = blk * 3;
    double M[9] = {0,0,0, 0,0,0, 0,0,0};

    /* Scan the 3 rows of this block and extract columns [base, base+3) */
    for (int r = 0; r < 3; r++) {
        int row = base + r;
        int lo = row_ptr[row], hi = row_ptr[row + 1];
        for (int k = lo; k < hi; k++) {
            int c = col_idx[k] - base;
            if (c >= 0 && c < 3)
                M[r * 3 + c] = csr_vals[k];
        }
    }

    /* Analytical 3x3 inverse */
    double det = M[0]*(M[4]*M[8]-M[5]*M[7])
               - M[1]*(M[3]*M[8]-M[5]*M[6])
               + M[2]*(M[3]*M[7]-M[4]*M[6]);
    if (det == 0.0) det = 1e-30;
    double inv_det = 1.0 / det;

    double *out = diag_inv + blk * 9;
    out[0] = (M[4]*M[8]-M[5]*M[7]) * inv_det;
    out[1] = (M[2]*M[7]-M[1]*M[8]) * inv_det;
    out[2] = (M[1]*M[5]-M[2]*M[4]) * inv_det;
    out[3] = (M[5]*M[6]-M[3]*M[8]) * inv_det;
    out[4] = (M[0]*M[8]-M[2]*M[6]) * inv_det;
    out[5] = (M[2]*M[3]-M[0]*M[5]) * inv_det;
    out[6] = (M[3]*M[7]-M[4]*M[6]) * inv_det;
    out[7] = (M[1]*M[6]-M[0]*M[7]) * inv_det;
    out[8] = (M[0]*M[4]-M[1]*M[3]) * inv_det;
}

/* =========================================================================
 * CUDA kernel: apply_block_jacobi  (z = diag_inv * r)
 * One thread per block of 3 DOFs.
 * ========================================================================= */
__global__ void apply_block_jacobi_dg(double *z, const double *r,
                                       const double *diag_inv, int n_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
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

/* =========================================================================
 * CUDA kernel: compute_motion_edges_kernel
 *
 * One thread per MotionEdge (i = 0..n_poses-2).
 * Inlines the full MotionEdge math (M, A, F, Omega, xi) from MotionEdge.c
 * and atomicAdds into d_csr_vals (double) and d_xi (double).
 *
 * Thread 0 additionally scatters the anchor (1e6 on diagonal rows 0,1,2).
 *
 * Binary search (FIND_SLOT) locates the CSR slot for each (row, col) pair;
 * col_idx within each row is sorted ascending by build_csr_structure().
 * ========================================================================= */
__global__ void compute_motion_edges_kernel(
    const double *hat_x,    /* [3 * n_poses] flat: x,y,theta per pose    */
    const double *us_nu,    /* [n_poses] control input nu                 */
    const double *us_omega, /* [n_poses] control input omega              */
    double        delta,    /* time step (integer cast to double)         */
    double        lambda,   /* edge weight scalar                         */
    double        mns0,     /* motion noise stds[0] = 0.19                */
    double        mns1,     /* motion noise stds[1] = 0.001               */
    double        mns2,     /* motion noise stds[2] = 0.13                */
    double        mns3,     /* motion noise stds[3] = 0.2                 */
    double       *csr_vals, /* [nnz] — atomicAdd target                   */
    double       *xi,       /* [dim] — atomicAdd target                   */
    const int    *row_ptr,  /* [dim+1]                                    */
    const int    *col_idx,  /* [nnz]                                      */
    int           n_poses
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Thread 0: anchor (1e6 on diagonal [0,0],[1,1],[2,2]) */
    if (tid == 0) {
#define FIND_SLOT_M(ROW, COL) ({                             \
    int lo_ = row_ptr[(ROW)], hi_ = row_ptr[(ROW)+1] - 1;   \
    int mid_, slot_ = -1;                                    \
    while (lo_ <= hi_) {                                     \
        mid_ = (lo_ + hi_) >> 1;                             \
        if (col_idx[mid_] == (COL)) { slot_ = mid_; break; } \
        else if (col_idx[mid_] < (COL)) lo_ = mid_ + 1;     \
        else hi_ = mid_ - 1;                                 \
    }                                                        \
    slot_;                                                   \
})
        for (int d = 0; d < 3; d++) {
            int slot = FIND_SLOT_M(d, d);
            if (slot >= 0) atomicAdd(&csr_vals[slot], 1e6);
        }
    }

    /* Each thread computes one MotionEdge (i, i+1) */
    int i = tid;
    if (i >= n_poses - 1) return;

    /* --- Read hat_x1 (pose i) and hat_x2 (pose i+1) --- */
    double hx1 = hat_x[i*3+0];
    double hy1 = hat_x[i*3+1];
    double ht1 = hat_x[i*3+2];
    double hx2 = hat_x[(i+1)*3+0];
    double hy2 = hat_x[(i+1)*3+1];
    double ht2 = hat_x[(i+1)*3+2];

    /* --- Read control input us[t2] = us[i+1] --- */
    double nu    = us_nu[i+1];
    double omega = us_omega[i+1];
    if (fabs(omega) < 1e-5) omega = 1e-5;

    /* --- M (2x2 diagonal motion noise covariance) ---
     * M[0,0] = mns0^2 * |nu|/delta + mns1^2 * |omega|/delta
     * M[1,1] = mns2^2 * |nu|/delta + mns3^2 * |omega|/delta
     */
    double inv_delta = 1.0 / delta;
    double M00 = mns0*mns0 * fabs(nu) * inv_delta + mns1*mns1 * fabs(omega) * inv_delta;
    double M11 = mns2*mns2 * fabs(nu) * inv_delta + mns3*mns3 * fabs(omega) * inv_delta;

    /* --- Trig for A and F --- */
    double st  = sin(ht1);
    double ct  = cos(ht1);
    double stw = sin(ht1 + omega * delta);
    double ctw = cos(ht1 + omega * delta);

    /* --- A (3x2) matrix ---
     * A[0,0] = (stw - st) / omega
     * A[0,1] = nu/omega*delta*ctw - nu/omega^2*(stw-st)
     * A[1,0] = (ct - ctw) / omega
     * A[1,1] = nu/omega*delta*stw - nu/omega^2*(ct-ctw)
     * A[2,0] = 0
     * A[2,1] = delta
     */
    double inv_om  = 1.0 / omega;
    double inv_om2 = inv_om * inv_om;
    double nu_inv_om = nu * inv_om;

    double A00 = (stw - st) * inv_om;
    double A01 = nu_inv_om * delta * ctw - nu * inv_om2 * (stw - st);
    double A10 = (ct - ctw) * inv_om;
    double A11 = nu_inv_om * delta * stw - nu * inv_om2 * (ct - ctw);
    /* A20=0, A21=delta */

    /* --- F (3x3) matrix ---
     * F = I with F[0,2] = nu/omega*(cos(ht1+omega*delta)-cos(ht1))
     *               F[1,2] = nu/omega*(sin(ht1+omega*delta)-sin(ht1))
     */
    double F02 = nu_inv_om * (ctw - ct);
    double F12 = nu_inv_om * (stw - st);
    /* F is identity + [F02,F12,0] in column 2 */

    /* --- Compute A*M*A^T (3x3) + 0.0001*I ---
     *
     * A*M = [ A00*M00,  A01*M11 ]   (3x2)
     *       [ A10*M00,  A11*M11 ]
     *       [   0,    delta*M11 ]
     *
     * (A*M)*A^T: rows of AM dotted with rows of A^T = cols of A
     * (A*M*A^T)[i,j] = sum_k AM[i,k]*A[j,k]
     */
    double AM00 = A00*M00;  double AM01 = A01*M11;
    double AM10 = A10*M00;  double AM11 = A11*M11;
    double AM20 = 0.0;      double AM21 = delta*M11;

    /* S = A*M*A^T (symmetric 3x3) */
    double S00 = AM00*A00 + AM01*A01;
    double S01 = AM00*A10 + AM01*A11;
    double S02 = AM00*0.0 + AM01*delta;
    double S11 = AM10*A10 + AM11*A11;
    double S12 = AM10*0.0 + AM11*delta;
    double S22 = AM20*0.0 + AM21*delta;

    /* Add 0.0001*I */
    S00 += 0.0001;
    S11 += 0.0001;
    S22 += 0.0001;

    /* --- Invert S (3x3) analytically --- */
    double det = S00*(S11*S22 - S12*S12)
               - S01*(S01*S22 - S12*S02)
               + S02*(S01*S12 - S11*S02);
    if (det == 0.0) det = 1e-30;
    double id = 1.0 / det;

    /* Omega = inv(S) — symmetric, store upper triangle + diagonal */
    double Om00 = (S11*S22 - S12*S12) * id;
    double Om01 = (S02*S12 - S01*S22) * id;
    double Om02 = (S01*S12 - S02*S11) * id;
    double Om11 = (S00*S22 - S02*S02) * id;
    double Om12 = (S02*S01 - S00*S12) * id;
    double Om22 = (S00*S11 - S01*S01) * id;

    /* Full Om (row-major, using symmetry) */
    /* Om[r][c]:
     *  [Om00, Om01, Om02]
     *  [Om01, Om11, Om12]
     *  [Om02, Om12, Om22]
     */

    /* --- FtOm = F^T * Omega (3x3) ---
     * F^T = [[1,0,0],[0,1,0],[F02,F12,1]]  (identity + F02/F12 in row 2)
     * FtOm[r][c] = sum_k F^T[r,k] * Om[k,c]
     *
     * Row 0 of F^T = [1,0,0]  -> FtOm[0,:] = Om[0,:]
     * Row 1 of F^T = [0,1,0]  -> FtOm[1,:] = Om[1,:]
     * Row 2 of F^T = [F02,F12,1] -> FtOm[2,:] = F02*Om[0,:]+F12*Om[1,:]+Om[2,:]
     */
    double FtOm00 = Om00;
    double FtOm01 = Om01;
    double FtOm02 = Om02;
    double FtOm10 = Om01;
    double FtOm11 = Om11;
    double FtOm12 = Om12;
    double FtOm20 = F02*Om00 + F12*Om01 + Om02;
    double FtOm21 = F02*Om01 + F12*Om11 + Om12;
    double FtOm22 = F02*Om02 + F12*Om12 + Om22;

    /* --- omega_upperleft = FtOm * F * lambda (3x3) ---
     * F = [[1,0,F02],[0,1,F12],[0,0,1]]
     * (FtOm * F)[r][c]:
     *  c=0: FtOm[r,0]
     *  c=1: FtOm[r,1]
     *  c=2: FtOm[r,0]*F02 + FtOm[r,1]*F12 + FtOm[r,2]
     */
    double UL[9];
    UL[0] = (FtOm00                          ) * lambda;
    UL[1] = (FtOm01                          ) * lambda;
    UL[2] = (FtOm00*F02 + FtOm01*F12 + FtOm02) * lambda;
    UL[3] = (FtOm10                          ) * lambda;
    UL[4] = (FtOm11                          ) * lambda;
    UL[5] = (FtOm10*F02 + FtOm11*F12 + FtOm12) * lambda;
    UL[6] = (FtOm20                          ) * lambda;
    UL[7] = (FtOm21                          ) * lambda;
    UL[8] = (FtOm20*F02 + FtOm21*F12 + FtOm22) * lambda;

    /* --- omega_upperright = -FtOm * lambda (3x3) --- */
    double UR[9];
    UR[0] = -FtOm00 * lambda;
    UR[1] = -FtOm01 * lambda;
    UR[2] = -FtOm02 * lambda;
    UR[3] = -FtOm10 * lambda;
    UR[4] = -FtOm11 * lambda;
    UR[5] = -FtOm12 * lambda;
    UR[6] = -FtOm20 * lambda;
    UR[7] = -FtOm21 * lambda;
    UR[8] = -FtOm22 * lambda;

    /* --- omega_bottomleft = -Om * F * lambda (3x3) ---
     * (-Om * F)[r][c]:
     *  c=0: -Om[r,0]
     *  c=1: -Om[r,1]
     *  c=2: -Om[r,0]*F02 - Om[r,1]*F12 - Om[r,2]
     */
    double BL[9];
    BL[0] = (-Om00                         ) * lambda;
    BL[1] = (-Om01                         ) * lambda;
    BL[2] = (-Om00*F02 - Om01*F12 - Om02   ) * lambda;
    BL[3] = (-Om01                         ) * lambda;
    BL[4] = (-Om11                         ) * lambda;
    BL[5] = (-Om01*F02 - Om11*F12 - Om12   ) * lambda;
    BL[6] = (-Om02                         ) * lambda;
    BL[7] = (-Om12                         ) * lambda;
    BL[8] = (-Om02*F02 - Om12*F12 - Om22   ) * lambda;

    /* --- omega_bottomright = Om * lambda (3x3) --- */
    double BR[9];
    BR[0] = Om00 * lambda;  BR[1] = Om01 * lambda;  BR[2] = Om02 * lambda;
    BR[3] = Om01 * lambda;  BR[4] = Om11 * lambda;  BR[5] = Om12 * lambda;
    BR[6] = Om02 * lambda;  BR[7] = Om12 * lambda;  BR[8] = Om22 * lambda;

    /* --- state_transition: x2_pred ---
     * x2 = hat_x1 + [nu/omega*(sin(ht1+omega*delta)-sin(ht1)),
     *                nu/omega*(cos(ht1)-cos(ht1+omega*delta)),
     *                omega*delta]
     */
    double x2p0 = hx1 + nu_inv_om * (stw - st);
    double x2p1 = hy1 + nu_inv_om * (ct - ctw);
    double x2p2 = ht1 + omega * delta;

    /* --- diff = hat_x2 - x2_pred --- */
    double dx = hx2 - x2p0;
    double dy = hy2 - x2p1;
    double dz = ht2 - x2p2;

    /* --- xi_upper = FtOm * diff * lambda (3x1) --- */
    double XIU[3];
    XIU[0] = (FtOm00*dx + FtOm01*dy + FtOm02*dz) * lambda;
    XIU[1] = (FtOm10*dx + FtOm11*dy + FtOm12*dz) * lambda;
    XIU[2] = (FtOm20*dx + FtOm21*dy + FtOm22*dz) * lambda;

    /* --- xi_bottom = -Om * diff * lambda (3x1) --- */
    double XIB[3];
    XIB[0] = -(Om00*dx + Om01*dy + Om02*dz) * lambda;
    XIB[1] = -(Om01*dx + Om11*dy + Om12*dz) * lambda;
    XIB[2] = -(Om02*dx + Om12*dy + Om22*dz) * lambda;

    /* --- Scatter into CSR --- */
    int f1 = i * 3;
    int f2 = (i + 1) * 3;

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int slot;
            slot = FIND_SLOT_M(f1+r, f1+c); if (slot>=0) atomicAdd(&csr_vals[slot], UL[r*3+c]);
            slot = FIND_SLOT_M(f1+r, f2+c); if (slot>=0) atomicAdd(&csr_vals[slot], UR[r*3+c]);
            slot = FIND_SLOT_M(f2+r, f1+c); if (slot>=0) atomicAdd(&csr_vals[slot], BL[r*3+c]);
            slot = FIND_SLOT_M(f2+r, f2+c); if (slot>=0) atomicAdd(&csr_vals[slot], BR[r*3+c]);
        }
    }

    atomicAdd(&xi[f1+0], XIU[0]);
    atomicAdd(&xi[f1+1], XIU[1]);
    atomicAdd(&xi[f1+2], XIU[2]);
    atomicAdd(&xi[f2+0], XIB[0]);
    atomicAdd(&xi[f2+1], XIB[1]);
    atomicAdd(&xi[f2+2], XIB[2]);

#undef FIND_SLOT_M
}

/* =========================================================================
 * CPU: build_csr_structure
 *
 * Builds row_ptr[] and col_idx[] from the complete edge set (ObsEdges +
 * MotionEdges).  Also allocates and fills:
 *   sorted_obs[]  — zlist indices sorted by (landmark_id, obs_within_lm)
 *   edge_j[], edge_k[], edge_lm[]  — per ObsEdge arrays
 *   lm_pos[], lm_count[]           — per landmark arrays
 * Returns nnz.
 * ========================================================================= */
static int build_csr_structure(
    struct Z     *zlist,
    unsigned int  n_obs,
    unsigned int  n_poses,
    unsigned int  dim,
    /* outputs (CPU, caller must free): */
    int         **row_ptr_out,
    int         **col_idx_out,
    int         **sorted_obs_out,
    int         **edge_j_out,
    int         **edge_k_out,
    int         **edge_lm_out,
    int         **lm_pos_out,
    int         **lm_count_out,
    int          *n_edges_out,
    int          *n_lm_out
)
{
    /* ---- 1. Find max landmark id ---- */
    unsigned int max_lid = 0;
    for (unsigned int i = 0; i < n_obs; i++)
        if (zlist[i].landmark_id > max_lid)
            max_lid = zlist[i].landmark_id;

    unsigned int n_lm_slots = max_lid + 1;

    /* ---- 2. Count obs per landmark ---- */
    int *lm_count = (int *)calloc(n_lm_slots, sizeof(int));
    for (unsigned int i = 0; i < n_obs; i++)
        lm_count[zlist[i].landmark_id]++;

    /* ---- 3. Prefix sum -> lm_pos (start in sorted_obs) ---- */
    int *lm_pos = (int *)malloc(n_lm_slots * sizeof(int));
    lm_pos[0] = 0;
    for (unsigned int i = 1; i < n_lm_slots; i++)
        lm_pos[i] = lm_pos[i-1] + lm_count[i-1];

    /* ---- 4. Build sorted_obs: for each lm, list zlist indices ---- */
    int *sorted_obs = (int *)malloc(n_obs * sizeof(int));
    int *fill_ptr   = (int *)calloc(n_lm_slots, sizeof(int));
    for (unsigned int i = 0; i < n_obs; i++) {
        unsigned int lid = zlist[i].landmark_id;
        sorted_obs[lm_pos[lid] + fill_ptr[lid]] = (int)i;
        fill_ptr[lid]++;
    }
    free(fill_ptr);

    /* ---- 5. Count total ObsEdges and build edge arrays ---- */
    int total_obs_edges = 0;
    for (unsigned int lid = 0; lid < n_lm_slots; lid++)
        if (lm_count[lid] >= 2)
            total_obs_edges += (int)comb2((unsigned int)lm_count[lid]);

    int *edge_j  = (int *)malloc(total_obs_edges * sizeof(int));
    int *edge_k  = (int *)malloc(total_obs_edges * sizeof(int));
    int *edge_lm = (int *)malloc(total_obs_edges * sizeof(int));

    int ebase = 0;
    int n_lm_active = 0;
    for (unsigned int lid = 0; lid < n_lm_slots; lid++)
        if (lm_count[lid] >= 2) n_lm_active++;

    int *compact_lm_pos   = (int *)malloc(n_lm_active * sizeof(int));
    int *compact_lm_count = (int *)malloc(n_lm_active * sizeof(int));

    int ci = 0;
    for (unsigned int lid = 0; lid < n_lm_slots; lid++) {
        int n = lm_count[lid];
        if (n < 2) continue;
        compact_lm_pos[ci]   = lm_pos[lid];
        compact_lm_count[ci] = n;
        for (int j = 0; j < n; j++) {
            for (int k = j + 1; k < n; k++) {
                edge_j[ebase]  = j;
                edge_k[ebase]  = k;
                edge_lm[ebase] = ci;
                ebase++;
            }
        }
        ci++;
    }

    /* ---- 6. Build CSR sparsity via marker array ---- */
    char *marker = (char *)calloc((size_t)dim * dim, sizeof(char));

    /* Anchor */
    marker[0 * dim + 0] = 1;
    marker[1 * dim + 1] = 1;
    marker[2 * dim + 2] = 1;

    /* ObsEdges */
    for (int e = 0; e < total_obs_edges; e++) {
        int lm  = edge_lm[e];
        int pos = compact_lm_pos[lm];
        int j   = edge_j[e];
        int k   = edge_k[e];
        int idx_j = sorted_obs[pos + j];
        int idx_k = sorted_obs[pos + k];
        unsigned int f1 = (unsigned int)zlist[idx_j].step * 3;
        unsigned int f2 = (unsigned int)zlist[idx_k].step * 3;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                marker[(f1+r)*dim+(f1+c)] = 1;
                marker[(f1+r)*dim+(f2+c)] = 1;
                marker[(f2+r)*dim+(f1+c)] = 1;
                marker[(f2+r)*dim+(f2+c)] = 1;
            }
    }

    /* MotionEdges: consecutive pose pairs 0..n_poses-2 */
    for (unsigned int i = 0; i < n_poses - 1; i++) {
        unsigned int f1 = i * 3;
        unsigned int f2 = (i + 1) * 3;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                marker[(f1+r)*dim+(f1+c)] = 1;
                marker[(f1+r)*dim+(f2+c)] = 1;
                marker[(f2+r)*dim+(f1+c)] = 1;
                marker[(f2+r)*dim+(f2+c)] = 1;
            }
    }

    /* ---- 7. Build row_ptr ---- */
    int *row_ptr = (int *)calloc(dim + 1, sizeof(int));
    for (unsigned int row = 0; row < dim; row++) {
        int cnt = 0;
        for (unsigned int col = 0; col < dim; col++)
            if (marker[row * dim + col]) cnt++;
        row_ptr[row + 1] = cnt;
    }
    for (unsigned int row = 0; row < dim; row++)
        row_ptr[row + 1] += row_ptr[row];

    int nnz = row_ptr[dim];

    /* ---- 8. Build col_idx ---- */
    int *col_idx = (int *)malloc(nnz * sizeof(int));
    {
        int pos = 0;
        for (unsigned int row = 0; row < dim; row++)
            for (unsigned int col = 0; col < dim; col++)
                if (marker[row * dim + col])
                    col_idx[pos++] = (int)col;
    }
    free(marker);

    /* ---- Return outputs ---- */
    *row_ptr_out    = row_ptr;
    *col_idx_out    = col_idx;
    *sorted_obs_out = sorted_obs;
    *edge_j_out     = edge_j;
    *edge_k_out     = edge_k;
    *edge_lm_out    = edge_lm;
    *lm_pos_out     = compact_lm_pos;
    *lm_count_out   = compact_lm_count;
    *n_edges_out    = total_obs_edges;
    *n_lm_out       = n_lm_active;

    free(lm_count);
    free(lm_pos);

    return nnz;
}

/* =========================================================================
 * CUDA kernel: compute_and_scatter_edges_kernel
 *
 * One thread per ObsEdge.  Computes all edge math inline (no function
 * calls) and atomicAdds into d_csr_vals (double) and d_xi (double).
 *
 * Binary search is used to locate CSR slots; col_idx within each row
 * is sorted ascending by construction.
 * ========================================================================= */
__global__ void compute_and_scatter_edges_kernel(
    const double *hat_x,       /* [3 * n_poses] flat: x,y,theta per pose  */
    const double *z_range,     /* [n_obs] observation ranges               */
    const double *z_bearing,   /* [n_obs] observation bearings             */
    const int    *z_step,      /* [n_obs] pose step index for each obs     */
    const int    *sorted_obs,  /* [n_obs] sorted obs indices per landmark  */
    const int    *lm_pos,      /* [n_lm_active] start in sorted_obs        */
    const int    *edge_j,      /* [n_edges] j within lm obs list           */
    const int    *edge_k,      /* [n_edges] k within lm obs list           */
    const int    *edge_lm,     /* [n_edges] compact lm index               */
    double       *csr_vals,    /* [nnz] — atomicAdd target                 */
    double       *xi,          /* [dim] — atomicAdd target                 */
    const int    *row_ptr,     /* [dim+1]                                  */
    const int    *col_idx,     /* [nnz]                                    */
    int           n_edges
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_edges) return;

    /* ---- Decode edge ---- */
    int lm  = edge_lm[tid];
    int pos = lm_pos[lm];
    int j   = edge_j[tid];
    int k   = edge_k[tid];

    int idx_j = sorted_obs[pos + j];
    int idx_k = sorted_obs[pos + k];

    int step1 = z_step[idx_j];
    int step2 = z_step[idx_k];

    double r1 = z_range[idx_j],   b1 = z_bearing[idx_j];
    double r2 = z_range[idx_k],   b2 = z_bearing[idx_k];

    double theta1 = hat_x[step1 * 3 + 2];
    double theta2 = hat_x[step2 * 3 + 2];

    double s1 = sin(theta1 + b1);
    double c1 = cos(theta1 + b1);
    double s2 = sin(theta2 + b2);
    double c2 = cos(theta2 + b2);

    /* ---- hat_e (2x1) ---- */
    double ex = hat_x[step2*3+0] - hat_x[step1*3+0] + r2*c2 - r1*c1;
    double ey = hat_x[step2*3+1] - hat_x[step1*3+1] + r2*s2 - r1*s1;

    double q10 = r1 * SNR0;  q10 *= q10;
    double q1b = SNR1;       q1b *= q1b;
    double q20 = r2 * SNR0;  q20 *= q20;
    double q2b = SNR1;       q2b *= q2b;

    double a00 = c1*c1*q10 + r1*r1*s1*s1*q1b;
    double a01 = c1*s1*q10 - r1*r1*s1*c1*q1b;
    double a11 = s1*s1*q10 + r1*r1*c1*c1*q1b;

    double b00 = c2*c2*q20 + r2*r2*s2*s2*q2b;
    double b01 = c2*s2*q20 - r2*r2*s2*c2*q2b;
    double b11 = s2*s2*q20 + r2*r2*c2*c2*q2b;

    double sig00 = a00 + b00;
    double sig01 = a01 + b01;
    double sig11 = a11 + b11;

    double det = sig00*sig11 - sig01*sig01;
    if (det == 0.0) det = 1e-30;
    double inv_det = 1.0 / det;
    double om00 =  sig11 * inv_det;
    double om01 = -sig01 * inv_det;
    double om11 =  sig00 * inv_det;

    double OB1_00 = -om00;
    double OB1_01 = -om01;
    double OB1_02 =  om00*r1*s1 - om01*r1*c1;
    double OB1_10 = -om01;
    double OB1_11 = -om11;
    double OB1_12 =  om01*r1*s1 - om11*r1*c1;

    double OB2_00 =  om00;
    double OB2_01 =  om01;
    double OB2_02 = -om00*r2*s2 + om01*r2*c2;
    double OB2_10 =  om01;
    double OB2_11 =  om11;
    double OB2_12 = -om01*r2*s2 + om11*r2*c2;

    double tB1r0c0=-1.0, tB1r0c1=0.0;
    double tB1r1c0=0.0,  tB1r1c1=-1.0;
    double tB1r2c0=r1*s1, tB1r2c1=-r1*c1;

    double tB2r0c0=1.0,    tB2r0c1=0.0;
    double tB2r1c0=0.0,    tB2r1c1=1.0;
    double tB2r2c0=-r2*s2, tB2r2c1=r2*c2;

    double ul[9], ur[9], bl[9], br[9];

    ul[0] = tB1r0c0*OB1_00 + tB1r0c1*OB1_10;
    ul[1] = tB1r0c0*OB1_01 + tB1r0c1*OB1_11;
    ul[2] = tB1r0c0*OB1_02 + tB1r0c1*OB1_12;
    ul[3] = tB1r1c0*OB1_00 + tB1r1c1*OB1_10;
    ul[4] = tB1r1c0*OB1_01 + tB1r1c1*OB1_11;
    ul[5] = tB1r1c0*OB1_02 + tB1r1c1*OB1_12;
    ul[6] = tB1r2c0*OB1_00 + tB1r2c1*OB1_10;
    ul[7] = tB1r2c0*OB1_01 + tB1r2c1*OB1_11;
    ul[8] = tB1r2c0*OB1_02 + tB1r2c1*OB1_12;

    ur[0] = tB1r0c0*OB2_00 + tB1r0c1*OB2_10;
    ur[1] = tB1r0c0*OB2_01 + tB1r0c1*OB2_11;
    ur[2] = tB1r0c0*OB2_02 + tB1r0c1*OB2_12;
    ur[3] = tB1r1c0*OB2_00 + tB1r1c1*OB2_10;
    ur[4] = tB1r1c0*OB2_01 + tB1r1c1*OB2_11;
    ur[5] = tB1r1c0*OB2_02 + tB1r1c1*OB2_12;
    ur[6] = tB1r2c0*OB2_00 + tB1r2c1*OB2_10;
    ur[7] = tB1r2c0*OB2_01 + tB1r2c1*OB2_11;
    ur[8] = tB1r2c0*OB2_02 + tB1r2c1*OB2_12;

    bl[0] = tB2r0c0*OB1_00 + tB2r0c1*OB1_10;
    bl[1] = tB2r0c0*OB1_01 + tB2r0c1*OB1_11;
    bl[2] = tB2r0c0*OB1_02 + tB2r0c1*OB1_12;
    bl[3] = tB2r1c0*OB1_00 + tB2r1c1*OB1_10;
    bl[4] = tB2r1c0*OB1_01 + tB2r1c1*OB1_11;
    bl[5] = tB2r1c0*OB1_02 + tB2r1c1*OB1_12;
    bl[6] = tB2r2c0*OB1_00 + tB2r2c1*OB1_10;
    bl[7] = tB2r2c0*OB1_01 + tB2r2c1*OB1_11;
    bl[8] = tB2r2c0*OB1_02 + tB2r2c1*OB1_12;

    br[0] = tB2r0c0*OB2_00 + tB2r0c1*OB2_10;
    br[1] = tB2r0c0*OB2_01 + tB2r0c1*OB2_11;
    br[2] = tB2r0c0*OB2_02 + tB2r0c1*OB2_12;
    br[3] = tB2r1c0*OB2_00 + tB2r1c1*OB2_10;
    br[4] = tB2r1c0*OB2_01 + tB2r1c1*OB2_11;
    br[5] = tB2r1c0*OB2_02 + tB2r1c1*OB2_12;
    br[6] = tB2r2c0*OB2_00 + tB2r2c1*OB2_10;
    br[7] = tB2r2c0*OB2_01 + tB2r2c1*OB2_11;
    br[8] = tB2r2c0*OB2_02 + tB2r2c1*OB2_12;

    double Oe0 = om00*ex + om01*ey;
    double Oe1 = om01*ex + om11*ey;

    double xiu0 = -(tB1r0c0*Oe0 + tB1r0c1*Oe1);
    double xiu1 = -(tB1r1c0*Oe0 + tB1r1c1*Oe1);
    double xiu2 = -(tB1r2c0*Oe0 + tB1r2c1*Oe1);
    double xib0 = -(tB2r0c0*Oe0 + tB2r0c1*Oe1);
    double xib1 = -(tB2r1c0*Oe0 + tB2r1c1*Oe1);
    double xib2 = -(tB2r2c0*Oe0 + tB2r2c1*Oe1);

    /* ---- Scatter into CSR using binary search ---- */
    int f1 = step1 * 3;
    int f2 = step2 * 3;

#define FIND_SLOT(ROW, COL) ({                             \
    int lo_ = row_ptr[(ROW)], hi_ = row_ptr[(ROW)+1] - 1; \
    int mid_, slot_ = -1;                                  \
    while (lo_ <= hi_) {                                   \
        mid_ = (lo_ + hi_) >> 1;                           \
        if (col_idx[mid_] == (COL)) { slot_ = mid_; break; }\
        else if (col_idx[mid_] < (COL)) lo_ = mid_ + 1;   \
        else hi_ = mid_ - 1;                               \
    }                                                      \
    slot_;                                                 \
})

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            atomicAdd(&csr_vals[FIND_SLOT(f1+r, f1+c)], ul[r*3+c]);

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            atomicAdd(&csr_vals[FIND_SLOT(f1+r, f2+c)], ur[r*3+c]);

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            atomicAdd(&csr_vals[FIND_SLOT(f2+r, f1+c)], bl[r*3+c]);

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            atomicAdd(&csr_vals[FIND_SLOT(f2+r, f2+c)], br[r*3+c]);

    atomicAdd(&xi[f1+0], xiu0);
    atomicAdd(&xi[f1+1], xiu1);
    atomicAdd(&xi[f1+2], xiu2);
    atomicAdd(&xi[f2+0], xib0);
    atomicAdd(&xi[f2+1], xib1);
    atomicAdd(&xi[f2+2], xib2);

#undef FIND_SLOT
}

/* =========================================================================
 * device_build_diag_inv
 * ========================================================================= */
void device_build_diag_inv(int dim)
{
    (void)dim;
}

/* =========================================================================
 * device_pcg_solve
 * ========================================================================= */
void device_pcg_solve(double *x_h, double *d_csr_vals, double *d_xi,
                      int dim, int nnz, int max_iter)
{
    int n_blocks = dim / 3;

    /* --- Build block-Jacobi preconditioner on device --- */
    {
        int block = 256;
        int grid  = (n_blocks + block - 1) / block;
        build_diag_inv_kernel<<<grid, block>>>(
            s_d_row_ptr, s_d_col_idx, d_csr_vals, s_pcg_diag_inv, n_blocks);
        cuda_check(cudaGetLastError(), "build_diag_inv_kernel");
    }

    /* --- x = 0,  r = xi --- */
    cuda_check(cudaMemset(s_pcg_x, 0, (size_t)dim * sizeof(double)), "memset x");
    cuda_check(cudaMemcpy(s_pcg_r, d_xi, (size_t)dim * sizeof(double),
                          cudaMemcpyDeviceToDevice), "copy r=xi");

    cusparseSpMatSetValues(s_mat_A, d_csr_vals);

    /* --- z = M_inv * r --- */
    {
        int block = 256, grid = (n_blocks + block - 1) / block;
        apply_block_jacobi_dg<<<grid, block>>>(s_pcg_z, s_pcg_r, s_pcg_diag_inv, n_blocks);
    }

    /* --- p = z --- */
    cuda_check(cudaMemcpy(s_pcg_p, s_pcg_z, (size_t)dim * sizeof(double),
                          cudaMemcpyDeviceToDevice), "copy p=z");

    double rz_old, rz_new, pq, alpha, beta, b_norm, r_norm;
    cublasDdot(s_bl_handle, dim, s_pcg_r, 1, s_pcg_z, 1, &rz_old);
    cublasDnrm2(s_bl_handle, dim, d_xi, 1, &b_norm);
    double tol = 1e-6 * b_norm;

    double alpha_spmv = 1.0, beta_spmv = 0.0;

    for (int k = 0; k < max_iter; k++) {
        cusparseSpMV(s_sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_spmv, s_mat_A, s_vec_p, &beta_spmv, s_vec_q,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, s_spmv_buf);

        cublasDdot(s_bl_handle, dim, s_pcg_p, 1, s_pcg_q, 1, &pq);
        if (pq == 0.0) break;
        alpha = rz_old / pq;

        cublasDaxpy(s_bl_handle, dim, &alpha, s_pcg_p, 1, s_pcg_x, 1);

        double neg_alpha = -alpha;
        cublasDaxpy(s_bl_handle, dim, &neg_alpha, s_pcg_q, 1, s_pcg_r, 1);

        cublasDnrm2(s_bl_handle, dim, s_pcg_r, 1, &r_norm);
        if (r_norm < tol) break;

        {
            int block = 256, grid = (n_blocks + block - 1) / block;
            apply_block_jacobi_dg<<<grid, block>>>(s_pcg_z, s_pcg_r, s_pcg_diag_inv, n_blocks);
        }

        cublasDdot(s_bl_handle, dim, s_pcg_r, 1, s_pcg_z, 1, &rz_new);
        beta = rz_new / rz_old;

        cublasDscal(s_bl_handle, dim, &beta, s_pcg_p, 1);
        cublasDaxpy(s_bl_handle, dim, &alpha_spmv, s_pcg_z, 1, s_pcg_p, 1);

        rz_old = rz_new;
    }

    cuda_check(cudaMemcpy(x_h, s_pcg_x, (size_t)dim * sizeof(double),
                          cudaMemcpyDeviceToHost), "copy result to host");
}

/* =========================================================================
 * device_graph_init
 * ========================================================================= */
void device_graph_init(
    struct Z     *zlist,
    unsigned int  n_obs,
    struct U     *us,
    unsigned int  n_poses,
    unsigned int  dim,
    double        delta,
    double        lambda,
    const double *mns,
    int          *nnz_out
)
{
    s_dim     = (int)dim;
    s_n_poses = (int)n_poses;
    s_delta   = delta;
    s_lambda  = lambda;
    s_mns[0]  = mns[0];  s_mns[1] = mns[1];
    s_mns[2]  = mns[2];  s_mns[3] = mns[3];

    /* CPU build */
    int *row_ptr_h, *col_idx_h;
    int *sorted_obs_h;
    int *edge_j_h, *edge_k_h, *edge_lm_h;
    int *lm_pos_h, *lm_count_h;
    int n_edges, n_lm_active;

    s_nnz = build_csr_structure(
        zlist, n_obs, n_poses, dim,
        &row_ptr_h, &col_idx_h,
        &sorted_obs_h,
        &edge_j_h, &edge_k_h, &edge_lm_h,
        &lm_pos_h, &lm_count_h,
        &n_edges, &n_lm_active
    );
    s_n_edges = n_edges;

    printf("device_graph_init: dim=%d nnz=%d obs_edges=%d lm_active=%d\n",
           s_dim, s_nnz, s_n_edges, n_lm_active);

    /* Upload CSR structure */
    cuda_check(cudaMalloc(&s_d_row_ptr, (dim+1)*sizeof(int)),     "malloc row_ptr");
    cuda_check(cudaMalloc(&s_d_col_idx, s_nnz*sizeof(int)),       "malloc col_idx");
    cuda_check(cudaMemcpy(s_d_row_ptr, row_ptr_h, (dim+1)*sizeof(int), cudaMemcpyHostToDevice), "cpy row_ptr");
    cuda_check(cudaMemcpy(s_d_col_idx, col_idx_h, s_nnz*sizeof(int),   cudaMemcpyHostToDevice), "cpy col_idx");

    /* Upload sorted obs */
    cuda_check(cudaMalloc(&s_d_sorted_obs, n_obs*sizeof(int)),    "malloc sorted_obs");
    cuda_check(cudaMemcpy(s_d_sorted_obs, sorted_obs_h, n_obs*sizeof(int), cudaMemcpyHostToDevice), "cpy sorted_obs");

    /* Upload edge arrays */
    cuda_check(cudaMalloc(&s_d_edge_j,  n_edges*sizeof(int)),     "malloc edge_j");
    cuda_check(cudaMalloc(&s_d_edge_k,  n_edges*sizeof(int)),     "malloc edge_k");
    cuda_check(cudaMalloc(&s_d_edge_lm, n_edges*sizeof(int)),     "malloc edge_lm");
    cuda_check(cudaMemcpy(s_d_edge_j,  edge_j_h,  n_edges*sizeof(int), cudaMemcpyHostToDevice), "cpy edge_j");
    cuda_check(cudaMemcpy(s_d_edge_k,  edge_k_h,  n_edges*sizeof(int), cudaMemcpyHostToDevice), "cpy edge_k");
    cuda_check(cudaMemcpy(s_d_edge_lm, edge_lm_h, n_edges*sizeof(int), cudaMemcpyHostToDevice), "cpy edge_lm");

    /* Upload per-lm arrays */
    cuda_check(cudaMalloc(&s_d_lm_pos,   n_lm_active*sizeof(int)), "malloc lm_pos");
    cuda_check(cudaMalloc(&s_d_lm_count, n_lm_active*sizeof(int)), "malloc lm_count");
    cuda_check(cudaMemcpy(s_d_lm_pos,   lm_pos_h,   n_lm_active*sizeof(int), cudaMemcpyHostToDevice), "cpy lm_pos");
    cuda_check(cudaMemcpy(s_d_lm_count, lm_count_h, n_lm_active*sizeof(int), cudaMemcpyHostToDevice), "cpy lm_count");

    /* Build and upload flat obs data arrays */
    double *z_range_h   = (double *)malloc(n_obs * sizeof(double));
    double *z_bearing_h = (double *)malloc(n_obs * sizeof(double));
    int    *z_step_h    = (int    *)malloc(n_obs * sizeof(int));
    for (unsigned int i = 0; i < n_obs; i++) {
        z_range_h[i]   = zlist[i].z[0];
        z_bearing_h[i] = zlist[i].z[1];
        z_step_h[i]    = (int)zlist[i].step;
    }
    cuda_check(cudaMalloc(&s_d_z_range,   n_obs*sizeof(double)), "malloc z_range");
    cuda_check(cudaMalloc(&s_d_z_bearing, n_obs*sizeof(double)), "malloc z_bearing");
    cuda_check(cudaMalloc(&s_d_z_step,    n_obs*sizeof(int)),    "malloc z_step");
    cuda_check(cudaMemcpy(s_d_z_range,   z_range_h,   n_obs*sizeof(double), cudaMemcpyHostToDevice), "cpy z_range");
    cuda_check(cudaMemcpy(s_d_z_bearing, z_bearing_h, n_obs*sizeof(double), cudaMemcpyHostToDevice), "cpy z_bearing");
    cuda_check(cudaMemcpy(s_d_z_step,    z_step_h,    n_obs*sizeof(int),    cudaMemcpyHostToDevice), "cpy z_step");

    /* Build and upload flat us nu/omega arrays */
    double *us_nu_h    = (double *)malloc(n_poses * sizeof(double));
    double *us_omega_h = (double *)malloc(n_poses * sizeof(double));
    for (unsigned int i = 0; i < n_poses; i++) {
        us_nu_h[i]    = us[i].nu;
        us_omega_h[i] = us[i].omega;
    }
    cuda_check(cudaMalloc(&s_d_us_nu,    n_poses*sizeof(double)), "malloc us_nu");
    cuda_check(cudaMalloc(&s_d_us_omega, n_poses*sizeof(double)), "malloc us_omega");
    cuda_check(cudaMemcpy(s_d_us_nu,    us_nu_h,    n_poses*sizeof(double), cudaMemcpyHostToDevice), "cpy us_nu");
    cuda_check(cudaMemcpy(s_d_us_omega, us_omega_h, n_poses*sizeof(double), cudaMemcpyHostToDevice), "cpy us_omega");
    free(us_nu_h);
    free(us_omega_h);

    /* Allocate hat_x device buffer */
    cuda_check(cudaMalloc(&s_d_hat_x, (size_t)n_poses * 3 * sizeof(double)), "malloc hat_x");

    /* ---- Allocate persistent PCG working buffers ---- */
    int n_blocks = (int)dim / 3;
    cuda_check(cudaMalloc(&s_pcg_x,        (size_t)dim       * sizeof(double)), "malloc pcg_x");
    cuda_check(cudaMalloc(&s_pcg_r,        (size_t)dim       * sizeof(double)), "malloc pcg_r");
    cuda_check(cudaMalloc(&s_pcg_z,        (size_t)dim       * sizeof(double)), "malloc pcg_z");
    cuda_check(cudaMalloc(&s_pcg_p,        (size_t)dim       * sizeof(double)), "malloc pcg_p");
    cuda_check(cudaMalloc(&s_pcg_q,        (size_t)dim       * sizeof(double)), "malloc pcg_q");
    cuda_check(cudaMalloc(&s_pcg_diag_inv, (size_t)n_blocks * 9 * sizeof(double)), "malloc pcg_diag_inv");

    /* ---- cuBLAS and cuSPARSE handles (created once) ---- */
    cublasCreate(&s_bl_handle);
    cusparseCreate(&s_sp_handle);

    double *tmp_vals = NULL;
    cuda_check(cudaMalloc(&tmp_vals, (size_t)s_nnz * sizeof(double)), "malloc tmp_vals");
    cuda_check(cudaMemset(tmp_vals, 0, (size_t)s_nnz * sizeof(double)), "memset tmp_vals");

    cusparseCreateCsr(&s_mat_A, (int64_t)dim, (int64_t)dim, (int64_t)s_nnz,
                      s_d_row_ptr, s_d_col_idx, tmp_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&s_vec_p, dim, s_pcg_p, CUDA_R_64F);
    cusparseCreateDnVec(&s_vec_q, dim, s_pcg_q, CUDA_R_64F);

    double alpha_tmp = 1.0, beta_tmp = 0.0;
    size_t buf_size = 0;
    cusparseSpMV_bufferSize(s_sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_tmp, s_mat_A, s_vec_p, &beta_tmp, s_vec_q,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);
    if (buf_size > 0)
        cuda_check(cudaMalloc(&s_spmv_buf, buf_size), "malloc spmv_buf");

    cudaFree(tmp_vals);

    /* Cleanup CPU temporaries */
    free(row_ptr_h); free(col_idx_h); free(sorted_obs_h);
    free(edge_j_h); free(edge_k_h); free(edge_lm_h);
    free(lm_pos_h); free(lm_count_h);
    free(z_range_h); free(z_bearing_h); free(z_step_h);

    *nnz_out = s_nnz;
}

/* =========================================================================
 * device_graph_update
 * ========================================================================= */
void device_graph_update(
    struct HAT_X *hat_xs,
    unsigned int  n_poses,
    double       *d_csr_vals,
    double       *d_xi
)
{
    /* Upload hat_xs as flat double array */
    double *flat_hat = (double *)malloc((size_t)n_poses * 3 * sizeof(double));
    for (unsigned int i = 0; i < n_poses; i++) {
        flat_hat[i*3+0] = hat_xs[i].hat_x[0];
        flat_hat[i*3+1] = hat_xs[i].hat_x[1];
        flat_hat[i*3+2] = hat_xs[i].hat_x[2];
    }
    cuda_check(cudaMemcpy(s_d_hat_x, flat_hat,
                          (size_t)n_poses*3*sizeof(double),
                          cudaMemcpyHostToDevice), "upload hat_x");
    free(flat_hat);

    /* Zero d_csr_vals and d_xi on device */
    cuda_check(cudaMemset(d_csr_vals, 0, (size_t)s_nnz * sizeof(double)), "zero csr_vals");
    cuda_check(cudaMemset(d_xi,       0, (size_t)s_dim  * sizeof(double)), "zero xi");

    /* Launch MotionEdge kernel: n_poses-1 edges + anchor (thread 0) */
    {
        int block = 256;
        int grid  = (s_n_poses - 1 + block - 1) / block;
        /* Ensure at least 1 block so thread-0 anchor runs */
        if (grid < 1) grid = 1;
        compute_motion_edges_kernel<<<grid, block>>>(
            s_d_hat_x,
            s_d_us_nu,
            s_d_us_omega,
            s_delta,
            s_lambda,
            s_mns[0], s_mns[1], s_mns[2], s_mns[3],
            d_csr_vals,
            d_xi,
            s_d_row_ptr,
            s_d_col_idx,
            s_n_poses
        );
        cuda_check(cudaGetLastError(), "compute_motion_edges_kernel");
    }

    /* Launch ObsEdge kernel */
    if (s_n_edges > 0) {
        int block = 256;
        int grid  = (s_n_edges + block - 1) / block;
        compute_and_scatter_edges_kernel<<<grid, block>>>(
            s_d_hat_x,
            s_d_z_range,
            s_d_z_bearing,
            s_d_z_step,
            s_d_sorted_obs,
            s_d_lm_pos,
            s_d_edge_j,
            s_d_edge_k,
            s_d_edge_lm,
            d_csr_vals,
            d_xi,
            s_d_row_ptr,
            s_d_col_idx,
            s_n_edges
        );
        cuda_check(cudaGetLastError(), "compute_and_scatter_edges_kernel");
    }

    cuda_check(cudaDeviceSynchronize(), "kernel sync");
}

/* =========================================================================
 * Accessors for persistent device CSR structure pointers
 * ========================================================================= */
int *device_graph_get_row_ptr(void) { return s_d_row_ptr; }
int *device_graph_get_col_idx(void) { return s_d_col_idx; }

/* =========================================================================
 * device_graph_free
 * ========================================================================= */
void device_graph_free(void)
{
    if (s_d_row_ptr)    { cudaFree(s_d_row_ptr);    s_d_row_ptr    = NULL; }
    if (s_d_col_idx)    { cudaFree(s_d_col_idx);    s_d_col_idx    = NULL; }
    if (s_d_hat_x)      { cudaFree(s_d_hat_x);      s_d_hat_x      = NULL; }
    if (s_d_sorted_obs) { cudaFree(s_d_sorted_obs); s_d_sorted_obs = NULL; }
    if (s_d_edge_j)     { cudaFree(s_d_edge_j);     s_d_edge_j     = NULL; }
    if (s_d_edge_k)     { cudaFree(s_d_edge_k);     s_d_edge_k     = NULL; }
    if (s_d_edge_lm)    { cudaFree(s_d_edge_lm);    s_d_edge_lm    = NULL; }
    if (s_d_lm_pos)     { cudaFree(s_d_lm_pos);     s_d_lm_pos     = NULL; }
    if (s_d_lm_count)   { cudaFree(s_d_lm_count);   s_d_lm_count   = NULL; }
    if (s_d_z_range)    { cudaFree(s_d_z_range);    s_d_z_range    = NULL; }
    if (s_d_z_bearing)  { cudaFree(s_d_z_bearing);  s_d_z_bearing  = NULL; }
    if (s_d_z_step)     { cudaFree(s_d_z_step);     s_d_z_step     = NULL; }
    if (s_d_us_nu)      { cudaFree(s_d_us_nu);      s_d_us_nu      = NULL; }
    if (s_d_us_omega)   { cudaFree(s_d_us_omega);   s_d_us_omega   = NULL; }

    /* PCG working buffers */
    if (s_pcg_x)        { cudaFree(s_pcg_x);        s_pcg_x        = NULL; }
    if (s_pcg_r)        { cudaFree(s_pcg_r);        s_pcg_r        = NULL; }
    if (s_pcg_z)        { cudaFree(s_pcg_z);        s_pcg_z        = NULL; }
    if (s_pcg_p)        { cudaFree(s_pcg_p);        s_pcg_p        = NULL; }
    if (s_pcg_q)        { cudaFree(s_pcg_q);        s_pcg_q        = NULL; }
    if (s_pcg_diag_inv) { cudaFree(s_pcg_diag_inv); s_pcg_diag_inv = NULL; }
    if (s_spmv_buf)     { cudaFree(s_spmv_buf);     s_spmv_buf     = NULL; }

    /* cuSPARSE / cuBLAS handles */
    if (s_vec_p)     { cusparseDestroyDnVec(s_vec_p);  s_vec_p    = NULL; }
    if (s_vec_q)     { cusparseDestroyDnVec(s_vec_q);  s_vec_q    = NULL; }
    if (s_mat_A)     { cusparseDestroySpMat(s_mat_A);  s_mat_A    = NULL; }
    if (s_sp_handle) { cusparseDestroy(s_sp_handle);   s_sp_handle = NULL; }
    if (s_bl_handle) { cublasDestroy(s_bl_handle);     s_bl_handle = NULL; }
}
