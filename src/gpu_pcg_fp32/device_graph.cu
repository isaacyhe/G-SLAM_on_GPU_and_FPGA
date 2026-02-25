/*
 * device_graph.cu  —  device-resident edge computation and CSR assembly
 *                     for gpu_pcg_fp32 (FP32 CSR values, FP64 xi)
 *
 * Architecture:
 *  - build_csr_structure(): one-time CPU function to derive row_ptr/col_idx.
 *  - device_graph_init(): uploads fixed arrays to persistent device buffers,
 *    including us nu/omega for the motion model.
 *  - compute_motion_edges_kernel_f(): one thread per MotionEdge (i=0..n-2).
 *    Computes full motion model inline; CSR uses double atomicAdd into
 *    s_d_csr_vals_d (deterministic), xi double.
 *    Thread 0 adds anchor (1e6 on diagonal rows 0,1,2).
 *  - compute_and_scatter_edges_kernel_f(): fused CUDA kernel — one thread per
 *    ObsEdge.  All edge math in double; double atomicAdd for CSR (deterministic),
 *    double atomicAdd for xi.
 *  - cast_d2f_kernel(): cast double CSR buffer to float after both kernels.
 *  - device_graph_update(): per-round entry point: upload hat_xs, zero buffers,
 *    launch both kernels, cast d→f, sync.
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
 * Static (persistent) device and host buffers
 * ========================================================================= */

static int  *s_d_row_ptr   = NULL;
static int  *s_d_col_idx   = NULL;
static int   s_nnz         = 0;
static int   s_dim         = 0;

static double *s_d_hat_x   = NULL;   /* updated each round */

static int    *s_d_sorted_obs = NULL;
static int    *s_d_edge_j     = NULL;
static int    *s_d_edge_k     = NULL;
static int    *s_d_edge_lm    = NULL;
static int    *s_d_lm_pos     = NULL;
static int    *s_d_lm_count   = NULL;

static double *s_d_z_range   = NULL;
static double *s_d_z_bearing = NULL;
static int    *s_d_z_step    = NULL;

/* Motion model data (fixed, uploaded once) */
static double *s_d_us_nu    = NULL;  /* [n_poses] us[i].nu   */
static double *s_d_us_omega = NULL;  /* [n_poses] us[i].omega */
static double  s_delta;
static double  s_lambda;
static double  s_mns[4];             /* {0.19, 0.001, 0.13, 0.2} */

/* Double-precision CSR accumulation buffer (avoids float atomicAdd nondeterminism) */
static double *s_d_csr_vals_d = NULL;

static int s_n_edges = 0;
static int s_n_poses = 0;

/* =========================================================================
 * Persistent PCG state (float CSR, double xi cast to float inside PCG)
 * ========================================================================= */
static float  *s_pcg_x        = NULL;
static float  *s_pcg_r        = NULL;
static float  *s_pcg_z        = NULL;
static float  *s_pcg_p        = NULL;
static float  *s_pcg_q        = NULL;
static float  *s_pcg_diag_inv = NULL;
static float  *s_pcg_xi_f     = NULL;   /* xi cast from double to float */

static cublasHandle_t       s_bl_handle  = NULL;
static cusparseHandle_t     s_sp_handle  = NULL;
static cusparseSpMatDescr_t s_mat_A      = NULL;
static cusparseDnVecDescr_t s_vec_p      = NULL;
static cusparseDnVecDescr_t s_vec_q      = NULL;
static void                *s_spmv_buf   = NULL;

/* =========================================================================
 * Helpers
 * ========================================================================= */
static void cuda_check(cudaError_t err, const char *where)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

static unsigned int comb2(unsigned int n) { return n * (n - 1) / 2; }

/* =========================================================================
 * CPU: build_csr_structure  (identical logic to gpu_pcg_fp64 version)
 * ========================================================================= */
static int build_csr_structure(
    struct Z     *zlist,
    unsigned int  n_obs,
    unsigned int  n_poses,
    unsigned int  dim,
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
    unsigned int max_lid = 0;
    for (unsigned int i = 0; i < n_obs; i++)
        if (zlist[i].landmark_id > max_lid) max_lid = zlist[i].landmark_id;

    unsigned int n_lm_slots = max_lid + 1;

    int *lm_count = (int *)calloc(n_lm_slots, sizeof(int));
    for (unsigned int i = 0; i < n_obs; i++)
        lm_count[zlist[i].landmark_id]++;

    int *lm_pos = (int *)malloc(n_lm_slots * sizeof(int));
    lm_pos[0] = 0;
    for (unsigned int i = 1; i < n_lm_slots; i++)
        lm_pos[i] = lm_pos[i-1] + lm_count[i-1];

    int *sorted_obs = (int *)malloc(n_obs * sizeof(int));
    int *fill_ptr   = (int *)calloc(n_lm_slots, sizeof(int));
    for (unsigned int i = 0; i < n_obs; i++) {
        unsigned int lid = zlist[i].landmark_id;
        sorted_obs[lm_pos[lid] + fill_ptr[lid]] = (int)i;
        fill_ptr[lid]++;
    }
    free(fill_ptr);

    int total_obs_edges = 0;
    for (unsigned int lid = 0; lid < n_lm_slots; lid++)
        if (lm_count[lid] >= 2)
            total_obs_edges += (int)comb2((unsigned int)lm_count[lid]);

    int *edge_j  = (int *)malloc(total_obs_edges * sizeof(int));
    int *edge_k  = (int *)malloc(total_obs_edges * sizeof(int));
    int *edge_lm = (int *)malloc(total_obs_edges * sizeof(int));

    int n_lm_active = 0;
    for (unsigned int lid = 0; lid < n_lm_slots; lid++)
        if (lm_count[lid] >= 2) n_lm_active++;

    int *compact_lm_pos   = (int *)malloc(n_lm_active * sizeof(int));
    int *compact_lm_count = (int *)malloc(n_lm_active * sizeof(int));

    int ebase = 0, ci = 0;
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

    /* Build CSR marker */
    char *marker = (char *)calloc((size_t)dim * dim, sizeof(char));
    marker[0*dim+0] = 1; marker[1*dim+1] = 1; marker[2*dim+2] = 1;

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

    for (unsigned int i = 0; i < n_poses - 1; i++) {
        unsigned int f1 = i * 3, f2 = (i+1) * 3;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                marker[(f1+r)*dim+(f1+c)] = 1;
                marker[(f1+r)*dim+(f2+c)] = 1;
                marker[(f2+r)*dim+(f1+c)] = 1;
                marker[(f2+r)*dim+(f2+c)] = 1;
            }
    }

    int *row_ptr = (int *)calloc(dim + 1, sizeof(int));
    for (unsigned int row = 0; row < dim; row++) {
        int cnt = 0;
        for (unsigned int col = 0; col < dim; col++)
            if (marker[row*dim+col]) cnt++;
        row_ptr[row+1] = cnt;
    }
    for (unsigned int row = 0; row < dim; row++)
        row_ptr[row+1] += row_ptr[row];
    int nnz = row_ptr[dim];

    int *col_idx = (int *)malloc(nnz * sizeof(int));
    {
        int pos = 0;
        for (unsigned int row = 0; row < dim; row++)
            for (unsigned int col = 0; col < dim; col++)
                if (marker[row*dim+col])
                    col_idx[pos++] = (int)col;
    }
    free(marker);

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
 * CUDA kernel: compute_motion_edges_kernel_f  (double CSR accumulation, FP64 xi)
 *
 * One thread per MotionEdge (i = 0..n_poses-2).
 * Same motion math as compute_motion_edges_kernel in gpu_pcg_fp64.
 * Uses double atomicAdd for csr_vals_d (deterministic); double atomicAdd for xi.
 * Thread 0 additionally sets anchor (1e6 on diagonal rows 0,1,2).
 * ========================================================================= */
__global__ void compute_motion_edges_kernel_f(
    const double *hat_x,
    const double *us_nu,
    const double *us_omega,
    double        delta,
    double        lambda,
    double        mns0,
    double        mns1,
    double        mns2,
    double        mns3,
    double       *csr_vals_d, /* double — deterministic atomicAdd */
    double       *xi,         /* double — atomicAdd               */
    const int    *row_ptr,
    const int    *col_idx,
    int           n_poses
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

#define FIND_SLOT_MF(ROW, COL) ({                            \
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

    /* Thread 0: anchor */
    if (tid == 0) {
        for (int d = 0; d < 3; d++) {
            int slot = FIND_SLOT_MF(d, d);
            if (slot >= 0) atomicAdd(&csr_vals_d[slot], 1000000.0);
        }
    }

    int i = tid;
    if (i >= n_poses - 1) return;

    double hx1 = hat_x[i*3+0], hy1 = hat_x[i*3+1], ht1 = hat_x[i*3+2];
    double hx2 = hat_x[(i+1)*3+0], hy2 = hat_x[(i+1)*3+1], ht2 = hat_x[(i+1)*3+2];

    double nu    = us_nu[i+1];
    double omega = us_omega[i+1];
    if (fabs(omega) < 1e-5) omega = 1e-5;

    double inv_delta = 1.0 / delta;
    double M00 = mns0*mns0 * fabs(nu) * inv_delta + mns1*mns1 * fabs(omega) * inv_delta;
    double M11 = mns2*mns2 * fabs(nu) * inv_delta + mns3*mns3 * fabs(omega) * inv_delta;

    double st  = sin(ht1), ct  = cos(ht1);
    double stw = sin(ht1 + omega * delta), ctw = cos(ht1 + omega * delta);

    double inv_om  = 1.0 / omega;
    double inv_om2 = inv_om * inv_om;
    double nu_inv_om = nu * inv_om;

    double A00 = (stw - st) * inv_om;
    double A01 = nu_inv_om * delta * ctw - nu * inv_om2 * (stw - st);
    double A10 = (ct - ctw) * inv_om;
    double A11 = nu_inv_om * delta * stw - nu * inv_om2 * (ct - ctw);

    double F02 = nu_inv_om * (ctw - ct);
    double F12 = nu_inv_om * (stw - st);

    double AM00 = A00*M00, AM01 = A01*M11;
    double AM10 = A10*M00, AM11 = A11*M11;
    double AM21 = delta*M11;

    double S00 = AM00*A00 + AM01*A01;
    double S01 = AM00*A10 + AM01*A11;
    double S02 = AM01*delta;
    double S11 = AM10*A10 + AM11*A11;
    double S12 = AM11*delta;
    double S22 = AM21*delta;
    S00 += 0.0001; S11 += 0.0001; S22 += 0.0001;

    double det = S00*(S11*S22 - S12*S12)
               - S01*(S01*S22 - S12*S02)
               + S02*(S01*S12 - S11*S02);
    if (det == 0.0) det = 1e-30;
    double id = 1.0 / det;

    double Om00 = (S11*S22 - S12*S12) * id;
    double Om01 = (S02*S12 - S01*S22) * id;
    double Om02 = (S01*S12 - S02*S11) * id;
    double Om11 = (S00*S22 - S02*S02) * id;
    double Om12 = (S02*S01 - S00*S12) * id;
    double Om22 = (S00*S11 - S01*S01) * id;

    double FtOm00 = Om00, FtOm01 = Om01, FtOm02 = Om02;
    double FtOm10 = Om01, FtOm11 = Om11, FtOm12 = Om12;
    double FtOm20 = F02*Om00 + F12*Om01 + Om02;
    double FtOm21 = F02*Om01 + F12*Om11 + Om12;
    double FtOm22 = F02*Om02 + F12*Om12 + Om22;

    double UL[9], UR[9], BL[9], BR[9];
    UL[0]=(FtOm00)*lambda;              UL[1]=(FtOm01)*lambda;              UL[2]=(FtOm00*F02+FtOm01*F12+FtOm02)*lambda;
    UL[3]=(FtOm10)*lambda;              UL[4]=(FtOm11)*lambda;              UL[5]=(FtOm10*F02+FtOm11*F12+FtOm12)*lambda;
    UL[6]=(FtOm20)*lambda;              UL[7]=(FtOm21)*lambda;              UL[8]=(FtOm20*F02+FtOm21*F12+FtOm22)*lambda;
    UR[0]=-FtOm00*lambda; UR[1]=-FtOm01*lambda; UR[2]=-FtOm02*lambda;
    UR[3]=-FtOm10*lambda; UR[4]=-FtOm11*lambda; UR[5]=-FtOm12*lambda;
    UR[6]=-FtOm20*lambda; UR[7]=-FtOm21*lambda; UR[8]=-FtOm22*lambda;
    BL[0]=(-Om00)*lambda;               BL[1]=(-Om01)*lambda;               BL[2]=(-Om00*F02-Om01*F12-Om02)*lambda;
    BL[3]=(-Om01)*lambda;               BL[4]=(-Om11)*lambda;               BL[5]=(-Om01*F02-Om11*F12-Om12)*lambda;
    BL[6]=(-Om02)*lambda;               BL[7]=(-Om12)*lambda;               BL[8]=(-Om02*F02-Om12*F12-Om22)*lambda;
    BR[0]=Om00*lambda; BR[1]=Om01*lambda; BR[2]=Om02*lambda;
    BR[3]=Om01*lambda; BR[4]=Om11*lambda; BR[5]=Om12*lambda;
    BR[6]=Om02*lambda; BR[7]=Om12*lambda; BR[8]=Om22*lambda;

    double x2p0 = hx1 + nu_inv_om * (stw - st);
    double x2p1 = hy1 + nu_inv_om * (ct - ctw);
    double x2p2 = ht1 + omega * delta;

    double dx = hx2 - x2p0, dy = hy2 - x2p1, dz = ht2 - x2p2;

    double XIU[3], XIB[3];
    XIU[0] = (FtOm00*dx + FtOm01*dy + FtOm02*dz) * lambda;
    XIU[1] = (FtOm10*dx + FtOm11*dy + FtOm12*dz) * lambda;
    XIU[2] = (FtOm20*dx + FtOm21*dy + FtOm22*dz) * lambda;
    XIB[0] = -(Om00*dx + Om01*dy + Om02*dz) * lambda;
    XIB[1] = -(Om01*dx + Om11*dy + Om12*dz) * lambda;
    XIB[2] = -(Om02*dx + Om12*dy + Om22*dz) * lambda;

    int f1 = i * 3, f2 = (i + 1) * 3;

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int slot;
            slot = FIND_SLOT_MF(f1+r, f1+c); if (slot>=0) atomicAdd(&csr_vals_d[slot], UL[r*3+c]);
            slot = FIND_SLOT_MF(f1+r, f2+c); if (slot>=0) atomicAdd(&csr_vals_d[slot], UR[r*3+c]);
            slot = FIND_SLOT_MF(f2+r, f1+c); if (slot>=0) atomicAdd(&csr_vals_d[slot], BL[r*3+c]);
            slot = FIND_SLOT_MF(f2+r, f2+c); if (slot>=0) atomicAdd(&csr_vals_d[slot], BR[r*3+c]);
        }
    }

    atomicAdd(&xi[f1+0], XIU[0]); atomicAdd(&xi[f1+1], XIU[1]); atomicAdd(&xi[f1+2], XIU[2]);
    atomicAdd(&xi[f2+0], XIB[0]); atomicAdd(&xi[f2+1], XIB[1]); atomicAdd(&xi[f2+2], XIB[2]);

#undef FIND_SLOT_MF
}

/* =========================================================================
 * CUDA kernel: compute_and_scatter_edges_kernel_f  (double CSR accumulation, FP64 xi)
 *
 * Same edge math as the fp64 kernel.
 * Uses double atomicAdd for csr_vals_d (deterministic); double atomicAdd for xi.
 * ========================================================================= */
__global__ void compute_and_scatter_edges_kernel_f(
    const double *hat_x,
    const double *z_range,
    const double *z_bearing,
    const int    *z_step,
    const int    *sorted_obs,
    const int    *lm_pos,
    const int    *edge_j,
    const int    *edge_k,
    const int    *edge_lm,
    double       *csr_vals_d,  /* double — deterministic atomicAdd */
    double       *xi,          /* double — atomicAdd               */
    const int    *row_ptr,
    const int    *col_idx,
    int           n_edges
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_edges) return;

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

    double theta1 = hat_x[step1*3+2];
    double theta2 = hat_x[step2*3+2];

    double s1 = sin(theta1 + b1), c1 = cos(theta1 + b1);
    double s2 = sin(theta2 + b2), c2 = cos(theta2 + b2);

    double ex = hat_x[step2*3+0] - hat_x[step1*3+0] + r2*c2 - r1*c1;
    double ey = hat_x[step2*3+1] - hat_x[step1*3+1] + r2*s2 - r1*s1;

    double q10 = r1*SNR0; q10 *= q10;
    double q1b = SNR1;    q1b *= q1b;
    double q20 = r2*SNR0; q20 *= q20;
    double q2b = SNR1;    q2b *= q2b;

    double a00 = c1*c1*q10 + r1*r1*s1*s1*q1b;
    double a01 = c1*s1*q10 - r1*r1*s1*c1*q1b;
    double a11 = s1*s1*q10 + r1*r1*c1*c1*q1b;
    double b00 = c2*c2*q20 + r2*r2*s2*s2*q2b;
    double b01 = c2*s2*q20 - r2*r2*s2*c2*q2b;
    double b11 = s2*s2*q20 + r2*r2*c2*c2*q2b;

    double sig00 = a00+b00, sig01 = a01+b01, sig11 = a11+b11;
    double det = sig00*sig11 - sig01*sig01;
    if (det == 0.0) det = 1e-30;
    double inv_det = 1.0/det;
    double om00 =  sig11*inv_det;
    double om01 = -sig01*inv_det;
    double om11 =  sig00*inv_det;

    /* Omega * B1 */
    double OB1_00 = -om00, OB1_01 = -om01, OB1_02 = om00*r1*s1 - om01*r1*c1;
    double OB1_10 = -om01, OB1_11 = -om11, OB1_12 = om01*r1*s1 - om11*r1*c1;

    /* Omega * B2 */
    double OB2_00 =  om00, OB2_01 =  om01, OB2_02 = -om00*r2*s2 + om01*r2*c2;
    double OB2_10 =  om01, OB2_11 =  om11, OB2_12 = -om01*r2*s2 + om11*r2*c2;

    double tB1r0c0=-1.0,   tB1r0c1=0.0;
    double tB1r1c0=0.0,    tB1r1c1=-1.0;
    double tB1r2c0=r1*s1,  tB1r2c1=-r1*c1;

    double tB2r0c0=1.0,    tB2r0c1=0.0;
    double tB2r1c0=0.0,    tB2r1c1=1.0;
    double tB2r2c0=-r2*s2, tB2r2c1=r2*c2;

    double ul[9], ur[9], bl[9], br[9];

    ul[0]=tB1r0c0*OB1_00+tB1r0c1*OB1_10; ul[1]=tB1r0c0*OB1_01+tB1r0c1*OB1_11; ul[2]=tB1r0c0*OB1_02+tB1r0c1*OB1_12;
    ul[3]=tB1r1c0*OB1_00+tB1r1c1*OB1_10; ul[4]=tB1r1c0*OB1_01+tB1r1c1*OB1_11; ul[5]=tB1r1c0*OB1_02+tB1r1c1*OB1_12;
    ul[6]=tB1r2c0*OB1_00+tB1r2c1*OB1_10; ul[7]=tB1r2c0*OB1_01+tB1r2c1*OB1_11; ul[8]=tB1r2c0*OB1_02+tB1r2c1*OB1_12;

    ur[0]=tB1r0c0*OB2_00+tB1r0c1*OB2_10; ur[1]=tB1r0c0*OB2_01+tB1r0c1*OB2_11; ur[2]=tB1r0c0*OB2_02+tB1r0c1*OB2_12;
    ur[3]=tB1r1c0*OB2_00+tB1r1c1*OB2_10; ur[4]=tB1r1c0*OB2_01+tB1r1c1*OB2_11; ur[5]=tB1r1c0*OB2_02+tB1r1c1*OB2_12;
    ur[6]=tB1r2c0*OB2_00+tB1r2c1*OB2_10; ur[7]=tB1r2c0*OB2_01+tB1r2c1*OB2_11; ur[8]=tB1r2c0*OB2_02+tB1r2c1*OB2_12;

    bl[0]=tB2r0c0*OB1_00+tB2r0c1*OB1_10; bl[1]=tB2r0c0*OB1_01+tB2r0c1*OB1_11; bl[2]=tB2r0c0*OB1_02+tB2r0c1*OB1_12;
    bl[3]=tB2r1c0*OB1_00+tB2r1c1*OB1_10; bl[4]=tB2r1c0*OB1_01+tB2r1c1*OB1_11; bl[5]=tB2r1c0*OB1_02+tB2r1c1*OB1_12;
    bl[6]=tB2r2c0*OB1_00+tB2r2c1*OB1_10; bl[7]=tB2r2c0*OB1_01+tB2r2c1*OB1_11; bl[8]=tB2r2c0*OB1_02+tB2r2c1*OB1_12;

    br[0]=tB2r0c0*OB2_00+tB2r0c1*OB2_10; br[1]=tB2r0c0*OB2_01+tB2r0c1*OB2_11; br[2]=tB2r0c0*OB2_02+tB2r0c1*OB2_12;
    br[3]=tB2r1c0*OB2_00+tB2r1c1*OB2_10; br[4]=tB2r1c0*OB2_01+tB2r1c1*OB2_11; br[5]=tB2r1c0*OB2_02+tB2r1c1*OB2_12;
    br[6]=tB2r2c0*OB2_00+tB2r2c1*OB2_10; br[7]=tB2r2c0*OB2_01+tB2r2c1*OB2_11; br[8]=tB2r2c0*OB2_02+tB2r2c1*OB2_12;

    double Oe0 = om00*ex + om01*ey;
    double Oe1 = om01*ex + om11*ey;

    double xiu0 = -(tB1r0c0*Oe0 + tB1r0c1*Oe1);
    double xiu1 = -(tB1r1c0*Oe0 + tB1r1c1*Oe1);
    double xiu2 = -(tB1r2c0*Oe0 + tB1r2c1*Oe1);
    double xib0 = -(tB2r0c0*Oe0 + tB2r0c1*Oe1);
    double xib1 = -(tB2r1c0*Oe0 + tB2r1c1*Oe1);
    double xib2 = -(tB2r2c0*Oe0 + tB2r2c1*Oe1);

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

    /* Scatter: double atomicAdd for CSR values (deterministic) */
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            atomicAdd(&csr_vals_d[FIND_SLOT(f1+r, f1+c)], ul[r*3+c]);
            atomicAdd(&csr_vals_d[FIND_SLOT(f1+r, f2+c)], ur[r*3+c]);
            atomicAdd(&csr_vals_d[FIND_SLOT(f2+r, f1+c)], bl[r*3+c]);
            atomicAdd(&csr_vals_d[FIND_SLOT(f2+r, f2+c)], br[r*3+c]);
        }
    }

    /* Scatter: double atomicAdd for xi */
    atomicAdd(&xi[f1+0], xiu0);
    atomicAdd(&xi[f1+1], xiu1);
    atomicAdd(&xi[f1+2], xiu2);
    atomicAdd(&xi[f2+0], xib0);
    atomicAdd(&xi[f2+1], xib1);
    atomicAdd(&xi[f2+2], xib2);

#undef FIND_SLOT
}

/* =========================================================================
 * CUDA kernel: build_diag_inv_kernel_f  (float CSR → float diag_inv)
 * ========================================================================= */
__global__ void build_diag_inv_kernel_f(
    const int   *row_ptr,
    const int   *col_idx,
    const float *csr_vals,
    float       *diag_inv,
    int          n_blocks
)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;

    int base = blk * 3;
    float M[9] = {0,0,0, 0,0,0, 0,0,0};

    for (int r = 0; r < 3; r++) {
        int row = base + r;
        for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
            int c = col_idx[k] - base;
            if (c >= 0 && c < 3) M[r * 3 + c] = csr_vals[k];
        }
    }

    float det = M[0]*(M[4]*M[8]-M[5]*M[7])
              - M[1]*(M[3]*M[8]-M[5]*M[6])
              + M[2]*(M[3]*M[7]-M[4]*M[6]);
    if (det == 0.0f) det = 1e-30f;
    float inv_det = 1.0f / det;

    float *out = diag_inv + blk * 9;
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

/* block_jacobi_apply_f: z = diag_inv * r  (float) */
__global__ void block_jacobi_apply_f_dg(float *z, const float *r,
                                         const float *diag_inv, int n_blocks)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;
    int base = blk * 3;
    const float *inv = diag_inv + blk * 9;
    z[base+0] = inv[0]*r[base+0] + inv[1]*r[base+1] + inv[2]*r[base+2];
    z[base+1] = inv[3]*r[base+0] + inv[4]*r[base+1] + inv[5]*r[base+2];
    z[base+2] = inv[6]*r[base+0] + inv[7]*r[base+1] + inv[8]*r[base+2];
}

/* double_to_float_kernel: cast double array to float array on device */
__global__ void double_to_float_dg(const double *src, float *dst, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = (float)src[tid];
}

/* =========================================================================
 * device_pcg_solve  (fp32 version)
 * d_csr_vals: float[nnz], d_xi: double[dim]
 * x_h: float[dim] result (CPU)
 * ========================================================================= */
void device_pcg_solve(float *x_h, float *d_csr_vals, double *d_xi,
                      int dim, int nnz, int max_iter)
{
    int n_blocks = dim / 3;
    int blk256 = 256, grid_blk = (n_blocks + blk256 - 1) / blk256;
    int grid_dim = (dim + blk256 - 1) / blk256;

    /* --- Build float diag_inv on device --- */
    build_diag_inv_kernel_f<<<grid_blk, blk256>>>(
        s_d_row_ptr, s_d_col_idx, d_csr_vals, s_pcg_diag_inv, n_blocks);
    cuda_check(cudaGetLastError(), "build_diag_inv_kernel_f");

    /* --- Cast double xi to float --- */
    double_to_float_dg<<<grid_dim, blk256>>>(d_xi, s_pcg_xi_f, dim);
    cuda_check(cudaGetLastError(), "double_to_float xi");

    /* --- Update SpMat values pointer --- */
    cusparseSpMatSetValues(s_mat_A, d_csr_vals);

    /* --- x = 0, r = xi_f --- */
    cuda_check(cudaMemset(s_pcg_x, 0, (size_t)dim * sizeof(float)), "memset x");
    cuda_check(cudaMemcpy(s_pcg_r, s_pcg_xi_f, (size_t)dim * sizeof(float),
                          cudaMemcpyDeviceToDevice), "copy r=xi_f");

    /* --- z = M_inv * r --- */
    block_jacobi_apply_f_dg<<<grid_blk, blk256>>>(s_pcg_z, s_pcg_r, s_pcg_diag_inv, n_blocks);

    /* --- p = z --- */
    cuda_check(cudaMemcpy(s_pcg_p, s_pcg_z, (size_t)dim * sizeof(float),
                          cudaMemcpyDeviceToDevice), "copy p=z");

    /* --- rz_old = dot(r, z), b_norm --- */
    float rz_old, rz_new, pq, alpha, beta, b_norm, r_norm;
    cublasSdot(s_bl_handle, dim, s_pcg_r, 1, s_pcg_z, 1, &rz_old);
    cublasSnrm2(s_bl_handle, dim, s_pcg_xi_f, 1, &b_norm);
    float tol = 1e-6f * b_norm;

    float alpha_spmv = 1.0f, beta_spmv = 0.0f;

    for (int k = 0; k < max_iter; k++) {
        /* q = Omega * p */
        cusparseSpMV(s_sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_spmv, s_mat_A, s_vec_p, &beta_spmv, s_vec_q,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, s_spmv_buf);

        cublasSdot(s_bl_handle, dim, s_pcg_p, 1, s_pcg_q, 1, &pq);
        if (pq == 0.0f) break;
        alpha = rz_old / pq;

        cublasSaxpy(s_bl_handle, dim, &alpha, s_pcg_p, 1, s_pcg_x, 1);

        float neg_alpha = -alpha;
        cublasSaxpy(s_bl_handle, dim, &neg_alpha, s_pcg_q, 1, s_pcg_r, 1);

        cublasSnrm2(s_bl_handle, dim, s_pcg_r, 1, &r_norm);
        if (r_norm < tol) break;

        block_jacobi_apply_f_dg<<<grid_blk, blk256>>>(s_pcg_z, s_pcg_r, s_pcg_diag_inv, n_blocks);

        cublasSdot(s_bl_handle, dim, s_pcg_r, 1, s_pcg_z, 1, &rz_new);
        beta = rz_new / rz_old;

        cublasSscal(s_bl_handle, dim, &beta, s_pcg_p, 1);
        cublasSaxpy(s_bl_handle, dim, &alpha_spmv, s_pcg_z, 1, s_pcg_p, 1);

        rz_old = rz_new;
    }

    cuda_check(cudaMemcpy(x_h, s_pcg_x, (size_t)dim * sizeof(float),
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
    s_mns[0]  = mns[0]; s_mns[1] = mns[1];
    s_mns[2]  = mns[2]; s_mns[3] = mns[3];

    int *row_ptr_h, *col_idx_h, *sorted_obs_h;
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

    printf("device_graph_init (fp32): dim=%d nnz=%d obs_edges=%d lm_active=%d\n",
           s_dim, s_nnz, s_n_edges, n_lm_active);

    cuda_check(cudaMalloc(&s_d_row_ptr, (dim+1)*sizeof(int)),     "malloc row_ptr");
    cuda_check(cudaMalloc(&s_d_col_idx, s_nnz*sizeof(int)),       "malloc col_idx");
    cuda_check(cudaMemcpy(s_d_row_ptr, row_ptr_h, (dim+1)*sizeof(int), cudaMemcpyHostToDevice), "cpy row_ptr");
    cuda_check(cudaMemcpy(s_d_col_idx, col_idx_h, s_nnz*sizeof(int),   cudaMemcpyHostToDevice), "cpy col_idx");

    cuda_check(cudaMalloc(&s_d_sorted_obs, n_obs*sizeof(int)),    "malloc sorted_obs");
    cuda_check(cudaMemcpy(s_d_sorted_obs, sorted_obs_h, n_obs*sizeof(int), cudaMemcpyHostToDevice), "cpy sorted_obs");

    cuda_check(cudaMalloc(&s_d_edge_j,  n_edges*sizeof(int)),     "malloc edge_j");
    cuda_check(cudaMalloc(&s_d_edge_k,  n_edges*sizeof(int)),     "malloc edge_k");
    cuda_check(cudaMalloc(&s_d_edge_lm, n_edges*sizeof(int)),     "malloc edge_lm");
    cuda_check(cudaMemcpy(s_d_edge_j,  edge_j_h,  n_edges*sizeof(int), cudaMemcpyHostToDevice), "cpy edge_j");
    cuda_check(cudaMemcpy(s_d_edge_k,  edge_k_h,  n_edges*sizeof(int), cudaMemcpyHostToDevice), "cpy edge_k");
    cuda_check(cudaMemcpy(s_d_edge_lm, edge_lm_h, n_edges*sizeof(int), cudaMemcpyHostToDevice), "cpy edge_lm");

    cuda_check(cudaMalloc(&s_d_lm_pos,   n_lm_active*sizeof(int)), "malloc lm_pos");
    cuda_check(cudaMalloc(&s_d_lm_count, n_lm_active*sizeof(int)), "malloc lm_count");
    cuda_check(cudaMemcpy(s_d_lm_pos,   lm_pos_h,   n_lm_active*sizeof(int), cudaMemcpyHostToDevice), "cpy lm_pos");
    cuda_check(cudaMemcpy(s_d_lm_count, lm_count_h, n_lm_active*sizeof(int), cudaMemcpyHostToDevice), "cpy lm_count");

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

    cuda_check(cudaMalloc(&s_d_hat_x, (size_t)n_poses * 3 * sizeof(double)), "malloc hat_x");
    cuda_check(cudaMalloc(&s_d_csr_vals_d, (size_t)s_nnz * sizeof(double)), "malloc csr_vals_d");

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

    /* ---- Persistent PCG working buffers (float) ---- */
    int n_blocks = (int)dim / 3;
    cuda_check(cudaMalloc(&s_pcg_x,        (size_t)dim * sizeof(float)),       "malloc pcg_x");
    cuda_check(cudaMalloc(&s_pcg_r,        (size_t)dim * sizeof(float)),       "malloc pcg_r");
    cuda_check(cudaMalloc(&s_pcg_z,        (size_t)dim * sizeof(float)),       "malloc pcg_z");
    cuda_check(cudaMalloc(&s_pcg_p,        (size_t)dim * sizeof(float)),       "malloc pcg_p");
    cuda_check(cudaMalloc(&s_pcg_q,        (size_t)dim * sizeof(float)),       "malloc pcg_q");
    cuda_check(cudaMalloc(&s_pcg_diag_inv, (size_t)n_blocks * 9 * sizeof(float)), "malloc pcg_diag_inv");
    cuda_check(cudaMalloc(&s_pcg_xi_f,     (size_t)dim * sizeof(float)),       "malloc pcg_xi_f");

    /* ---- cuBLAS / cuSPARSE ---- */
    cublasCreate(&s_bl_handle);
    cusparseCreate(&s_sp_handle);

    float *tmp_vals = NULL;
    cuda_check(cudaMalloc(&tmp_vals, (size_t)s_nnz * sizeof(float)), "malloc tmp_vals");
    cuda_check(cudaMemset(tmp_vals, 0, (size_t)s_nnz * sizeof(float)), "memset tmp_vals");

    cusparseCreateCsr(&s_mat_A, (int64_t)dim, (int64_t)dim, (int64_t)s_nnz,
                      s_d_row_ptr, s_d_col_idx, tmp_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&s_vec_p, dim, s_pcg_p, CUDA_R_32F);
    cusparseCreateDnVec(&s_vec_q, dim, s_pcg_q, CUDA_R_32F);

    float alpha_tmp = 1.0f, beta_tmp = 0.0f;
    size_t buf_size = 0;
    cusparseSpMV_bufferSize(s_sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_tmp, s_mat_A, s_vec_p, &beta_tmp, s_vec_q,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);
    if (buf_size > 0)
        cuda_check(cudaMalloc(&s_spmv_buf, buf_size), "malloc spmv_buf");

    cudaFree(tmp_vals);

    free(row_ptr_h); free(col_idx_h); free(sorted_obs_h);
    free(edge_j_h);  free(edge_k_h);  free(edge_lm_h);
    free(lm_pos_h);  free(lm_count_h);
    free(z_range_h); free(z_bearing_h); free(z_step_h);

    *nnz_out = s_nnz;
}

/* =========================================================================
 * device_graph_update
 * ========================================================================= */
void device_graph_update(
    struct HAT_X *hat_xs,
    unsigned int  n_poses,
    float        *d_csr_vals,
    double       *d_xi
)
{
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

    /* Zero double CSR buffer and d_xi on device */
    cuda_check(cudaMemset(s_d_csr_vals_d, 0, (size_t)s_nnz * sizeof(double)), "zero csr_vals_d");
    cuda_check(cudaMemset(d_xi,           0, (size_t)s_dim  * sizeof(double)), "zero xi");

    /* Launch MotionEdge kernel (accumulates into double CSR buffer) */
    {
        int block = 256;
        int grid  = (s_n_poses - 1 + block - 1) / block;
        if (grid < 1) grid = 1;
        compute_motion_edges_kernel_f<<<grid, block>>>(
            s_d_hat_x,
            s_d_us_nu,
            s_d_us_omega,
            s_delta,
            s_lambda,
            s_mns[0], s_mns[1], s_mns[2], s_mns[3],
            s_d_csr_vals_d,
            d_xi,
            s_d_row_ptr,
            s_d_col_idx,
            s_n_poses
        );
        cuda_check(cudaGetLastError(), "compute_motion_edges_kernel_f");
    }

    if (s_n_edges > 0) {
        int block = 256;
        int grid  = (s_n_edges + block - 1) / block;
        compute_and_scatter_edges_kernel_f<<<grid, block>>>(
            s_d_hat_x,
            s_d_z_range,
            s_d_z_bearing,
            s_d_z_step,
            s_d_sorted_obs,
            s_d_lm_pos,
            s_d_edge_j,
            s_d_edge_k,
            s_d_edge_lm,
            s_d_csr_vals_d,
            d_xi,
            s_d_row_ptr,
            s_d_col_idx,
            s_n_edges
        );
        cuda_check(cudaGetLastError(), "compute_and_scatter_edges_kernel_f");
    }

    /* Cast double CSR buffer → float CSR buffer for PCG */
    {
        int block = 256;
        int grid  = (s_nnz + block - 1) / block;
        double_to_float_dg<<<grid, block>>>(s_d_csr_vals_d, d_csr_vals, s_nnz);
        cuda_check(cudaGetLastError(), "cast csr d2f");
    }

    cuda_check(cudaDeviceSynchronize(), "kernel sync");
}

/* =========================================================================
 * Accessors
 * ========================================================================= */
int *device_graph_get_row_ptr(void) { return s_d_row_ptr; }
int *device_graph_get_col_idx(void) { return s_d_col_idx; }

/* =========================================================================
 * device_graph_free
 * ========================================================================= */
void device_graph_free(void)
{
    if (s_d_row_ptr)   { cudaFree(s_d_row_ptr);   s_d_row_ptr   = NULL; }
    if (s_d_col_idx)   { cudaFree(s_d_col_idx);   s_d_col_idx   = NULL; }
    if (s_d_hat_x)     { cudaFree(s_d_hat_x);     s_d_hat_x     = NULL; }
    if (s_d_sorted_obs){ cudaFree(s_d_sorted_obs); s_d_sorted_obs= NULL; }
    if (s_d_edge_j)    { cudaFree(s_d_edge_j);    s_d_edge_j    = NULL; }
    if (s_d_edge_k)    { cudaFree(s_d_edge_k);    s_d_edge_k    = NULL; }
    if (s_d_edge_lm)   { cudaFree(s_d_edge_lm);   s_d_edge_lm   = NULL; }
    if (s_d_lm_pos)    { cudaFree(s_d_lm_pos);    s_d_lm_pos    = NULL; }
    if (s_d_lm_count)  { cudaFree(s_d_lm_count);  s_d_lm_count  = NULL; }
    if (s_d_z_range)    { cudaFree(s_d_z_range);    s_d_z_range    = NULL; }
    if (s_d_z_bearing)  { cudaFree(s_d_z_bearing);  s_d_z_bearing  = NULL; }
    if (s_d_z_step)     { cudaFree(s_d_z_step);     s_d_z_step     = NULL; }
    if (s_d_us_nu)       { cudaFree(s_d_us_nu);       s_d_us_nu       = NULL; }
    if (s_d_us_omega)    { cudaFree(s_d_us_omega);    s_d_us_omega    = NULL; }
    if (s_d_csr_vals_d)  { cudaFree(s_d_csr_vals_d);  s_d_csr_vals_d  = NULL; }

    if (s_pcg_x)        { cudaFree(s_pcg_x);        s_pcg_x        = NULL; }
    if (s_pcg_r)        { cudaFree(s_pcg_r);        s_pcg_r        = NULL; }
    if (s_pcg_z)        { cudaFree(s_pcg_z);        s_pcg_z        = NULL; }
    if (s_pcg_p)        { cudaFree(s_pcg_p);        s_pcg_p        = NULL; }
    if (s_pcg_q)        { cudaFree(s_pcg_q);        s_pcg_q        = NULL; }
    if (s_pcg_diag_inv) { cudaFree(s_pcg_diag_inv); s_pcg_diag_inv = NULL; }
    if (s_pcg_xi_f)     { cudaFree(s_pcg_xi_f);     s_pcg_xi_f     = NULL; }
    if (s_spmv_buf)     { cudaFree(s_spmv_buf);     s_spmv_buf     = NULL; }

    if (s_vec_p)    { cusparseDestroyDnVec(s_vec_p);  s_vec_p    = NULL; }
    if (s_vec_q)    { cusparseDestroyDnVec(s_vec_q);  s_vec_q    = NULL; }
    if (s_mat_A)    { cusparseDestroySpMat(s_mat_A);  s_mat_A    = NULL; }
    if (s_sp_handle){ cusparseDestroy(s_sp_handle);   s_sp_handle = NULL; }
    if (s_bl_handle){ cublasDestroy(s_bl_handle);     s_bl_handle = NULL; }
}
