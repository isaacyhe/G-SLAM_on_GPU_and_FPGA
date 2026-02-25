/*
 * device_graph.cpp  —  Device-resident edge computation and CSR assembly
 *                      for fpga_pcg_fp32 (SYCL/oneAPI, FP32)
 *
 * Architecture is identical to fpga_pcg_fp64/device_graph.cpp except:
 *   - All ObsEdge math in the SYCL kernel is done in float.
 *   - CSR values (buf_csr_vals, csr_vals_host) are float.
 *   - hat_x buffer is float.
 *   - z_range / z_bearing buffers are float.
 *   - MotionEdge scatter computed in double on CPU then stored as float.
 *   - xi is accumulated as float on device; xi_host is double (matching the
 *     original GSLAM.c which builds xi_d in double then casts to float).
 *
 * Compile with:   icpx -O3 -fsycl *.c *.cpp -lm -o GSLAM.cpu_test
 */

#include "device_graph.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>

/* =========================================================================
 * Internal helper: binary search for a CSR slot (host-side).
 * ========================================================================= */
static int find_slot_host(const int *row_ptr, const int *col_idx,
                          int row, int col)
{
    int lo = row_ptr[row], hi = row_ptr[row + 1] - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (col_idx[mid] == col) return mid;
        if (col_idx[mid] < col)  lo = mid + 1;
        else                     hi = mid - 1;
    }
    return -1;
}

/* =========================================================================
 * DeviceGraph::init
 * ========================================================================= */
void DeviceGraph::init(struct Z *zlist, unsigned int n_obs_in,
                       unsigned int n_poses_in, unsigned int dim_in,
                       sycl::queue &q)
{
    n_obs    = (int)n_obs_in;
    n_poses  = (int)n_poses_in;
    dim      = (int)dim_in;

    /* ------------------------------------------------------------------
     * 1.  Find distinct landmark IDs and sort observations by landmark.
     *     Replicates the same bucket-sort as make_edges() / init FP64.
     * ------------------------------------------------------------------ */
    unsigned int max_id = 0;
    for (int i = 0; i < n_obs; i++)
        if (zlist[i].landmark_id > max_id)
            max_id = zlist[i].landmark_id;

    unsigned int n_ids = max_id + 1;

    unsigned int *id_count = (unsigned int *)calloc(n_ids, sizeof(unsigned int));
    for (int i = 0; i < n_obs; i++)
        id_count[zlist[i].landmark_id]++;

    unsigned int *id_list = (unsigned int *)malloc(n_ids * sizeof(unsigned int));
    unsigned int *temp_ids = (unsigned int *)malloc(n_obs * sizeof(unsigned int));
    for (unsigned int i = 0; i < n_ids; i++) id_list[i] = n_ids;
    for (int i = 0; i < n_obs; i++) temp_ids[i] = zlist[i].landmark_id;

    for (unsigned int i = 0; i < n_ids; i++) {
        for (int j = 0; j < n_obs; j++) {
            if (id_list[i] == n_ids && temp_ids[j] != n_ids) {
                id_list[i] = temp_ids[j];
                temp_ids[j] = n_ids;
            } else if (id_list[i] != n_ids && id_list[i] == temp_ids[j]) {
                temp_ids[j] = n_ids;
            }
        }
    }
    free(temp_ids);

    n_lm = 0;
    n_obs_edges = 0;
    for (unsigned int i = 0; i < n_ids; i++) {
        if (id_list[i] == n_ids) continue;
        unsigned int cnt = id_count[id_list[i]];
        n_lm++;
        if (cnt >= 2)
            n_obs_edges += (int)(cnt * (cnt - 1) / 2);
    }

    lm_pos   = (int *)malloc(n_lm * sizeof(int));
    lm_count = (int *)malloc(n_lm * sizeof(int));
    int *active_ids = (int *)malloc(n_lm * sizeof(int));

    {
        int lm_idx = 0, pos = 0;
        for (unsigned int i = 0; i < n_ids; i++) {
            if (id_list[i] == n_ids) continue;
            active_ids[lm_idx] = (int)id_list[i];
            lm_pos[lm_idx]   = pos;
            lm_count[lm_idx] = (int)id_count[id_list[i]];
            pos += lm_count[lm_idx];
            lm_idx++;
        }
    }

    sorted_obs = (int *)malloc(n_obs * sizeof(int));
    {
        int pos = 0;
        for (int lm_idx = 0; lm_idx < n_lm; lm_idx++) {
            int lid = active_ids[lm_idx];
            for (int j = 0; j < n_obs; j++) {
                if ((int)zlist[j].landmark_id == lid) {
                    sorted_obs[pos++] = j;
                }
            }
        }
    }

    /* ------------------------------------------------------------------
     * 2.  Build flat observation arrays (float) and edge index arrays.
     * ------------------------------------------------------------------ */
    z_range   = (float *)malloc(n_obs * sizeof(float));
    z_bearing = (float *)malloc(n_obs * sizeof(float));
    z_step    = (int   *)malloc(n_obs * sizeof(int));
    for (int i = 0; i < n_obs; i++) {
        z_range[i]   = (float)zlist[i].z[0];
        z_bearing[i] = (float)zlist[i].z[1];
        z_step[i]    = (int)zlist[i].step;
    }

    edge_lm = (int *)malloc(n_obs_edges * sizeof(int));
    edge_j  = (int *)malloc(n_obs_edges * sizeof(int));
    edge_k  = (int *)malloc(n_obs_edges * sizeof(int));

    {
        int eidx = 0;
        for (int lm_idx = 0; lm_idx < n_lm; lm_idx++) {
            int cnt = lm_count[lm_idx];
            for (int j = 0; j < cnt; j++) {
                for (int k = j + 1; k < cnt; k++) {
                    edge_lm[eidx] = lm_idx;
                    edge_j[eidx]  = j;
                    edge_k[eidx]  = k;
                    eidx++;
                }
            }
        }
    }

    /* ------------------------------------------------------------------
     * 3.  Build CSR structure (row_ptr, col_idx).
     *     Include MotionEdge pairs (i, i+1) and anchor diagonal [0..2].
     *
     * Sparse approach: collect (row,col) pairs, sort, deduplicate.
     * Avoids the O(dim^2) dense marker which OOMs for large dim.
     * ------------------------------------------------------------------ */
    size_t max_pairs = 3
        + (size_t)n_obs_edges * 36
        + (size_t)(n_poses > 1 ? n_poses - 1 : 0) * 36;

    int *pair_row = (int *)malloc(max_pairs * sizeof(int));
    int *pair_col = (int *)malloc(max_pairs * sizeof(int));
    size_t np = 0;

    for (int d = 0; d < 3; d++) { pair_row[np] = d; pair_col[np] = d; np++; }

    for (int eidx = 0; eidx < n_obs_edges; eidx++) {
        int lm_idx = edge_lm[eidx];
        int pos    = lm_pos[lm_idx];
        int j      = edge_j[eidx];
        int k      = edge_k[eidx];
        int idx_j  = sorted_obs[pos + j];
        int idx_k  = sorted_obs[pos + k];
        int f1 = z_step[idx_j] * 3;
        int f2 = z_step[idx_k] * 3;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
            pair_row[np]=f1+r; pair_col[np]=f1+c; np++;
            pair_row[np]=f1+r; pair_col[np]=f2+c; np++;
            pair_row[np]=f2+r; pair_col[np]=f1+c; np++;
            pair_row[np]=f2+r; pair_col[np]=f2+c; np++;
        }
    }

    for (int i = 0; i < n_poses - 1; i++) {
        int f1 = i * 3, f2 = (i + 1) * 3;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
            pair_row[np]=f1+r; pair_col[np]=f1+c; np++;
            pair_row[np]=f1+r; pair_col[np]=f2+c; np++;
            pair_row[np]=f2+r; pair_col[np]=f1+c; np++;
            pair_row[np]=f2+r; pair_col[np]=f2+c; np++;
        }
    }

    long long *keys = (long long *)malloc(np * sizeof(long long));
    for (size_t i = 0; i < np; i++)
        keys[i] = (long long)pair_row[i] * (dim + 1) + pair_col[i];
    free(pair_row);
    free(pair_col);

    qsort(keys, np, sizeof(long long),
          [](const void *a, const void *b) -> int {
              long long x = *(const long long *)a;
              long long y = *(const long long *)b;
              return (x > y) - (x < y);
          });

    row_ptr = (int *)calloc(dim + 2, sizeof(int));
    long long prev = -1;
    size_t n_unique = 0;
    for (size_t i = 0; i < np; i++) {
        if (keys[i] == prev) continue;
        prev = keys[i];
        int row = (int)(keys[i] / (dim + 1));
        row_ptr[row + 1]++;
        n_unique++;
    }
    nnz = (int)n_unique;
    for (int i = 0; i < dim; i++)
        row_ptr[i + 1] += row_ptr[i];

    col_idx = (int *)malloc(nnz * sizeof(int));
    int *fill = (int *)calloc(dim, sizeof(int));
    prev = -1;
    for (size_t i = 0; i < np; i++) {
        if (keys[i] == prev) continue;
        prev = keys[i];
        int row = (int)(keys[i] / (dim + 1));
        int col = (int)(keys[i] % (dim + 1));
        col_idx[row_ptr[row] + fill[row]++] = col;
    }
    free(fill);
    free(keys);

    /* ------------------------------------------------------------------
     * 4.  Allocate per-round staging and output arrays.
     * ------------------------------------------------------------------ */
    motion_csr_vals_f = (float  *)calloc(nnz, sizeof(float));
    motion_xi_f       = (float  *)calloc(dim,  sizeof(float));
    csr_vals_host     = (float  *)calloc(nnz, sizeof(float));
    xi_host           = (double *)calloc(dim,  sizeof(double));

    /* ------------------------------------------------------------------
     * 5.  Allocate persistent SYCL buffers for fixed data and upload.
     * ------------------------------------------------------------------ */
    buf_sorted_obs = new sycl::buffer<int,1>(sorted_obs, sycl::range<1>{(size_t)n_obs});
    buf_edge_lm    = new sycl::buffer<int,1>(edge_lm,    sycl::range<1>{(size_t)n_obs_edges});
    buf_edge_j     = new sycl::buffer<int,1>(edge_j,     sycl::range<1>{(size_t)n_obs_edges});
    buf_edge_k     = new sycl::buffer<int,1>(edge_k,     sycl::range<1>{(size_t)n_obs_edges});
    buf_lm_pos     = new sycl::buffer<int,1>(lm_pos,     sycl::range<1>{(size_t)n_lm});
    buf_lm_count   = new sycl::buffer<int,1>(lm_count,   sycl::range<1>{(size_t)n_lm});
    buf_z_range    = new sycl::buffer<float,1>(z_range,   sycl::range<1>{(size_t)n_obs});
    buf_z_bearing  = new sycl::buffer<float,1>(z_bearing, sycl::range<1>{(size_t)n_obs});
    buf_z_step     = new sycl::buffer<int,1>(z_step,     sycl::range<1>{(size_t)n_obs});
    buf_row_ptr    = new sycl::buffer<int,1>(row_ptr,    sycl::range<1>{(size_t)(dim+1)});
    buf_col_idx    = new sycl::buffer<int,1>(col_idx,    sycl::range<1>{(size_t)nnz});

    buf_hat_x    = new sycl::buffer<float,1>(sycl::range<1>{(size_t)(3 * n_poses)});
    buf_csr_vals = new sycl::buffer<float,1>(sycl::range<1>{(size_t)nnz});
    buf_xi       = new sycl::buffer<float,1>(sycl::range<1>{(size_t)dim});

    int n_blocks = dim / 3;
    buf_diag_inv = new sycl::buffer<float,1>(sycl::range<1>{(size_t)(n_blocks * 9)});
    buf_pcg_x    = new sycl::buffer<float,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_r    = new sycl::buffer<float,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_z    = new sycl::buffer<float,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_p    = new sycl::buffer<float,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_q    = new sycl::buffer<float,1>(sycl::range<1>{(size_t)dim});

    free(active_ids);
    free(id_count);
    free(id_list);
}

/* =========================================================================
 * DeviceGraph::update
 * ========================================================================= */
void DeviceGraph::update(struct HAT_X *hat_xs,
                         struct U *us, double delta, double lambda,
                         const double *mns,
                         sycl::queue &q)
{
    /* ------------------------------------------------------------------
     * A.  Upload hat_xs as float to the per-round device buffer.
     * ------------------------------------------------------------------ */
    {
        auto h_hx = buf_hat_x->get_host_access();
        for (int i = 0; i < n_poses; i++) {
            h_hx[i*3+0] = (float)hat_xs[i].hat_x[0];
            h_hx[i*3+1] = (float)hat_xs[i].hat_x[1];
            h_hx[i*3+2] = (float)hat_xs[i].hat_x[2];
        }
    }

    /* ------------------------------------------------------------------
     * B.  MotionEdge CSR scatter on CPU (double math, stored as float).
     *     xi from MotionEdge kept as double here, then cast to float
     *     for the device init.
     * ------------------------------------------------------------------ */
    memset(motion_csr_vals_f, 0, (size_t)nnz * sizeof(float));
    memset(motion_xi_f,       0, (size_t)dim  * sizeof(float));

    /* Anchor */
    for (int d = 0; d < 3; d++) {
        int slot = find_slot_host(row_ptr, col_idx, d, d);
        if (slot >= 0) motion_csr_vals_f[slot] += 1000000.0f;
    }

    /* Temporary double xi buffer for MotionEdge accumulation */
    double *motion_xi_d = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < n_poses - 1; i++) {
        int t1 = i, t2 = i + 1;
        int f1 = t1 * 3, f2 = t2 * 3;

        double nu    = us[t2].nu;
        double omega = us[t2].omega;
        if (std::fabs(omega) < 1e-5) omega = 1e-5;

        double mns0 = mns[0], mns1 = mns[1], mns2 = mns[2], mns3 = mns[3];

        double M00 = mns0*mns0*std::fabs(nu)/delta + mns1*mns1*std::fabs(omega)/delta;
        double M11 = mns2*mns2*std::fabs(nu)/delta + mns3*mns3*std::fabs(omega)/delta;

        double t0  = hat_xs[t1].hat_x[2];
        double st  = std::sin(t0);
        double ct  = std::cos(t0);
        double stw = std::sin(t0 + omega * delta);
        double ctw = std::cos(t0 + omega * delta);

        double A[6];
        A[0] = stw/omega - st/omega;
        A[1] = nu/omega*delta*ctw - nu/omega/omega*stw + nu/omega/omega*st;
        A[2] = ct/omega - ctw/omega;
        A[3] = nu/omega*delta*stw - nu/omega/omega*ct + nu/omega/omega*ctw;
        A[4] = 0.0;
        A[5] = delta;

        double F[9] = {1,0,0, 0,1,0, 0,0,1};
        F[0*3+2] = nu/omega*std::cos(t0+omega*delta) - nu/omega*std::cos(t0);
        F[1*3+2] = nu/omega*std::sin(t0+omega*delta) - nu/omega*std::sin(t0);

        double AM[6];
        for (int r = 0; r < 3; r++) {
            AM[r*2+0] = A[r*2+0] * M00;
            AM[r*2+1] = A[r*2+1] * M11;
        }
        double AMAT[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                double s = 0.0;
                for (int k2 = 0; k2 < 2; k2++)
                    s += AM[r*2+k2] * A[c*2+k2];
                AMAT[r*3+c] = s;
            }
        for (int d = 0; d < 3; d++) AMAT[d*3+d] += 0.0001;

        double det = AMAT[0]*(AMAT[4]*AMAT[8]-AMAT[5]*AMAT[7])
                   - AMAT[1]*(AMAT[3]*AMAT[8]-AMAT[5]*AMAT[6])
                   + AMAT[2]*(AMAT[3]*AMAT[7]-AMAT[4]*AMAT[6]);
        if (det == 0.0) det = 1e-30;
        double Om[9];
        Om[0] = (AMAT[4]*AMAT[8]-AMAT[5]*AMAT[7])/det;
        Om[1] = (AMAT[2]*AMAT[7]-AMAT[1]*AMAT[8])/det;
        Om[2] = (AMAT[1]*AMAT[5]-AMAT[2]*AMAT[4])/det;
        Om[3] = (AMAT[5]*AMAT[6]-AMAT[3]*AMAT[8])/det;
        Om[4] = (AMAT[0]*AMAT[8]-AMAT[2]*AMAT[6])/det;
        Om[5] = (AMAT[2]*AMAT[3]-AMAT[0]*AMAT[5])/det;
        Om[6] = (AMAT[3]*AMAT[7]-AMAT[4]*AMAT[6])/det;
        Om[7] = (AMAT[1]*AMAT[6]-AMAT[0]*AMAT[7])/det;
        Om[8] = (AMAT[0]*AMAT[4]-AMAT[1]*AMAT[3])/det;

        double FtOm[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                double s = 0.0;
                for (int k2 = 0; k2 < 3; k2++)
                    s += F[k2*3+r] * Om[k2*3+c];
                FtOm[r*3+c] = s;
            }

        double FtOmF[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                double s = 0.0;
                for (int k2 = 0; k2 < 3; k2++)
                    s += FtOm[r*3+k2] * F[k2*3+c];
                FtOmF[r*3+c] = s;
            }

        double OmF[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                double s = 0.0;
                for (int k2 = 0; k2 < 3; k2++)
                    s += Om[r*3+k2] * F[k2*3+c];
                OmF[r*3+c] = s;
            }

        double x2_pred[3];
        {
            double t0p = hat_xs[t1].hat_x[2];
            if (std::fabs(omega) < 1e-10) {
                x2_pred[0] = hat_xs[t1].hat_x[0] + nu*std::cos(t0p)*delta;
                x2_pred[1] = hat_xs[t1].hat_x[1] + nu*std::sin(t0p)*delta;
                x2_pred[2] = hat_xs[t1].hat_x[2] + omega*delta;
            } else {
                x2_pred[0] = hat_xs[t1].hat_x[0] + nu/omega*(std::sin(t0p+omega*delta)-std::sin(t0p));
                x2_pred[1] = hat_xs[t1].hat_x[1] + nu/omega*(std::cos(t0p)-std::cos(t0p+omega*delta));
                x2_pred[2] = hat_xs[t1].hat_x[2] + omega*delta;
            }
        }
        double he[3];
        for (int d = 0; d < 3; d++)
            he[d] = hat_xs[t2].hat_x[d] - x2_pred[d];

        double FtOmhe[3], Omhe[3];
        for (int r = 0; r < 3; r++) {
            double s1 = 0.0, s2 = 0.0;
            for (int k2 = 0; k2 < 3; k2++) {
                s1 += FtOm[r*3+k2] * he[k2];
                s2 += Om[r*3+k2]   * he[k2];
            }
            FtOmhe[r] = s1;
            Omhe[r]   = s2;
        }

        /* Scatter into motion_csr_vals_f (cast to float on store) */
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                int sl;
                sl = find_slot_host(row_ptr, col_idx, f1+r, f1+c);
                if (sl >= 0) motion_csr_vals_f[sl] += (float)(lambda * FtOmF[r*3+c]);
                sl = find_slot_host(row_ptr, col_idx, f1+r, f2+c);
                if (sl >= 0) motion_csr_vals_f[sl] += (float)(-lambda * FtOm[r*3+c]);
                sl = find_slot_host(row_ptr, col_idx, f2+r, f1+c);
                if (sl >= 0) motion_csr_vals_f[sl] += (float)(-lambda * OmF[r*3+c]);
                sl = find_slot_host(row_ptr, col_idx, f2+r, f2+c);
                if (sl >= 0) motion_csr_vals_f[sl] += (float)(lambda * Om[r*3+c]);
            }
            motion_xi_d[f1+r] +=  lambda * FtOmhe[r];
            motion_xi_d[f2+r] += -lambda * Omhe[r];
        }
    }

    /* Cast motion_xi_d to float for device init */
    for (int i = 0; i < dim; i++)
        motion_xi_f[i] = (float)motion_xi_d[i];
    free(motion_xi_d);

    /* Upload initial csr_vals and xi to device buffers */
    {
        auto h_cv = buf_csr_vals->get_host_access();
        for (int i = 0; i < nnz; i++) h_cv[i] = motion_csr_vals_f[i];
    }
    {
        auto h_xi = buf_xi->get_host_access();
        for (int i = 0; i < dim; i++) h_xi[i] = motion_xi_f[i];
    }

    /* ------------------------------------------------------------------
     * C.  Launch fused ObsEdge kernel (float).
     * ------------------------------------------------------------------ */
    int _n_obs_edges = n_obs_edges;

    q.submit([&](sycl::handler &h) {
        auto sorted_obs_acc = buf_sorted_obs->get_access<sycl::access::mode::read>(h);
        auto edge_lm_acc    = buf_edge_lm->get_access<sycl::access::mode::read>(h);
        auto edge_j_acc     = buf_edge_j->get_access<sycl::access::mode::read>(h);
        auto edge_k_acc     = buf_edge_k->get_access<sycl::access::mode::read>(h);
        auto lm_pos_acc     = buf_lm_pos->get_access<sycl::access::mode::read>(h);
        auto z_range_acc    = buf_z_range->get_access<sycl::access::mode::read>(h);
        auto z_bearing_acc  = buf_z_bearing->get_access<sycl::access::mode::read>(h);
        auto z_step_acc     = buf_z_step->get_access<sycl::access::mode::read>(h);
        auto row_ptr_acc    = buf_row_ptr->get_access<sycl::access::mode::read>(h);
        auto col_idx_acc    = buf_col_idx->get_access<sycl::access::mode::read>(h);
        auto hat_x_acc      = buf_hat_x->get_access<sycl::access::mode::read>(h);
        auto csr_vals_acc   = buf_csr_vals->get_access<sycl::access::mode::read_write>(h);
        auto xi_acc         = buf_xi->get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>{(size_t)_n_obs_edges},
            [=](sycl::id<1> tid_id)
#ifdef FPGA_HARDWARE
            [[intel::scheduler_target_fmax_mhz(400)]]
#endif
        {
            int tid = (int)tid_id[0];

            int lm   = edge_lm_acc[tid];
            int pos  = lm_pos_acc[lm];
            int j    = edge_j_acc[tid];
            int k    = edge_k_acc[tid];

            int idx_j = sorted_obs_acc[pos + j];
            int idx_k = sorted_obs_acc[pos + k];

            int step1 = z_step_acc[idx_j];
            int step2 = z_step_acc[idx_k];

            float r1 = z_range_acc[idx_j],   b1 = z_bearing_acc[idx_j];
            float r2 = z_range_acc[idx_k],   b2 = z_bearing_acc[idx_k];

            float theta1 = hat_x_acc[step1*3+2];
            float theta2 = hat_x_acc[step2*3+2];

            float s1 = sycl::sin(theta1+b1), c1 = sycl::cos(theta1+b1);
            float s2 = sycl::sin(theta2+b2), c2 = sycl::cos(theta2+b2);

            float ex = hat_x_acc[step2*3+0] - hat_x_acc[step1*3+0] + r2*c2 - r1*c1;
            float ey = hat_x_acc[step2*3+1] - hat_x_acc[step1*3+1] + r2*s2 - r1*s1;

            const float snr0 = 0.14f, snr1 = 0.05f;
            float q1r = r1*snr0, q1b = snr1, q2r = r2*snr0, q2b = snr1;

            float a00 = c1*c1*(q1r*q1r) + r1*r1*s1*s1*(q1b*q1b);
            float a01 = c1*s1*(q1r*q1r) - r1*r1*s1*c1*(q1b*q1b);
            float a11 = s1*s1*(q1r*q1r) + r1*r1*c1*c1*(q1b*q1b);
            float b00 = c2*c2*(q2r*q2r) + r2*r2*s2*s2*(q2b*q2b);
            float b01 = c2*s2*(q2r*q2r) - r2*r2*s2*c2*(q2b*q2b);
            float b11 = s2*s2*(q2r*q2r) + r2*r2*c2*c2*(q2b*q2b);

            float sig00 = a00+b00, sig01 = a01+b01, sig11 = a11+b11;
            float det = sig00*sig11 - sig01*sig01;
            if (det == 0.0f) det = 1e-30f;
            float inv_d = 1.0f/det;
            float om00 = sig11*inv_d, om01 = -sig01*inv_d, om11 = sig00*inv_d;

            float OB1[6], OB2[6];
            OB1[0] = om00*(-1.0f) + om01*(0.0f);
            OB1[1] = om00*( 0.0f) + om01*(-1.0f);
            OB1[2] = om00*(r1*s1) + om01*(-r1*c1);
            OB1[3] = om01*(-1.0f) + om11*(0.0f);
            OB1[4] = om01*( 0.0f) + om11*(-1.0f);
            OB1[5] = om01*(r1*s1) + om11*(-r1*c1);
            OB2[0] = om00*( 1.0f) + om01*( 0.0f);
            OB2[1] = om00*( 0.0f) + om01*( 1.0f);
            OB2[2] = om00*(-r2*s2) + om01*(r2*c2);
            OB2[3] = om01*( 1.0f) + om11*( 0.0f);
            OB2[4] = om01*( 0.0f) + om11*( 1.0f);
            OB2[5] = om01*(-r2*s2) + om11*(r2*c2);

            float tB1[6], tB2[6];
            tB1[0]=-1.0f;  tB1[1]=0.0f;
            tB1[2]=0.0f;   tB1[3]=-1.0f;
            tB1[4]=r1*s1;  tB1[5]=-r1*c1;
            tB2[0]=1.0f;   tB2[1]=0.0f;
            tB2[2]=0.0f;   tB2[3]=1.0f;
            tB2[4]=-r2*s2; tB2[5]=r2*c2;

            float ul[9], ur[9], bl[9], br[9];
#ifdef FPGA_HARDWARE
            #pragma unroll
#endif
            for (int r = 0; r < 3; r++) {
#ifdef FPGA_HARDWARE
                #pragma unroll
#endif
                for (int c = 0; c < 3; c++) {
                    ul[r*3+c] = tB1[r*2]*OB1[c]   + tB1[r*2+1]*OB1[3+c];
                    ur[r*3+c] = tB1[r*2]*OB2[c]   + tB1[r*2+1]*OB2[3+c];
                    bl[r*3+c] = tB2[r*2]*OB1[c]   + tB2[r*2+1]*OB1[3+c];
                    br[r*3+c] = tB2[r*2]*OB2[c]   + tB2[r*2+1]*OB2[3+c];
                }
            }

            float Oe0 = om00*ex + om01*ey;
            float Oe1 = om01*ex + om11*ey;
            float xiu[3], xib[3];
            xiu[0] = -(tB1[0]*Oe0+tB1[1]*Oe1);
            xiu[1] = -(tB1[2]*Oe0+tB1[3]*Oe1);
            xiu[2] = -(tB1[4]*Oe0+tB1[5]*Oe1);
            xib[0] = -(tB2[0]*Oe0+tB2[1]*Oe1);
            xib[1] = -(tB2[2]*Oe0+tB2[3]*Oe1);
            xib[2] = -(tB2[4]*Oe0+tB2[5]*Oe1);

            int f1 = step1*3, f2 = step2*3;

            using ar_t = sycl::atomic_ref<float,
#ifdef FPGA_HARDWARE
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
#else
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
#endif
                             sycl::access::address_space::global_space>;

#ifdef FPGA_HARDWARE
            #pragma unroll
#endif
            for (int r = 0; r < 3; r++) {
#ifdef FPGA_HARDWARE
                #pragma unroll
#endif
                for (int c = 0; c < 3; c++) {
                    /* ul -> (f1+r, f1+c) */
                    {
                        int lo = row_ptr_acc[f1+r], hi = row_ptr_acc[f1+r+1]-1;
                        while (lo <= hi) { int mid=(lo+hi)/2;
                            if (col_idx_acc[mid]==f1+c) { ar_t(csr_vals_acc[mid]).fetch_add(ul[r*3+c]); break; }
                            if (col_idx_acc[mid]<f1+c) lo=mid+1; else hi=mid-1; }
                    }
                    /* ur -> (f1+r, f2+c) */
                    {
                        int lo = row_ptr_acc[f1+r], hi = row_ptr_acc[f1+r+1]-1;
                        while (lo <= hi) { int mid=(lo+hi)/2;
                            if (col_idx_acc[mid]==f2+c) { ar_t(csr_vals_acc[mid]).fetch_add(ur[r*3+c]); break; }
                            if (col_idx_acc[mid]<f2+c) lo=mid+1; else hi=mid-1; }
                    }
                    /* bl -> (f2+r, f1+c) */
                    {
                        int lo = row_ptr_acc[f2+r], hi = row_ptr_acc[f2+r+1]-1;
                        while (lo <= hi) { int mid=(lo+hi)/2;
                            if (col_idx_acc[mid]==f1+c) { ar_t(csr_vals_acc[mid]).fetch_add(bl[r*3+c]); break; }
                            if (col_idx_acc[mid]<f1+c) lo=mid+1; else hi=mid-1; }
                    }
                    /* br -> (f2+r, f2+c) */
                    {
                        int lo = row_ptr_acc[f2+r], hi = row_ptr_acc[f2+r+1]-1;
                        while (lo <= hi) { int mid=(lo+hi)/2;
                            if (col_idx_acc[mid]==f2+c) { ar_t(csr_vals_acc[mid]).fetch_add(br[r*3+c]); break; }
                            if (col_idx_acc[mid]<f2+c) lo=mid+1; else hi=mid-1; }
                    }
                }
                ar_t(xi_acc[f1+r]).fetch_add(xiu[r]);
                ar_t(xi_acc[f2+r]).fetch_add(xib[r]);
            }
        });
    }).wait();

    /* ------------------------------------------------------------------
     * D.  Build block-Jacobi diag_inv on device (float).
     *     Each work item processes one 3x3 diagonal block.
     * ------------------------------------------------------------------ */
    int _n_blocks = dim / 3;
    int _nnz      = nnz;
    int _dim      = dim;

    q.submit([&](sycl::handler &h) {
        auto rptr = buf_row_ptr->get_access<sycl::access::mode::read>(h);
        auto cidx = buf_col_idx->get_access<sycl::access::mode::read>(h);
        auto vals = buf_csr_vals->get_access<sycl::access::mode::read>(h);
        auto dinv = buf_diag_inv->get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>{(size_t)_n_blocks},
            [=](sycl::id<1> blk_id)
#ifdef FPGA_HARDWARE
            [[intel::scheduler_target_fmax_mhz(400)]]
#endif
        {
            int blk  = (int)blk_id[0];
            int base = blk * 3;
            float m[9] = {0.0f};
            for (int r = 0; r < 3; r++) {
                for (int s = rptr[base+r]; s < rptr[base+r+1]; s++) {
                    int c = cidx[s];
                    if (c >= base && c < base+3)
                        m[r*3 + (c-base)] = vals[s];
                }
            }
            float det = m[0]*(m[4]*m[8]-m[5]*m[7])
                      - m[1]*(m[3]*m[8]-m[5]*m[6])
                      + m[2]*(m[3]*m[7]-m[4]*m[6]);
            if (det == 0.0f) det = 1e-30f;
            float id = 1.0f / det;
            int bo = blk * 9;
            dinv[bo+0] =  (m[4]*m[8]-m[5]*m[7])*id;
            dinv[bo+1] = -(m[1]*m[8]-m[2]*m[7])*id;
            dinv[bo+2] =  (m[1]*m[5]-m[2]*m[4])*id;
            dinv[bo+3] = -(m[3]*m[8]-m[5]*m[6])*id;
            dinv[bo+4] =  (m[0]*m[8]-m[2]*m[6])*id;
            dinv[bo+5] = -(m[0]*m[5]-m[2]*m[3])*id;
            dinv[bo+6] =  (m[3]*m[7]-m[4]*m[6])*id;
            dinv[bo+7] = -(m[0]*m[7]-m[1]*m[6])*id;
            dinv[bo+8] =  (m[0]*m[4]-m[1]*m[3])*id;
        });
    }).wait();
}

/* =========================================================================
 * DeviceGraph::pcg_solve  —  fully on-device PCG (float CSR, float vectors)
 * delta_xs_out is a CPU pointer, double[dim]; result is cast on download.
 * ========================================================================= */
void DeviceGraph::pcg_solve(sycl::queue &q, int max_iter, double *delta_xs_out)
{
    int _dim      = dim;
    int _n_blocks = dim / 3;

    /* --- Reset x = 0, r = xi -------------------------------------------- */
    q.submit([&](sycl::handler &h) {
        auto xo  = buf_pcg_x->get_access<sycl::access::mode::write>(h);
        auto ro  = buf_pcg_r->get_access<sycl::access::mode::write>(h);
        auto xi  = buf_xi->get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>{(size_t)_dim}, [=](sycl::id<1> i) {
            xo[i] = 0.0f;
            ro[i] = xi[i];
        });
    }).wait();

    /* --- Precondition: z = diag_inv * r (block-3x3) --------------------- */
    q.submit([&](sycl::handler &h) {
        auto zo   = buf_pcg_z->get_access<sycl::access::mode::write>(h);
        auto ri   = buf_pcg_r->get_access<sycl::access::mode::read>(h);
        auto dinv = buf_diag_inv->get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>{(size_t)_n_blocks}, [=](sycl::id<1> bid) {
            int blk  = (int)bid[0];
            int base = blk * 3;
            int bo   = blk * 9;
            for (int r = 0; r < 3; r++) {
                float s = 0.0f;
                for (int c = 0; c < 3; c++)
                    s += dinv[bo + r*3+c] * ri[base+c];
                zo[base+r] = s;
            }
        });
    }).wait();

    /* --- p = z ----------------------------------------------------------- */
    q.submit([&](sycl::handler &h) {
        auto po = buf_pcg_p->get_access<sycl::access::mode::write>(h);
        auto zi = buf_pcg_z->get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>{(size_t)_dim}, [=](sycl::id<1> i) {
            po[i] = zi[i];
        });
    }).wait();

    /* --- rz_old = dot(r, z) ---------------------------------------------- */
    float rz_old = 0.0f;
    {
        sycl::buffer<float,1> buf_scalar(&rz_old, sycl::range<1>{1});
        q.submit([&](sycl::handler &h) {
            auto ri = buf_pcg_r->get_access<sycl::access::mode::read>(h);
            auto zi = buf_pcg_z->get_access<sycl::access::mode::read>(h);
            auto red = sycl::reduction(buf_scalar, h, sycl::plus<float>{});
            h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                [=](sycl::id<1> i, auto &acc) { acc += ri[i] * zi[i]; });
        }).wait();
    }

    /* --- b_norm = norm(xi) for convergence tol --------------------------- */
    float b_norm_sq = 0.0f;
    {
        sycl::buffer<float,1> buf_bn(&b_norm_sq, sycl::range<1>{1});
        q.submit([&](sycl::handler &h) {
            auto xi = buf_xi->get_access<sycl::access::mode::read>(h);
            auto red = sycl::reduction(buf_bn, h, sycl::plus<float>{});
            h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                [=](sycl::id<1> i, auto &acc) { acc += xi[i]*xi[i]; });
        }).wait();
    }
    float tol = 1e-6f * std::sqrt(b_norm_sq);

    /* --- PCG iterations -------------------------------------------------- */
    for (int iter = 0; iter < max_iter; iter++) {
        /* SpMV: q = Omega * p */
        {
            auto _nnz_local = nnz;
            q.submit([&](sycl::handler &h) {
                auto qo   = buf_pcg_q->get_access<sycl::access::mode::write>(h);
                auto pi   = buf_pcg_p->get_access<sycl::access::mode::read>(h);
                auto vals = buf_csr_vals->get_access<sycl::access::mode::read>(h);
                auto rptr = buf_row_ptr->get_access<sycl::access::mode::read>(h);
                auto cidx = buf_col_idx->get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>{(size_t)_dim},
                    [=](sycl::id<1> row_id)
#ifdef FPGA_HARDWARE
                    [[intel::scheduler_target_fmax_mhz(400)]]
#endif
                {
                    int row = (int)row_id[0];
                    float s = 0.0f;
                    for (int j = rptr[row]; j < rptr[row+1]; j++)
                        s += vals[j] * pi[cidx[j]];
                    qo[row] = s;
                });
            }).wait();
        }

        /* pq = dot(p, q) */
        float pq = 0.0f;
        {
            sycl::buffer<float,1> buf_pq(&pq, sycl::range<1>{1});
            q.submit([&](sycl::handler &h) {
                auto pi = buf_pcg_p->get_access<sycl::access::mode::read>(h);
                auto qi = buf_pcg_q->get_access<sycl::access::mode::read>(h);
                auto red = sycl::reduction(buf_pq, h, sycl::plus<float>{});
                h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                    [=](sycl::id<1> i, auto &acc) { acc += pi[i]*qi[i]; });
            }).wait();
        }
        if (pq == 0.0f) break;
        float alpha = rz_old / pq;

        /* x += alpha*p; r -= alpha*q; r_norm_sq */
        float r_norm_sq = 0.0f;
        {
            sycl::buffer<float,1> buf_rn(&r_norm_sq, sycl::range<1>{1});
            q.submit([&](sycl::handler &h) {
                auto xo = buf_pcg_x->get_access<sycl::access::mode::read_write>(h);
                auto ro = buf_pcg_r->get_access<sycl::access::mode::read_write>(h);
                auto pi = buf_pcg_p->get_access<sycl::access::mode::read>(h);
                auto qi = buf_pcg_q->get_access<sycl::access::mode::read>(h);
                auto red = sycl::reduction(buf_rn, h, sycl::plus<float>{});
                h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                    [=](sycl::id<1> i, auto &acc) {
                        xo[i] += alpha * pi[i];
                        ro[i] -= alpha * qi[i];
                        acc   += ro[i] * ro[i];
                    });
            }).wait();
        }
        if (std::sqrt(r_norm_sq) < tol) break;

        /* Precond: z = diag_inv * r */
        q.submit([&](sycl::handler &h) {
            auto zo   = buf_pcg_z->get_access<sycl::access::mode::write>(h);
            auto ri   = buf_pcg_r->get_access<sycl::access::mode::read>(h);
            auto dinv = buf_diag_inv->get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>{(size_t)_n_blocks}, [=](sycl::id<1> bid) {
                int blk  = (int)bid[0];
                int base = blk * 3;
                int bo   = blk * 9;
                for (int r = 0; r < 3; r++) {
                    float s = 0.0f;
                    for (int c = 0; c < 3; c++)
                        s += dinv[bo + r*3+c] * ri[base+c];
                    zo[base+r] = s;
                }
            });
        }).wait();

        /* rz_new = dot(r, z) */
        float rz_new = 0.0f;
        {
            sycl::buffer<float,1> buf_rzn(&rz_new, sycl::range<1>{1});
            q.submit([&](sycl::handler &h) {
                auto ri = buf_pcg_r->get_access<sycl::access::mode::read>(h);
                auto zi = buf_pcg_z->get_access<sycl::access::mode::read>(h);
                auto red = sycl::reduction(buf_rzn, h, sycl::plus<float>{});
                h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                    [=](sycl::id<1> i, auto &acc) { acc += ri[i] * zi[i]; });
            }).wait();
        }
        float beta = rz_new / rz_old;

        /* p = z + beta*p */
        q.submit([&](sycl::handler &h) {
            auto po = buf_pcg_p->get_access<sycl::access::mode::read_write>(h);
            auto zi = buf_pcg_z->get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>{(size_t)_dim}, [=](sycl::id<1> i) {
                po[i] = zi[i] + beta * po[i];
            });
        }).wait();

        rz_old = rz_new;
    }

    /* --- Download x to CPU (cast float->double) -------------------------- */
    {
        auto h_x = buf_pcg_x->get_host_access(sycl::read_only);
        for (int i = 0; i < _dim; i++)
            delta_xs_out[i] = (double)h_x[i];
    }
}

/* =========================================================================
 * DeviceGraph::free_all
 * ========================================================================= */
void DeviceGraph::free_all()
{
    delete buf_sorted_obs;
    delete buf_edge_lm;
    delete buf_edge_j;
    delete buf_edge_k;
    delete buf_lm_pos;
    delete buf_lm_count;
    delete buf_z_range;
    delete buf_z_bearing;
    delete buf_z_step;
    delete buf_row_ptr;
    delete buf_col_idx;
    delete buf_hat_x;
    delete buf_csr_vals;
    delete buf_xi;
    delete buf_diag_inv;
    delete buf_pcg_x;
    delete buf_pcg_r;
    delete buf_pcg_z;
    delete buf_pcg_p;
    delete buf_pcg_q;

    free(lm_pos);
    free(lm_count);
    free(edge_lm);
    free(edge_j);
    free(edge_k);
    free(sorted_obs);
    free(z_range);
    free(z_bearing);
    free(z_step);
    free(row_ptr);
    free(col_idx);
    free(motion_csr_vals_f);
    free(motion_xi_f);
    free(csr_vals_host);
    free(xi_host);
}
