/*
 * device_graph.cpp  —  Device-resident edge computation and CSR assembly
 *                      for fpga_pcg_fp64 (SYCL/oneAPI, FP64)
 *
 * Architecture:
 *   init()   — called once before the main loop. Builds the fixed CSR
 *               structure (row_ptr, col_idx) from all (t1,t2) pairs seen in
 *               the observation data, pre-sorts observations by landmark,
 *               and allocates persistent SYCL buffers for everything that
 *               never changes across rounds.
 *
 *   update() — called once per round. Computes MotionEdge contributions on
 *               the CPU (N-1 edges, cheap), uploads them together with the
 *               current hat_xs to the device, then launches a fused SYCL
 *               parallel_for over n_obs_edges work items that computes every
 *               ObsEdge inline and atomically scatters it into d_csr_vals /
 *               d_xi.  After the kernel the values are retrieved back to the
 *               host so that the existing PCG loop (which takes raw C
 *               pointers) can proceed unchanged.
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
 * col_idx for each row is stored in sorted ascending order.
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
    return -1;  /* should never happen if structure was built correctly */
}

/* =========================================================================
 * DeviceGraph::init
 *
 * One-time setup.  Mirrors the landmark-sorting logic from make_edges()
 * and build_csr_from_edges() but produces the flat arrays needed by the
 * device kernel instead of the Edge struct array.
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
     * ------------------------------------------------------------------ */
    unsigned int max_id = 0;
    for (int i = 0; i < n_obs; i++)
        if (zlist[i].landmark_id > max_id)
            max_id = zlist[i].landmark_id;

    unsigned int n_ids = max_id + 1;

    /* id_count[lm] = how many observations see landmark lm */
    unsigned int *id_count = (unsigned int *)calloc(n_ids, sizeof(unsigned int));
    for (int i = 0; i < n_obs; i++)
        id_count[zlist[i].landmark_id]++;

    /* Build a compacted list of landmark IDs that appear in the data,
     * preserving the same ordering as make_edges() uses (id_list).
     * We replicate the exact bucket-sort algorithm from make_edges so
     * that the obs ordering in sorted_obs matches landmark_keys_zlist. */
    unsigned int *id_list = (unsigned int *)malloc(n_ids * sizeof(unsigned int));
    unsigned int *temp_ids = (unsigned int *)malloc(n_obs * sizeof(unsigned int));
    for (unsigned int i = 0; i < n_ids; i++) id_list[i] = n_ids; /* sentinel */
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

    /* Count active landmarks (those with >= 1 obs) and obs-edge pairs */
    n_lm = 0;
    n_obs_edges = 0;
    for (unsigned int i = 0; i < n_ids; i++) {
        if (id_list[i] == n_ids) continue;
        unsigned int cnt = id_count[id_list[i]];
        n_lm++;
        if (cnt >= 2)
            n_obs_edges += (int)(cnt * (cnt - 1) / 2);
    }

    /* lm_pos[lm_idx]   = start of this landmark's obs in sorted_obs[]
     * lm_count[lm_idx] = number of obs for this landmark
     * active_ids[lm_idx] = actual landmark_id for entry i */
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

    /* sorted_obs[i] = index into zlist[], sorted by landmark */
    sorted_obs = (int *)malloc(n_obs * sizeof(int));
    {
        /* For each active landmark (in id_list order) collect zlist indices */
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
     * 2.  Build flat observation arrays and edge (j,k) index arrays.
     * ------------------------------------------------------------------ */
    z_range   = (double *)malloc(n_obs * sizeof(double));
    z_bearing = (double *)malloc(n_obs * sizeof(double));
    z_step    = (int    *)malloc(n_obs * sizeof(int));
    for (int i = 0; i < n_obs; i++) {
        z_range[i]   = zlist[i].z[0];
        z_bearing[i] = zlist[i].z[1];
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
     * 3.  Build CSR structure (row_ptr, col_idx) from all edge (t1, t2).
     *     Include MotionEdge pairs (i, i+1) for i in [0, n_poses-2].
     *     Also add the anchor: diagonal entries [0,0],[1,1],[2,2].
     *     col_idx per row is sorted ascending.
     *
     * Sparse approach: collect (row,col) pairs, sort, deduplicate.
     * Avoids the O(dim^2) dense marker which OOMs for large dim.
     * ------------------------------------------------------------------ */

    /* Upper bound on number of (row,col) pairs before dedup:
     *   anchor: 3
     *   obs edges: n_obs_edges * 4 blocks * 9 entries = n_obs_edges * 36
     *   motion edges: (n_poses-1) * 4 * 9
     * Use size_t arithmetic to avoid overflow. */
    size_t max_pairs = 3
        + (size_t)n_obs_edges * 36
        + (size_t)(n_poses > 1 ? n_poses - 1 : 0) * 36;

    int *pair_row = (int *)malloc(max_pairs * sizeof(int));
    int *pair_col = (int *)malloc(max_pairs * sizeof(int));
    size_t np = 0;

    /* Anchor */
    for (int d = 0; d < 3; d++) { pair_row[np] = d; pair_col[np] = d; np++; }

    /* ObsEdge contributions */
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

    /* MotionEdge contributions */
    for (int i = 0; i < n_poses - 1; i++) {
        int f1 = i * 3, f2 = (i + 1) * 3;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
            pair_row[np]=f1+r; pair_col[np]=f1+c; np++;
            pair_row[np]=f1+r; pair_col[np]=f2+c; np++;
            pair_row[np]=f2+r; pair_col[np]=f1+c; np++;
            pair_row[np]=f2+r; pair_col[np]=f2+c; np++;
        }
    }

    /* Sort pairs by (row, col) using an index array + qsort comparator.
     * Encode as 64-bit key = row * (dim+1) + col for simple comparison. */
    long long *keys = (long long *)malloc(np * sizeof(long long));
    for (size_t i = 0; i < np; i++)
        keys[i] = (long long)pair_row[i] * (dim + 1) + pair_col[i];
    free(pair_row);
    free(pair_col);

    /* Simple sort: use stdlib qsort on keys */
    int cmp_ll(const void *a, const void *b);
    qsort(keys, np, sizeof(long long),
          [](const void *a, const void *b) -> int {
              long long x = *(const long long *)a;
              long long y = *(const long long *)b;
              return (x > y) - (x < y);
          });

    /* Deduplicate and build col_idx, row_ptr */
    row_ptr = (int *)calloc(dim + 2, sizeof(int));

    /* Pass 1: count unique entries per row */
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

    /* Prefix sum to get row_ptr */
    for (int i = 0; i < dim; i++)
        row_ptr[i + 1] += row_ptr[i];

    /* Pass 2: fill col_idx */
    col_idx = (int *)malloc(nnz * sizeof(int));
    int *fill = (int *)calloc(dim, sizeof(int));  /* per-row fill pointer */
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
     * 4.  Allocate host arrays for per-round MotionEdge CSR scatter
     *     and retrieve buffers.
     * ------------------------------------------------------------------ */
    motion_csr_vals = (double *)calloc(nnz,  sizeof(double));
    motion_xi       = (double *)calloc(dim,   sizeof(double));

    /* Host buffers for csr_vals and xi that will be populated each round */
    csr_vals_host = (double *)calloc(nnz, sizeof(double));
    xi_host       = (double *)calloc(dim, sizeof(double));

    /* ------------------------------------------------------------------
     * 5.  Allocate persistent SYCL buffers for fixed data and upload.
     * ------------------------------------------------------------------ */
    buf_sorted_obs = new sycl::buffer<int,1>(sorted_obs, sycl::range<1>{(size_t)n_obs});
    buf_edge_lm    = new sycl::buffer<int,1>(edge_lm,    sycl::range<1>{(size_t)n_obs_edges});
    buf_edge_j     = new sycl::buffer<int,1>(edge_j,     sycl::range<1>{(size_t)n_obs_edges});
    buf_edge_k     = new sycl::buffer<int,1>(edge_k,     sycl::range<1>{(size_t)n_obs_edges});
    buf_lm_pos     = new sycl::buffer<int,1>(lm_pos,     sycl::range<1>{(size_t)n_lm});
    buf_lm_count   = new sycl::buffer<int,1>(lm_count,   sycl::range<1>{(size_t)n_lm});
    buf_z_range    = new sycl::buffer<double,1>(z_range,   sycl::range<1>{(size_t)n_obs});
    buf_z_bearing  = new sycl::buffer<double,1>(z_bearing, sycl::range<1>{(size_t)n_obs});
    buf_z_step     = new sycl::buffer<int,1>(z_step,     sycl::range<1>{(size_t)n_obs});
    buf_row_ptr    = new sycl::buffer<int,1>(row_ptr,    sycl::range<1>{(size_t)(dim+1)});
    buf_col_idx    = new sycl::buffer<int,1>(col_idx,    sycl::range<1>{(size_t)nnz});

    /* Per-round buffers (allocated once, overwritten each round).
     * hat_x: [3*n_poses] doubles. */
    buf_hat_x    = new sycl::buffer<double,1>(sycl::range<1>{(size_t)(3 * n_poses)});
    buf_csr_vals = new sycl::buffer<double,1>(sycl::range<1>{(size_t)nnz});
    buf_xi       = new sycl::buffer<double,1>(sycl::range<1>{(size_t)dim});

    /* Persistent PCG working buffers */
    int n_blocks = (int)dim / 3;
    buf_diag_inv = new sycl::buffer<double,1>(sycl::range<1>{(size_t)(n_blocks * 9)});
    buf_pcg_x    = new sycl::buffer<double,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_r    = new sycl::buffer<double,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_z    = new sycl::buffer<double,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_p    = new sycl::buffer<double,1>(sycl::range<1>{(size_t)dim});
    buf_pcg_q    = new sycl::buffer<double,1>(sycl::range<1>{(size_t)dim});

    free(active_ids);
    free(id_count);
    free(id_list);
}

/* =========================================================================
 * DeviceGraph::update
 *
 * Per-round pipeline:
 *   A. Upload hat_xs to device.
 *   B. Compute MotionEdge CSR scatter on CPU, upload as initial d_csr_vals
 *      and d_xi (also includes anchor constraint).
 *   C. Launch fused ObsEdge kernel (atomic scatter).
 *   D. Retrieve d_csr_vals and d_xi back to host via host accessors.
 * ========================================================================= */
void DeviceGraph::update(struct HAT_X *hat_xs,
                         struct U *us, double delta, double lambda,
                         const double *mns,
                         sycl::queue &q)
{
    /* ------------------------------------------------------------------
     * A.  Flatten hat_xs into a contiguous [3*n_poses] array and
     *     upload via the per-round buffer.
     * ------------------------------------------------------------------ */
    {
        /* Write directly through a host accessor so we don't need a
         * separate staging allocation. */
        auto h_hx = buf_hat_x->get_host_access();
        for (int i = 0; i < n_poses; i++) {
            h_hx[i*3+0] = hat_xs[i].hat_x[0];
            h_hx[i*3+1] = hat_xs[i].hat_x[1];
            h_hx[i*3+2] = hat_xs[i].hat_x[2];
        }
    }

    /* ------------------------------------------------------------------
     * B.  MotionEdge CSR scatter on CPU.
     *
     * For each consecutive pair (i, i+1) we compute the full MotionEdge
     * math (same as MotionEdge_create) and scatter into motion_csr_vals /
     * motion_xi.  Then we add the anchor and upload both as the initial
     * values of d_csr_vals and d_xi.
     * ------------------------------------------------------------------ */
    memset(motion_csr_vals, 0, (size_t)nnz * sizeof(double));
    memset(motion_xi,       0, (size_t)dim * sizeof(double));

    /* Anchor: +1e6 on diagonal entries 0,1,2 */
    for (int d = 0; d < 3; d++) {
        int slot = find_slot_host(row_ptr, col_idx, d, d);
        if (slot >= 0) motion_csr_vals[slot] += 1000000.0;
    }

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

        /* A [3x2] */
        double A[6];
        A[0] = stw/omega - st/omega;
        A[1] = nu/omega*delta*ctw - nu/omega/omega*stw + nu/omega/omega*st;
        A[2] = ct/omega - ctw/omega;
        A[3] = nu/omega*delta*stw - nu/omega/omega*ct + nu/omega/omega*ctw;
        A[4] = 0.0;
        A[5] = delta;

        /* F [3x3] = I + partial derivatives */
        double F[9] = {1,0,0, 0,1,0, 0,0,1};
        F[0*3+2] = nu/omega*std::cos(t0+omega*delta) - nu/omega*std::cos(t0);
        F[1*3+2] = nu/omega*std::sin(t0+omega*delta) - nu/omega*std::sin(t0);

        /* Sigma_motion = A * M * A^T + 0.0001*I   [3x3] */
        /* A*M: [3x2] * diag(M00,M11) */
        double AM[6];
        for (int r = 0; r < 3; r++) {
            AM[r*2+0] = A[r*2+0] * M00;
            AM[r*2+1] = A[r*2+1] * M11;
        }
        /* AM * A^T [3x3] */
        double AMAT[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                double s = 0.0;
                for (int k2 = 0; k2 < 2; k2++)
                    s += AM[r*2+k2] * A[c*2+k2];
                AMAT[r*3+c] = s;
            }
        /* + 0.0001 * I */
        for (int d = 0; d < 3; d++) AMAT[d*3+d] += 0.0001;

        /* Omega_m = inv(AMAT)  [3x3] — 3x3 analytical inverse */
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

        /* F^T * Om [3x3] */
        double FtOm[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                double s = 0.0;
                for (int k2 = 0; k2 < 3; k2++)
                    s += F[k2*3+r] * Om[k2*3+c];
                FtOm[r*3+c] = s;
            }

        /* omega_upperleft  = lambda * F^T*Om*F [3x3] */
        /* omega_upperright = -lambda * F^T*Om  [3x3] */
        /* omega_bottomleft = -lambda * Om*F    [3x3] */
        /* omega_bottomright= lambda * Om       [3x3] */
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

        /* hat_e for MotionEdge: hat_xs[t2] - state_transition(hat_xs[t1]) */
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

        /* xi contributions: F^T*Om*he and -Om*he, scaled by lambda */
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

        /* Scatter into motion_csr_vals */
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                int sl;
                sl = find_slot_host(row_ptr, col_idx, f1+r, f1+c);
                if (sl >= 0) motion_csr_vals[sl] += lambda * FtOmF[r*3+c];
                sl = find_slot_host(row_ptr, col_idx, f1+r, f2+c);
                if (sl >= 0) motion_csr_vals[sl] += -lambda * FtOm[r*3+c];
                sl = find_slot_host(row_ptr, col_idx, f2+r, f1+c);
                if (sl >= 0) motion_csr_vals[sl] += -lambda * OmF[r*3+c];
                sl = find_slot_host(row_ptr, col_idx, f2+r, f2+c);
                if (sl >= 0) motion_csr_vals[sl] += lambda * Om[r*3+c];
            }
            motion_xi[f1+r] +=  lambda * FtOmhe[r];
            motion_xi[f2+r] += -lambda * Omhe[r];
        }
    }

    /* Upload motion_csr_vals and motion_xi as initial values for device
     * buffers via host accessors. */
    {
        auto h_cv = buf_csr_vals->get_host_access();
        for (int i = 0; i < nnz; i++) h_cv[i] = motion_csr_vals[i];
    }
    {
        auto h_xi = buf_xi->get_host_access();
        for (int i = 0; i < dim; i++) h_xi[i] = motion_xi[i];
    }

    /* ------------------------------------------------------------------
     * C.  Launch fused ObsEdge kernel.
     *     Each work item handles one (j,k) edge pair for one landmark.
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

            double r1 = z_range_acc[idx_j],   b1 = z_bearing_acc[idx_j];
            double r2 = z_range_acc[idx_k],   b2 = z_bearing_acc[idx_k];

            double theta1 = hat_x_acc[step1*3+2];
            double theta2 = hat_x_acc[step2*3+2];

            double s1 = sycl::sin(theta1+b1), c1 = sycl::cos(theta1+b1);
            double s2 = sycl::sin(theta2+b2), c2 = sycl::cos(theta2+b2);

            double ex = hat_x_acc[step2*3+0] - hat_x_acc[step1*3+0] + r2*c2 - r1*c1;
            double ey = hat_x_acc[step2*3+1] - hat_x_acc[step1*3+1] + r2*s2 - r1*s1;

            const double snr0 = 0.14, snr1 = 0.05;
            double q1r = r1*snr0, q1b = snr1, q2r = r2*snr0, q2b = snr1;

            /* Sigma = R1*Q1*R1^T + R2*Q2*R2^T  (2x2 symmetric) */
            double a00 = c1*c1*(q1r*q1r) + r1*r1*s1*s1*(q1b*q1b);
            double a01 = c1*s1*(q1r*q1r) - r1*r1*s1*c1*(q1b*q1b);
            double a11 = s1*s1*(q1r*q1r) + r1*r1*c1*c1*(q1b*q1b);
            double b00 = c2*c2*(q2r*q2r) + r2*r2*s2*s2*(q2b*q2b);
            double b01 = c2*s2*(q2r*q2r) - r2*r2*s2*c2*(q2b*q2b);
            double b11 = s2*s2*(q2r*q2r) + r2*r2*c2*c2*(q2b*q2b);

            double sig00 = a00+b00, sig01 = a01+b01, sig11 = a11+b11;
            double det = sig00*sig11 - sig01*sig01;
            if (det == 0.0) det = 1e-30;
            double inv_d = 1.0/det;
            double om00 = sig11*inv_d, om01 = -sig01*inv_d, om11 = sig00*inv_d;

            /* B1 [2x3], B2 [2x3] */
            /* OB1 = Om * B1  [2x3],  OB2 = Om * B2  [2x3]
             * stored as flat 6-element arrays, row-major */
            double OB1[6], OB2[6];
            /* B1 = [[-1, 0, r1*s1],[-0,-1,-r1*c1]] */
            OB1[0] = om00*(-1) + om01*(-0);  /* Om*B1[0,0] */
            OB1[1] = om00*( 0) + om01*(-1);  /* Om*B1[0,1] */
            OB1[2] = om00*(r1*s1) + om01*(-r1*c1); /* Om*B1[0,2] */
            OB1[3] = om01*(-1) + om11*(-0);
            OB1[4] = om01*( 0) + om11*(-1);
            OB1[5] = om01*(r1*s1) + om11*(-r1*c1);
            /* B2 = [[1,0,-r2*s2],[0,1,r2*c2]] */
            OB2[0] = om00*( 1) + om01*( 0);
            OB2[1] = om00*( 0) + om01*( 1);
            OB2[2] = om00*(-r2*s2) + om01*(r2*c2);
            OB2[3] = om01*( 1) + om11*( 0);
            OB2[4] = om01*( 0) + om11*( 1);
            OB2[5] = om01*(-r2*s2) + om11*(r2*c2);

            /* tB1 = B1^T [3x2] row-major:
             *  [[-1, 0],
             *   [ 0,-1],
             *   [r1*s1, -r1*c1]]
             * tB2 = B2^T [3x2]:
             *  [[1, 0],
             *   [0, 1],
             *   [-r2*s2, r2*c2]]
             */
            double tB1[6], tB2[6];
            tB1[0]=-1;    tB1[1]=0;
            tB1[2]=0;     tB1[3]=-1;
            tB1[4]=r1*s1; tB1[5]=-r1*c1;
            tB2[0]=1;     tB2[1]=0;
            tB2[2]=0;     tB2[3]=1;
            tB2[4]=-r2*s2;tB2[5]=r2*c2;

            /* 4 blocks [3x3]: tB1*OB1, tB1*OB2, tB2*OB1, tB2*OB2 */
            double ul[9], ur[9], bl[9], br[9];
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

            /* xi contributions: -tB1*Om*e, -tB2*Om*e */
            double Oe0 = om00*ex + om01*ey;
            double Oe1 = om01*ex + om11*ey;
            double xiu[3], xib[3];
            xiu[0] = -(tB1[0]*Oe0+tB1[1]*Oe1);
            xiu[1] = -(tB1[2]*Oe0+tB1[3]*Oe1);
            xiu[2] = -(tB1[4]*Oe0+tB1[5]*Oe1);
            xib[0] = -(tB2[0]*Oe0+tB2[1]*Oe1);
            xib[1] = -(tB2[2]*Oe0+tB2[3]*Oe1);
            xib[2] = -(tB2[4]*Oe0+tB2[5]*Oe1);

            /* Binary search for CSR slot (inline, no nested lambda) */
            int f1 = step1*3, f2 = step2*3;

            using ar_t = sycl::atomic_ref<double,
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
     * D.  Build block-Jacobi preconditioner (diag_inv) on device.
     *     Each work item handles one 3x3 diagonal block.
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
            double M[9] = {0,0,0, 0,0,0, 0,0,0};

            for (int r = 0; r < 3; r++) {
                int row = base + r;
                for (int k = rptr[row]; k < rptr[row + 1]; k++) {
                    int c = cidx[k] - base;
                    if (c >= 0 && c < 3) M[r * 3 + c] = vals[k];
                }
            }

            double det = M[0]*(M[4]*M[8]-M[5]*M[7])
                       - M[1]*(M[3]*M[8]-M[5]*M[6])
                       + M[2]*(M[3]*M[7]-M[4]*M[6]);
            if (det == 0.0) det = 1e-30;
            double inv_det = 1.0 / det;

            double *out = &dinv[blk * 9];
            out[0] = (M[4]*M[8]-M[5]*M[7]) * inv_det;
            out[1] = (M[2]*M[7]-M[1]*M[8]) * inv_det;
            out[2] = (M[1]*M[5]-M[2]*M[4]) * inv_det;
            out[3] = (M[5]*M[6]-M[3]*M[8]) * inv_det;
            out[4] = (M[0]*M[8]-M[2]*M[6]) * inv_det;
            out[5] = (M[2]*M[3]-M[0]*M[5]) * inv_det;
            out[6] = (M[3]*M[7]-M[4]*M[6]) * inv_det;
            out[7] = (M[1]*M[6]-M[0]*M[7]) * inv_det;
            out[8] = (M[0]*M[4]-M[1]*M[3]) * inv_det;
        });
    }).wait();

    (void)_nnz; (void)_dim;
}

/* =========================================================================
 * DeviceGraph::pcg_solve
 *
 * Fully device-resident PCG using buf_csr_vals, buf_xi, buf_diag_inv
 * from the last update() call.  All vector operations run as SYCL
 * parallel_for + reduction kernels.  One host-accessor read at the end
 * to download delta_xs.
 * ========================================================================= */
void DeviceGraph::pcg_solve(sycl::queue &q, int max_iter, double *delta_xs_out)
{
    int _dim      = dim;
    int _n_blocks = dim / 3;

    /* --- Reset x = 0, r = xi --- */
    q.submit([&](sycl::handler &h) {
        auto x = buf_pcg_x->get_access<sycl::access::mode::write>(h);
        auto r = buf_pcg_r->get_access<sycl::access::mode::write>(h);
        auto xi_acc = buf_xi->get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>{(size_t)_dim}, [=](sycl::id<1> i) {
            x[i] = 0.0;
            r[i] = xi_acc[i];
        });
    }).wait();

    /* --- z = diag_inv * r (precond apply) --- */
    q.submit([&](sycl::handler &h) {
        auto z    = buf_pcg_z->get_access<sycl::access::mode::write>(h);
        auto r    = buf_pcg_r->get_access<sycl::access::mode::read>(h);
        auto dinv = buf_diag_inv->get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>{(size_t)_n_blocks}, [=](sycl::id<1> blk_id) {
            int blk = (int)blk_id[0], base = blk * 3;
            const auto *inv = &dinv[blk * 9];
            for (int row = 0; row < 3; row++) {
                double s = 0.0;
                for (int col = 0; col < 3; col++) s += inv[row*3+col] * r[base+col];
                z[base+row] = s;
            }
        });
    }).wait();

    /* --- p = z --- */
    q.submit([&](sycl::handler &h) {
        auto p = buf_pcg_p->get_access<sycl::access::mode::write>(h);
        auto z = buf_pcg_z->get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>{(size_t)_dim}, [=](sycl::id<1> i) { p[i] = z[i]; });
    }).wait();

    /* --- rz_old = dot(r, z) --- */
    double rz_old = 0.0;
    {
        sycl::buffer<double,1> rz_buf(&rz_old, sycl::range<1>{1});
        q.submit([&](sycl::handler &h) {
            auto r = buf_pcg_r->get_access<sycl::access::mode::read>(h);
            auto z = buf_pcg_z->get_access<sycl::access::mode::read>(h);
            auto red = sycl::reduction(rz_buf, h, sycl::plus<double>());
            h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                [=](sycl::id<1> i, auto &acc) { acc += r[i] * z[i]; });
        }).wait();
        auto h_rz = rz_buf.get_host_access();
        rz_old = h_rz[0];
    }

    /* --- b_norm for tolerance --- */
    double b_norm_sq = 0.0;
    {
        sycl::buffer<double,1> bn_buf(&b_norm_sq, sycl::range<1>{1});
        q.submit([&](sycl::handler &h) {
            auto xi_acc = buf_xi->get_access<sycl::access::mode::read>(h);
            auto red = sycl::reduction(bn_buf, h, sycl::plus<double>());
            h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                [=](sycl::id<1> i, auto &acc) { acc += xi_acc[i] * xi_acc[i]; });
        }).wait();
        auto h_bn = bn_buf.get_host_access();
        b_norm_sq = h_bn[0];
    }
    double tol = 1e-6 * std::sqrt(b_norm_sq);

    /* --- PCG iteration --- */
    for (int k = 0; k < max_iter; k++) {
        /* q_vec = Omega * p  (SpMV) */
        {
            int d = _dim, n = nnz;
            q.submit([&](sycl::handler &h) {
                auto q_acc    = buf_pcg_q->get_access<sycl::access::mode::write>(h);
                auto p_acc    = buf_pcg_p->get_access<sycl::access::mode::read>(h);
                auto vals_acc = buf_csr_vals->get_access<sycl::access::mode::read>(h);
                auto rptr_acc = buf_row_ptr->get_access<sycl::access::mode::read>(h);
                auto cidx_acc = buf_col_idx->get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>{(size_t)d},
                    [=](sycl::id<1> i)
#ifdef FPGA_HARDWARE
                    [[intel::scheduler_target_fmax_mhz(400)]]
#endif
                {
                    double sum = 0.0;
                    for (int j = rptr_acc[i]; j < rptr_acc[(int)i+1]; j++)
                        sum += vals_acc[j] * p_acc[cidx_acc[j]];
                    q_acc[i] = sum;
                });
            }).wait();
        }

        /* pq = dot(p, q) */
        double pq = 0.0;
        {
            sycl::buffer<double,1> pq_buf(&pq, sycl::range<1>{1});
            q.submit([&](sycl::handler &h) {
                auto p_acc = buf_pcg_p->get_access<sycl::access::mode::read>(h);
                auto q_acc = buf_pcg_q->get_access<sycl::access::mode::read>(h);
                auto red = sycl::reduction(pq_buf, h, sycl::plus<double>());
                h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                    [=](sycl::id<1> i, auto &acc) { acc += p_acc[i] * q_acc[i]; });
            }).wait();
            auto h_pq = pq_buf.get_host_access();
            pq = h_pq[0];
        }
        if (pq == 0.0) break;
        double alpha = rz_old / pq;

        /* x += alpha*p,  r -= alpha*q */
        double r_norm_sq = 0.0;
        {
            sycl::buffer<double,1> rn_buf(&r_norm_sq, sycl::range<1>{1});
            q.submit([&](sycl::handler &h) {
                auto x_acc = buf_pcg_x->get_access<sycl::access::mode::read_write>(h);
                auto r_acc = buf_pcg_r->get_access<sycl::access::mode::read_write>(h);
                auto p_acc = buf_pcg_p->get_access<sycl::access::mode::read>(h);
                auto q_acc = buf_pcg_q->get_access<sycl::access::mode::read>(h);
                auto red = sycl::reduction(rn_buf, h, sycl::plus<double>());
                double _alpha = alpha;
                h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                    [=](sycl::id<1> i, auto &acc) {
                        x_acc[i] += _alpha * p_acc[i];
                        r_acc[i] -= _alpha * q_acc[i];
                        acc += r_acc[i] * r_acc[i];
                    });
            }).wait();
            auto h_rn = rn_buf.get_host_access();
            r_norm_sq = h_rn[0];
        }
        if (std::sqrt(r_norm_sq) < tol) break;

        /* z = diag_inv * r */
        q.submit([&](sycl::handler &h) {
            auto z    = buf_pcg_z->get_access<sycl::access::mode::write>(h);
            auto r    = buf_pcg_r->get_access<sycl::access::mode::read>(h);
            auto dinv = buf_diag_inv->get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>{(size_t)_n_blocks}, [=](sycl::id<1> blk_id) {
                int blk = (int)blk_id[0], base = blk * 3;
                const auto *inv = &dinv[blk * 9];
                for (int row = 0; row < 3; row++) {
                    double s = 0.0;
                    for (int col = 0; col < 3; col++) s += inv[row*3+col] * r[base+col];
                    z[base+row] = s;
                }
            });
        }).wait();

        /* rz_new = dot(r, z) */
        double rz_new = 0.0;
        {
            sycl::buffer<double,1> rz2_buf(&rz_new, sycl::range<1>{1});
            q.submit([&](sycl::handler &h) {
                auto r = buf_pcg_r->get_access<sycl::access::mode::read>(h);
                auto z = buf_pcg_z->get_access<sycl::access::mode::read>(h);
                auto red = sycl::reduction(rz2_buf, h, sycl::plus<double>());
                h.parallel_for(sycl::range<1>{(size_t)_dim}, red,
                    [=](sycl::id<1> i, auto &acc) { acc += r[i] * z[i]; });
            }).wait();
            auto h_rz2 = rz2_buf.get_host_access();
            rz_new = h_rz2[0];
        }
        double beta = rz_new / rz_old;

        /* p = z + beta*p */
        q.submit([&](sycl::handler &h) {
            auto p = buf_pcg_p->get_access<sycl::access::mode::read_write>(h);
            auto z = buf_pcg_z->get_access<sycl::access::mode::read>(h);
            double _beta = beta;
            h.parallel_for(sycl::range<1>{(size_t)_dim},
                [=](sycl::id<1> i) { p[i] = z[i] + _beta * p[i]; });
        }).wait();

        rz_old = rz_new;
    }

    /* --- Download result --- */
    {
        auto h_x = buf_pcg_x->get_host_access(sycl::read_only);
        for (int i = 0; i < _dim; i++) delta_xs_out[i] = h_x[i];
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
    free(motion_csr_vals);
    free(motion_xi);
    free(csr_vals_host);
    free(xi_host);
}
