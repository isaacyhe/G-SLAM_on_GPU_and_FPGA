/**
 * @file GSLAM.cu  (gpu_pcg_fp32)
 * @brief CUDA-accelerated G-SLAM — fully device-resident pipeline (FP32 CSR)
 *
 * Per-round pipeline (no CPU computation after init):
 *  1. Upload hat_xs (~240 KB float, stored as double) to device.
 *  2. Zero d_csr_vals (float) and d_xi (double) on device.
 *  3. compute_motion_edges_kernel_f: anchor + N-1 MotionEdges (float CSR, double xi).
 *  4. compute_and_scatter_edges_kernel_f: ObsEdges.
 *  5. build_diag_inv_f + PCG solve (float) fully on device.
 *  6. Download float delta_xs, cast to double, update poses.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "GSLAM.h"
#include "MatrixOps.h"
#include "MapEdge.h"
#include "device_graph.cuh"

#include "Python.h"

int main( int argc, char *argv[] ) {
    if( argc == 6 ) {
        printf("Diff: %s ", argv[1]);
        printf("Lambda: %s ", argv[2]);
        printf("Rounds: %s ", argv[3]);
        printf("Drawing: %s ", argv[4]);
        printf("Input file: %s\n", argv[5]);
    }
    else {
        printf("Usage: ");
        printf("%s Diff_0|1 Lambda Number_of_rounds Drawing_0|1|2|3 path/to/input/file\n", argv[0]);
        exit( 1 );
    }

    unsigned int toUseDiff      = (unsigned int)atoi(argv[1]);
    float        lambda         = (float)atof(argv[2]);
    unsigned int numberOfRounds = (unsigned int)atoi(argv[3]);
    unsigned int toDraw         = (unsigned int)atoi(argv[4]);

    float        *delta          = (float *)calloc ( 1, sizeof( float ) );
    unsigned int *size_of_us     = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );
    unsigned int *size_of_hat_xs = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );
    unsigned int *size_of_zlist  = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );

    struct U     *us     = (struct U     *)malloc ( sizeof( struct U     ) * 131072 );
    struct HAT_X *hat_xs = (struct HAT_X *)malloc ( sizeof( struct HAT_X ) * 131072 );
    struct Z     *zlist  = (struct Z     *)malloc ( sizeof( struct Z     ) * 524288 );

    unsigned int dim = read_data( argv[5], delta, us, size_of_us,
                                  hat_xs, size_of_hat_xs, zlist, size_of_zlist ) * 3;

    printf( "delta: %f ", *delta );
    printf( "size_of_us: %d ", *size_of_us );
    printf( "size_of_hat_xs: %d ", *size_of_hat_xs );
    printf( "size_of_zlist: %d ", *size_of_zlist );
    printf( "dim: %d\n", dim );

    if ( toDraw >= 1 )
        Py_Initialize();

    if ( toDraw >= 3 ) {
        PyObject *SYS = PyImport_ImportModule( "sys" );
        PyObject *PATH = PyObject_GetAttrString( SYS, "path" );
        PyList_Append( PATH, PyUnicode_FromString( "." ) );
        PyObject *NAME   = PyUnicode_FromString( "GSLAM_draw" );
        PyObject *MODULE = PyImport_Import( NAME );
        PyObject *DICT   = PyModule_GetDict( MODULE );
        PyObject *FUNC   = PyDict_GetItemString( DICT, "draw" );
        PyObject *ARGS   = PyTuple_New( 7 );
        PyObject *HAT_Xs_STEPs       = PyList_New( *size_of_hat_xs );
        PyObject *HAT_Xs             = PyList_New( *size_of_hat_xs * 3 );
        PyObject *ZLIST_STEPs        = PyList_New( *size_of_zlist );
        PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
        PyObject *Zs                 = PyList_New( *size_of_zlist * 2 );
        PyObject *MSes_INDEXes       = PyList_New( 0 );
        PyObject *MSes               = PyList_New( 0 );
        for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
            PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[i].step ) );
            PyList_SetItem( HAT_Xs, i*3,   PyFloat_FromDouble( hat_xs[i].hat_x[0] ) );
            PyList_SetItem( HAT_Xs, i*3+1, PyFloat_FromDouble( hat_xs[i].hat_x[1] ) );
            PyList_SetItem( HAT_Xs, i*3+2, PyFloat_FromDouble( hat_xs[i].hat_x[2] ) );
        }
        PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs ); PyTuple_SetItem( ARGS, 1, HAT_Xs );
        for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
            PyList_SetItem( ZLIST_STEPs,        i, PyLong_FromUnsignedLong( zlist[i].step ) );
            PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[i].landmark_id ) );
            PyList_SetItem( Zs, i*2,   PyFloat_FromDouble( zlist[i].z[0] ) );
            PyList_SetItem( Zs, i*2+1, PyFloat_FromDouble( zlist[i].z[1] ) );
        }
        PyTuple_SetItem( ARGS, 2, ZLIST_STEPs ); PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
        PyTuple_SetItem( ARGS, 4, Zs );
        PyTuple_SetItem( ARGS, 5, MSes_INDEXes ); PyTuple_SetItem( ARGS, 6, MSes );
        PyObject_CallObject( FUNC, ARGS );
    }

    /* ===================================================================
     * ONE-TIME SETUP
     * =================================================================== */
    double mns[4] = { 0.19, 0.001, 0.13, 0.2 };
    int nnz = 0;
    device_graph_init( zlist, *size_of_zlist,
                       us, *size_of_hat_xs, dim,
                       (double)*delta, (double)lambda, mns,
                       &nnz );

    /* Device buffers: float CSR values, double xi */
    float  *d_csr_vals = NULL;
    double *d_xi       = NULL;
    cudaMalloc(&d_csr_vals, (size_t)nnz * sizeof(float));
    cudaMalloc(&d_xi,       (size_t)dim * sizeof(double));

    float last_cost = 0.0f, cost = 0.0f, diff = 0.0f;
    float best_diff = 1e30f;
    unsigned int no_improve_count = 0;

    /* ===================================================================
     * MAIN OPTIMIZATION LOOP
     * =================================================================== */
    for ( unsigned int n = 1; n <= numberOfRounds; n ++ ) {

        /* -----------------------------------------------------------
         * STEP 1+2: Upload hat_xs, launch MotionEdge + ObsEdge kernels.
         * ----------------------------------------------------------- */
        device_graph_update( hat_xs, *size_of_hat_xs,
                             d_csr_vals, d_xi );

        /* -----------------------------------------------------------
         * STEP 3+4: Build diag_inv on device + full PCG on device.
         * ----------------------------------------------------------- */
        float *delta_xs_f = (float *)calloc( dim, sizeof(float) );
        device_pcg_solve( delta_xs_f, d_csr_vals, d_xi, (int)dim, nnz, 500 );

        /* Cast to double for pose update (hat_xs stores double) */
        double *delta_xs = (double *)malloc( dim * sizeof(double) );
        for (unsigned int j = 0; j < dim; j++) delta_xs[j] = (double)delta_xs_f[j];

        /* Norm for convergence (float), normalised by sqrt(dim) for scale-independence */
        float delta_xs_f_norm = vec_norm( delta_xs_f, dim, 1 ) / sqrtf( (float)dim );

        free( delta_xs_f );

        /* -----------------------------------------------------------
         * STEP 5: Update poses
         * ----------------------------------------------------------- */
        for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
            hat_xs[i].hat_x[0] += delta_xs[i*3  ];
            hat_xs[i].hat_x[1] += delta_xs[i*3+1];
            hat_xs[i].hat_x[2] += delta_xs[i*3+2];
        }

        /* -----------------------------------------------------------
         * STEP 6: Convergence
         * ----------------------------------------------------------- */
        if ( toUseDiff )
            diff = delta_xs_f_norm;
        else
            diff = delta_xs_f_norm;  /* cost mode not supported; use norm */

        printf( "%d rounds executed: %.17g\n", n, (double)diff );

        free( delta_xs );

        if ( diff < 0.01f ) {
            if ( toDraw >= 2 ) {
                PyObject *SYS = PyImport_ImportModule( "sys" );
                PyObject *PATH = PyObject_GetAttrString( SYS, "path" );
                PyList_Append( PATH, PyUnicode_FromString( "." ) );
                PyObject *NAME   = PyUnicode_FromString( "GSLAM_draw" );
                PyObject *MODULE = PyImport_Import( NAME );
                PyObject *DICT   = PyModule_GetDict( MODULE );
                PyObject *FUNC   = PyDict_GetItemString( DICT, "draw" );
                PyObject *ARGS   = PyTuple_New( 7 );
                PyObject *HAT_Xs_STEPs       = PyList_New( *size_of_hat_xs );
                PyObject *HAT_Xs             = PyList_New( *size_of_hat_xs * 3 );
                PyObject *ZLIST_STEPs        = PyList_New( *size_of_zlist );
                PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
                PyObject *Zs                 = PyList_New( *size_of_zlist * 2 );
                PyObject *MSes_INDEXes       = PyList_New( 0 );
                PyObject *MSes               = PyList_New( 0 );
                for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
                    PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[i].step ) );
                    PyList_SetItem( HAT_Xs, i*3,   PyFloat_FromDouble( hat_xs[i].hat_x[0] ) );
                    PyList_SetItem( HAT_Xs, i*3+1, PyFloat_FromDouble( hat_xs[i].hat_x[1] ) );
                    PyList_SetItem( HAT_Xs, i*3+2, PyFloat_FromDouble( hat_xs[i].hat_x[2] ) );
                }
                PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs ); PyTuple_SetItem( ARGS, 1, HAT_Xs );
                for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
                    PyList_SetItem( ZLIST_STEPs,        i, PyLong_FromUnsignedLong( zlist[i].step ) );
                    PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[i].landmark_id ) );
                    PyList_SetItem( Zs, i*2,   PyFloat_FromDouble( zlist[i].z[0] ) );
                    PyList_SetItem( Zs, i*2+1, PyFloat_FromDouble( zlist[i].z[1] ) );
                }
                PyTuple_SetItem( ARGS, 2, ZLIST_STEPs ); PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
                PyTuple_SetItem( ARGS, 4, Zs );
                PyTuple_SetItem( ARGS, 5, MSes_INDEXes ); PyTuple_SetItem( ARGS, 6, MSes );
                PyObject_CallObject( FUNC, ARGS );
            }
            break;
        }

        /* No-improvement plateau stop */
        if ( diff < best_diff ) {
            best_diff = diff;
            no_improve_count = 0;
        } else {
            if ( ++no_improve_count >= 20 ) break;
        }

        (void)last_cost; (void)cost;
    }

    /* ===================================================================
     * POST-LOOP: landmark positions + final draw
     * =================================================================== */
    struct LandmarkKeysZList *landmark_keys_zlist =
        (struct LandmarkKeysZList *)malloc( sizeof( struct LandmarkKeysZList ) * 4194304 );
    struct Edge *edges_final =
        (struct Edge *)malloc( sizeof( struct Edge ) * 4194304 );
    unsigned int *size_of_edges_final = (unsigned int *)calloc( 1, sizeof(unsigned int) );
    unsigned int *size_of_lkz         = (unsigned int *)calloc( 1, sizeof(unsigned int) );

    make_edges( hat_xs, *size_of_hat_xs, zlist, *size_of_zlist,
                edges_final, size_of_edges_final,
                landmark_keys_zlist, size_of_lkz );
    free( edges_final );
    free( size_of_edges_final );

    unsigned int number_of_landmarks  = 0;
    unsigned int landmark_id_last     = *size_of_lkz;
    unsigned int *order_of_landmark_ids = (unsigned int *)calloc( 131072, sizeof(unsigned int) );
    unsigned int *count_of_landmark_ids = (unsigned int *)calloc( 131072, sizeof(unsigned int) );
    struct MapEdge *map_edges           = (struct MapEdge *)malloc( sizeof(struct MapEdge) * 1048576 );
    unsigned int    count_of_map_edges  = 0;
    float          *ms   = (float *)malloc( sizeof(float) * 2 * 131072 );
    float           snr[2] = { 0.14f, 0.05f };
    struct Z        head_z = { 0 };

    for ( unsigned int i = 0; i < *size_of_lkz; i ++ ) {
        if ( landmark_id_last != landmark_keys_zlist[i].landmark_id ) {
            landmark_id_last = landmark_keys_zlist[i].landmark_id;
            order_of_landmark_ids[number_of_landmarks] = landmark_id_last;
            number_of_landmarks++;
            head_z            = landmark_keys_zlist[i].z;
            count_of_map_edges = 0;
        }
        map_edges[count_of_map_edges] =
            *MapEdge_create( landmark_keys_zlist[i].z, head_z, hat_xs, snr );
        count_of_map_edges++;
        count_of_landmark_ids[landmark_id_last]++;

        if ( i == *size_of_lkz - 1 ||
             ( i < *size_of_lkz - 1 &&
               landmark_keys_zlist[i].landmark_id !=
               landmark_keys_zlist[i+1].landmark_id ) )
        {
            float *mean = (float *)calloc( 2, sizeof(float) );
            float *m    = (float *)calloc( 2 * count_of_map_edges, sizeof(float) );
            for ( unsigned int k = 0; k < count_of_map_edges; k++ ) {
                m[2*k]   = map_edges[k].m[0];
                m[2*k+1] = map_edges[k].m[1];
            }
            vec_mean( mean, m, 2, count_of_map_edges );
            free( m );
            ms[2*(number_of_landmarks-1)]   = mean[0];
            ms[2*(number_of_landmarks-1)+1] = mean[1];
            free( mean );
        }
    }

    if ( toDraw >= 1 ) {
        PyObject *SYS = PyImport_ImportModule( "sys" );
        PyObject *PATH = PyObject_GetAttrString( SYS, "path" );
        PyList_Append( PATH, PyUnicode_FromString( "." ) );
        PyObject *NAME   = PyUnicode_FromString( "GSLAM_draw" );
        PyObject *MODULE = PyImport_Import( NAME );
        PyObject *DICT   = PyModule_GetDict( MODULE );
        PyObject *FUNC   = PyDict_GetItemString( DICT, "draw" );
        PyObject *ARGS   = PyTuple_New( 7 );
        PyObject *HAT_Xs_STEPs       = PyList_New( *size_of_hat_xs );
        PyObject *HAT_Xs             = PyList_New( *size_of_hat_xs * 3 );
        PyObject *ZLIST_STEPs        = PyList_New( *size_of_zlist );
        PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
        PyObject *Zs                 = PyList_New( *size_of_zlist * 2 );
        PyObject *MSes_INDEXes       = PyList_New( number_of_landmarks );
        PyObject *MSes               = PyList_New( number_of_landmarks * 2 );
        for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
            PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[i].step ) );
            PyList_SetItem( HAT_Xs, i*3,   PyFloat_FromDouble( hat_xs[i].hat_x[0] ) );
            PyList_SetItem( HAT_Xs, i*3+1, PyFloat_FromDouble( hat_xs[i].hat_x[1] ) );
            PyList_SetItem( HAT_Xs, i*3+2, PyFloat_FromDouble( hat_xs[i].hat_x[2] ) );
        }
        PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs ); PyTuple_SetItem( ARGS, 1, HAT_Xs );
        for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
            PyList_SetItem( ZLIST_STEPs,        i, PyLong_FromUnsignedLong( zlist[i].step ) );
            PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[i].landmark_id ) );
            PyList_SetItem( Zs, i*2,   PyFloat_FromDouble( zlist[i].z[0] ) );
            PyList_SetItem( Zs, i*2+1, PyFloat_FromDouble( zlist[i].z[1] ) );
        }
        PyTuple_SetItem( ARGS, 2, ZLIST_STEPs ); PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
        PyTuple_SetItem( ARGS, 4, Zs );
        for ( unsigned int i = 0; i < number_of_landmarks; i ++ ) {
            PyList_SetItem( MSes_INDEXes, i, PyLong_FromUnsignedLong( order_of_landmark_ids[i] ) );
            PyList_SetItem( MSes, i*2,   PyFloat_FromDouble( (double)ms[i*2]   ) );
            PyList_SetItem( MSes, i*2+1, PyFloat_FromDouble( (double)ms[i*2+1] ) );
        }
        PyTuple_SetItem( ARGS, 5, MSes_INDEXes ); PyTuple_SetItem( ARGS, 6, MSes );
        PyObject_CallObject( FUNC, ARGS );
    }

    /* ===================================================================
     * Cleanup
     * =================================================================== */
    device_graph_free();
    cudaFree( d_csr_vals );
    cudaFree( d_xi );

    free( map_edges );
    free( ms );
    free( order_of_landmark_ids );
    free( count_of_landmark_ids );
    free( landmark_keys_zlist );
    free( size_of_lkz );

    if ( toDraw >= 1 )
        Py_Finalize();

    free( delta );
    free( us );
    free( hat_xs );
    free( zlist );
    free( size_of_us );
    free( size_of_hat_xs );
    free( size_of_zlist );

    return 0;
}

/* =========================================================================
 * make_edges  (kept for post-loop landmark estimation)
 * ========================================================================= */
void make_edges( struct HAT_X *hat_xs, unsigned int size_of_hat_xs,
                 struct Z *zlist, unsigned int size_of_zlist,
                 struct Edge *edges, unsigned int *size_of_edges,
                 struct LandmarkKeysZList *landmark_keys_zlist,
                 unsigned int *size_of_landmark_keys_zlist )
{
    unsigned int max_id = 0;
    for ( unsigned int i = 0; i < size_of_zlist; i ++ )
        if ( zlist[i].landmark_id > max_id ) max_id = zlist[i].landmark_id;

    unsigned int *id_count          = (unsigned int *)calloc( max_id+1, sizeof(unsigned int) );
    unsigned int *temp_landmark_ids = (unsigned int *)malloc( sizeof(unsigned int)*size_of_zlist );

    for ( unsigned int i = 0; i < size_of_zlist; i ++ ) {
        temp_landmark_ids[i] = zlist[i].landmark_id;
        id_count[temp_landmark_ids[i]]++;
    }

    unsigned int *id_list = (unsigned int *)malloc( sizeof(unsigned int)*(max_id+1) );
    for ( unsigned int i = 0; i < max_id+1; i ++ ) id_list[i] = max_id+1;

    for ( unsigned int i = 0; i < max_id+1; i ++ ) {
        for ( unsigned int j = 0; j < size_of_zlist; j ++ ) {
            if ( id_list[i]==max_id+1 && id_list[i]!=temp_landmark_ids[j] ) {
                id_list[i] = temp_landmark_ids[j]; temp_landmark_ids[j] = max_id+1;
            } else if ( id_list[i]!=max_id+1 && id_list[i]==temp_landmark_ids[j] ) {
                temp_landmark_ids[j] = max_id+1;
            }
        }
    }

    *size_of_landmark_keys_zlist = size_of_zlist;
    unsigned int count = 0;
    for ( unsigned int i = 0; i < max_id+1; i ++ )
        for ( unsigned int j = 0; j < size_of_zlist; j ++ )
            if ( zlist[j].landmark_id == id_list[i] ) {
                landmark_keys_zlist[count].landmark_id = zlist[j].landmark_id;
                landmark_keys_zlist[count].z           = zlist[j];
                count++;
            }

    unsigned int sum = 0;
    for ( unsigned int i = 0; i < max_id+1; i ++ )
        if ( id_list[i]!=max_id+1 && id_count[id_list[i]]>=2 )
            sum += combination( id_count[id_list[i]], 2 );
    *size_of_edges = sum;

    free( id_count ); free( temp_landmark_ids ); free( id_list );
}

void add_edge( struct Edge edge, float *Omega, unsigned int Omega_y, unsigned int Omega_x, float *xi ) {
    unsigned int f1 = edge.t1 * 3, f2 = edge.t2 * 3;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            Omega[(f1+r)*Omega_x+(f1+c)] += edge.omega_upperleft[r*3+c];
            Omega[(f1+r)*Omega_x+(f2+c)] += edge.omega_upperright[r*3+c];
            Omega[(f2+r)*Omega_x+(f1+c)] += edge.omega_bottomleft[r*3+c];
            Omega[(f2+r)*Omega_x+(f2+c)] += edge.omega_bottomright[r*3+c];
        }
        xi[f1+r] += edge.xi_upper[r];
        xi[f2+r] += edge.xi_bottom[r];
    }
}

void state_transition( float *result, float nu, float omega, unsigned int time, struct HAT_X pose ) {
    float t0 = (float)pose.hat_x[2];
    if ( fabsf(omega) < 1e-10f ) {
        float v[3] = { nu*cosf(t0), nu*sinf(t0), omega };
        float m[3] = { 0.0f };
        mul( m, v, 3, 1, (float)time );
        float a[3] = { 0.0f };
        float ph[3] = { (float)pose.hat_x[0], (float)pose.hat_x[1], (float)pose.hat_x[2] };
        add( a, ph, 3, 1, m, 3, 1 );
        for (int i=0;i<3;i++) result[i]=a[i];
    } else {
        float v[3] = {
            nu/omega*sinf(t0+omega*time)-nu/omega*sinf(t0),
            nu/omega*cosf(t0)-nu/omega*cosf(t0+omega*time),
            omega*(float)time
        };
        float a[3] = { 0.0f };
        float ph[3] = { (float)pose.hat_x[0], (float)pose.hat_x[1], (float)pose.hat_x[2] };
        add( a, ph, 3, 1, v, 3, 1 );
        for (int i=0;i<3;i++) result[i]=a[i];
    }
}

unsigned int combination( unsigned int n, unsigned int r ) {
    unsigned int p=n, f=r;
    for ( unsigned int i=1;i<r;i++ ) { p=p*(n-i); f=f*(r-i); }
    return p/f;
}

unsigned int read_data( char *file_name, float *delta,
                        struct U *us,    unsigned int *size_of_us,
                        struct HAT_X *hat_xs, unsigned int *size_of_hat_xs,
                        struct Z *zlist, unsigned int *size_of_zlist )
{
    char  line[1024];
    FILE *input_file;
    if ( (input_file=fopen(file_name,"r"))==NULL ) { printf("Error! Cannot open file!"); exit(1); }
    unsigned int step_index=0, num_of_us=0, num_of_xs=0, num_of_zs=0;
    while ( fgets(line,sizeof(line),input_file)!=0 ) {
        char *token = strtok(line," ");
        if ( strcmp(token,"u")==0 ) {
            token=strtok(NULL," "); step_index=atoi(token); us[num_of_us].step=step_index;
            token=strtok(NULL," "); us[num_of_us].nu   =atof(token);
            token=strtok(NULL," "); us[num_of_us].omega=atof(token);
            num_of_us++;
        } else if ( strcmp(token,"x")==0 ) {
            token=strtok(NULL," "); step_index=atoi(token); hat_xs[num_of_xs].step=step_index;
            token=strtok(NULL," "); hat_xs[num_of_xs].hat_x[0]=atof(token);
            token=strtok(NULL," "); hat_xs[num_of_xs].hat_x[1]=atof(token);
            token=strtok(NULL," "); hat_xs[num_of_xs].hat_x[2]=atof(token);
            num_of_xs++;
        } else if ( strcmp(token,"z")==0 ) {
            token=strtok(NULL," "); step_index=atoi(token); zlist[num_of_zs].step=step_index;
            token=strtok(NULL," "); zlist[num_of_zs].landmark_id=atoi(token);
            token=strtok(NULL," "); zlist[num_of_zs].z[0]=atof(token);
            token=strtok(NULL," "); zlist[num_of_zs].z[1]=atof(token);
            num_of_zs++;
        } else {
            token=strtok(NULL," "); *delta=(float)atof(token);
        }
    }
    fclose(input_file);
    *size_of_us=num_of_us; *size_of_hat_xs=num_of_xs; *size_of_zlist=num_of_zs;
    return num_of_us;
}
