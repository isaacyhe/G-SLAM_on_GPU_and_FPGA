/**
 * @file GSLAM.cu
 * @brief CUDA-accelerated Graph-based Simultaneous Localization and Mapping
 *
 * This is the main implementation of G-SLAM with GPU acceleration for matrix inversion.
 *
 * G-SLAM Algorithm Overview:
 * --------------------------
 * Graph-based SLAM formulates robot localization as a graph optimization problem:
 * - Nodes: Robot poses (x, y, theta) at different timesteps
 * - Edges: Constraints from sensor measurements (observations, odometry)
 * - Goal: Minimize error by optimizing all poses simultaneously
 *
 * The algorithm iteratively:
 * 1. Constructs a graph with edges representing measurement constraints
 * 2. Builds the Information Matrix (Omega) and coefficient vector (xi)
 * 3. Solves: delta_xs = Omega^-1 * xi  ← BOTTLENECK: Matrix inversion!
 * 4. Updates robot poses: hat_xs += delta_xs
 * 5. Repeats until convergence (delta_xs becomes negligible)
 *
 * GPU Acceleration:
 * -----------------
 * This implementation offloads ONLY the matrix inversion to GPU (Step 3).
 * Profiling showed matrix inversion consumes ~95% of execution time.
 *
 * Key optimization: Gauss-Jordan elimination parallelized on CUDA
 * - Sequential CPU: O(n³) operations, single threaded
 * - GPU implementation: O(n) sequential iterations, each with O(n²) parallel ops
 * - Result: Up to 19.7x speedup on NVIDIA Titan V
 *
 * Memory Management:
 * ------------------
 * - Information Matrix (Omega): Allocated on CPU, copied to GPU for inversion
 * - Inverse matrix: Computed on GPU, copied back to CPU for use
 * - Other operations (edge creation, updates): Remain on CPU
 *
 * @see inv.cu for GPU matrix inversion implementation
 * @see MatrixOps.c for CPU baseline operations
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "GSLAM.h"
#include "MatrixOps.h"
#include "MotionEdge.h"
#include "ObsEdge.h"
#include "MapEdge.h"

#include "Python.h"

/**
 * @brief Main entry point for G-SLAM with GPU acceleration
 *
 * Usage: ./GSLAM <use_diff> <lambda> <num_rounds> <drawing_mode> <input_file>
 *
 * Parameters:
 * - use_diff: 0 or 1 (convergence metric: 0=cost difference, 1=norm of delta_xs)
 * - lambda: Regularization weight for motion edges (e.g., 0.1)
 * - num_rounds: Maximum number of optimization iterations
 * - drawing_mode: 0-4 (visualization options, requires Python)
 * - input_file: Path to dataset file (e.g., data/cityTrees800.txt)
 */
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
    
    unsigned int toUseDiff = (unsigned int)atoi(argv[1]);
    double lambda = (double)atof(argv[2]);
    unsigned int numberOfRounds = (unsigned int)atoi(argv[3]);
    unsigned int toDraw = (unsigned int)atoi(argv[4]);

    /* hat_xs, zlist, us, delta = read_data()
    dim = len(hat_xs)*3 */

    double *delta = (double *)calloc ( 1, sizeof( double ) );

    unsigned int *size_of_us = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );
    unsigned int *size_of_hat_xs = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );
    unsigned int *size_of_zlist = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );

    struct U *us = (struct U *)malloc ( sizeof( struct U ) * 131072 );
    struct HAT_X *hat_xs = (struct HAT_X *)malloc ( sizeof( struct HAT_X ) * 131072 );
    struct Z *zlist = (struct Z *)malloc ( sizeof( struct Z ) * 524288 );

    unsigned int dim = read_data( argv[5], delta, us, size_of_us, hat_xs, size_of_hat_xs, zlist, size_of_zlist ) * 3;

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

        PyObject *NAME = PyUnicode_FromString( "GSLAM_draw" );
        PyObject *MODULE = PyImport_Import( NAME );
        PyObject *DICT = PyModule_GetDict( MODULE );
        PyObject *FUNC = PyDict_GetItemString( DICT, "draw" );
        PyObject *ARGS = PyTuple_New( 7 );

        PyObject *HAT_Xs_STEPs = PyList_New( *size_of_hat_xs );
        PyObject *HAT_Xs = PyList_New( *size_of_hat_xs * 3 );

        PyObject *ZLIST_STEPs = PyList_New( *size_of_zlist );
        PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
        PyObject *Zs = PyList_New( *size_of_zlist * 2 );

        PyObject *MSes_INDEXes = PyList_New( 0 );
        PyObject *MSes = PyList_New( 0 );

        for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
            PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[ i ].step ) );

            PyList_SetItem( HAT_Xs, i * 3, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 0 ] ) );
            PyList_SetItem( HAT_Xs, i * 3 + 1, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 1 ] ) );
            PyList_SetItem( HAT_Xs, i * 3 + 2, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 2 ] ) );
        }

        PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs );
        PyTuple_SetItem( ARGS, 1, HAT_Xs );

        for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
            PyList_SetItem( ZLIST_STEPs, i, PyLong_FromUnsignedLong( zlist[ i ].step ) );

            PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[ i ].landmark_id ) );

            PyList_SetItem( Zs, i * 2, PyFloat_FromDouble( zlist[ i ].z[ 0 ] ) );
            PyList_SetItem( Zs, i * 2 + 1, PyFloat_FromDouble( zlist[ i ].z[ 1 ] ) );
        }

        PyTuple_SetItem( ARGS, 2, ZLIST_STEPs );
        PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
        PyTuple_SetItem( ARGS, 4, Zs );

        PyTuple_SetItem( ARGS, 5, MSes_INDEXes );
        PyTuple_SetItem( ARGS, 6, MSes );

        PyObject_CallObject( FUNC, ARGS );
    }

    /* for n in range(1, 10000):
        ##エッジ、大きな精度行列、係数ベクトルの作成##
        edges, _ = make_edges(hat_xs, zlist)  #返す変数が2つになるので「_」で合わせる

        for i in range(len(hat_xs)-1): #行動エッジの追加
            edges.append(MotionEdge(i, i+1, hat_xs, us, delta, 10.0)) #lambda=100

        draw(hat_xs, zlist, edges)

        Omega = np.zeros((dim, dim))
        xi = np.zeros(dim)
        Omega[0:3, 0:3] += np.eye(3)*1000000

        ##軌跡を動かす量（差分）の計算##
        for e in edges:
            add_edge(e, Omega, xi)

        delta_xs = np.linalg.inv(Omega).dot(xi)

        ##推定値の更新##
        for i in range(len(hat_xs)):
            hat_xs[i] += delta_xs[i*3:(i+1)*3]

        ##終了判定##
        diff = np.linalg.norm(delta_xs)
        print("{}回目の繰り返し: {}".format(n, diff))
        if diff < 0.01:
            draw(hat_xs, zlist, edges)
            break */

    double last_cost = 0.0, cost = 0.0, diff = 0.0;
    double best_diff_even = 1e30, best_diff_odd = 1e30;
    unsigned int no_improve_even = 0, no_improve_odd = 0;
    //printf("before the for loop");

    /**
     * MAIN OPTIMIZATION LOOP
     * ======================
     * Each iteration refines the robot trajectory by:
     * 1. Creating constraint edges from sensor data
     * 2. Building the Information Matrix (Omega) and coefficient vector (xi)
     * 3. Solving the linear system: Omega * delta_xs = xi  (via matrix inversion)
     * 4. Updating robot poses with the computed corrections (delta_xs)
     * 5. Checking for convergence
     *
     * The loop terminates when:
     * - Maximum rounds reached, OR
     * - Convergence criterion satisfied (diff < 0.01)
     */
    for ( unsigned int n = 1; n <= numberOfRounds; n ++ ) {
        //printf( "n: %d\n", n );
	    //printf("for loop");

        // STEP 1: Create observation edges
        // These edges connect robot poses to landmark observations
        struct Edge *edges = (struct Edge *)malloc( sizeof( struct Edge ) * 4194304 );
        struct LandmarkKeysZList *landmark_keys_zlist = (struct LandmarkKeysZList *)malloc( sizeof( struct LandmarkKeysZList ) * 4194304 );

        unsigned int *size_of_edges = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );
        unsigned int *size_of_landmark_keys_zlist = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );

        // Create edges from landmark observations (ObsEdge and MapEdge)
        make_edges( hat_xs, *size_of_hat_xs, zlist, *size_of_zlist, edges, size_of_edges, landmark_keys_zlist, size_of_landmark_keys_zlist );

        free( landmark_keys_zlist );
        free( size_of_landmark_keys_zlist );
        
        //printf( "size_of_hat_xs: %d\n", *size_of_hat_xs );
        //printf( "size_of_edges: %d\n", *size_of_edges );

        // STEP 2: Add motion edges
        // These edges constrain consecutive robot poses based on odometry
        double mns[4] = { 0.19, 0.001, 0.13, 0.2 };  // Motion noise parameters

        for ( unsigned int i = 0; i < *size_of_hat_xs - 1; i ++ ) {
            edges[ *size_of_edges + i ] = *MotionEdge_create( i, i + 1, hat_xs, us, *delta, lambda, mns );
        }

        *size_of_edges = *size_of_edges + *size_of_hat_xs - 1;

        //printf( "size_of_hat_xs: %d\n", *size_of_hat_xs );
        //printf( "size_of_edges: %d\n", *size_of_edges );

        if ( !toUseDiff ) {
	    for( unsigned int i = 0; i < *size_of_edges; i ++ ) {
                double transposeOFE[ 1 * 3 ] = { 0.0 };
                tra( transposeOFE, edges[ i ].hat_e, 3, 1 );

                double dot_productOFtEOmega[ 1 * 3 ] = { 0.0 };
                dot( dot_productOFtEOmega, transposeOFE, 1, 3, edges[ i ].Omega, 3, 3 );

                double dot_productOFtEOmegaE[ 1 * 1 ] = { 0.0 };
                dot( dot_productOFtEOmegaE, dot_productOFtEOmega, 1, 3, edges[ i ].hat_e, 3, 1 );

                cost += dot_productOFtEOmegaE[ 0 ] * edges[ i ].lambda;
            }
	}

        if ( toDraw >= 4 ) {
            PyObject *SYS = PyImport_ImportModule( "sys" );
            PyObject *PATH = PyObject_GetAttrString( SYS, "path" );
            PyList_Append( PATH, PyUnicode_FromString( "." ) );

            PyObject *NAME = PyUnicode_FromString( "GSLAM_draw" );
            PyObject *MODULE = PyImport_Import( NAME );
            PyObject *DICT = PyModule_GetDict( MODULE );
            PyObject *FUNC = PyDict_GetItemString( DICT, "draw" );
            PyObject *ARGS = PyTuple_New( 7 );

            PyObject *HAT_Xs_STEPs = PyList_New( *size_of_hat_xs );
            PyObject *HAT_Xs = PyList_New( *size_of_hat_xs * 3 );

            PyObject *ZLIST_STEPs = PyList_New( *size_of_zlist );
            PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
            PyObject *Zs = PyList_New( *size_of_zlist * 2 );

            PyObject *MSes_INDEXes = PyList_New( 0 );
            PyObject *MSes = PyList_New( 0 );

            for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
                PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[ i ].step ) );

                PyList_SetItem( HAT_Xs, i * 3, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 0 ] ) );
                PyList_SetItem( HAT_Xs, i * 3 + 1, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 1 ] ) );
                PyList_SetItem( HAT_Xs, i * 3 + 2, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 2 ] ) );
            }

            PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs );
            PyTuple_SetItem( ARGS, 1, HAT_Xs );

            for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
                PyList_SetItem( ZLIST_STEPs, i, PyLong_FromUnsignedLong( zlist[ i ].step ) );

                PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[ i ].landmark_id ) );

                PyList_SetItem( Zs, i * 2, PyFloat_FromDouble( zlist[ i ].z[ 0 ] ) );
                PyList_SetItem( Zs, i * 2 + 1, PyFloat_FromDouble( zlist[ i ].z[ 1 ] ) );
            }

            PyTuple_SetItem( ARGS, 2, ZLIST_STEPs );
            PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
            PyTuple_SetItem( ARGS, 4, Zs );

            PyTuple_SetItem( ARGS, 5, MSes_INDEXes );
            PyTuple_SetItem( ARGS, 6, MSes );

            PyObject_CallObject( FUNC, ARGS );
        }

        // STEP 3: Build Information Matrix (Omega) and coefficient vector (xi)
        // The Information Matrix encodes all constraints from edges
        // Size: dim × dim where dim = 3 * number_of_poses (x, y, theta per pose)
    	double *Omega = (double *)malloc ( sizeof( double ) * dim * dim ); // dim * dim

        // Initialize Omega to zero, except anchor first pose with high confidence
        for ( unsigned int j = 0; j < dim; j ++ ) {
            for ( unsigned int k = 0; k < dim; k ++ ) {
                if ( j == k && j < 3 ) {
                    Omega[ dim * j + k ] = 1000000.0;  // Fix first pose (anchor)
                }

                else {
                    Omega[ dim * j + k ] = 0.0;
                }
            }
        }

        // Initialize coefficient vector (right-hand side of linear system)
        double *xi = (double *)malloc ( sizeof( double ) * dim ); // dim

        for ( unsigned int i = 0; i < dim; i ++ ) {
            xi[ i ] = 0.0;
        }

        // Accumulate contributions from all edges into Omega and xi
        for ( unsigned int i = 0; i < *size_of_edges; i ++ ) {
            //printf( "edges.t1: %d\n", edges[i].t1 );
            //printf( "edges.t2: %d\n", edges[i].t2 );

            add_edge( edges[ i ], Omega, dim, dim, xi );
        }

        double *delta_xs = (double *)malloc ( sizeof( double ) * dim ); // dim
        double *inverse = (double *)malloc ( sizeof( double ) * dim * dim ); // dim * dim;

        /*printf( "Omega: " );
        for ( unsigned int i = dim * dim - dim - dim; i < dim * dim - dim; i ++ ) {
        	printf( "%f ", Omega[ i ] );
        }
        printf( "\n" );*/

        /*printf( "Omega: " );
        for ( unsigned int i = dim * dim - dim; i < dim * dim; i ++ ) {
        	printf( "%f ", Omega[ i ] );
        }
        printf( "\n" );*/

	/*printf("Omega: ");
	for(unsigned int i = 0; i < dim * dim; i++)
	{
		printf("%f ", Omega[i]);
	}
	printf("\n");by jzheng*/
        // ===================================================================
        // STEP 4: GPU-ACCELERATED MATRIX INVERSION (THE CRITICAL BOTTLENECK)
        // ===================================================================
        // This is where ~95% of execution time is spent in the CPU version!
        //
        // We solve: Omega * delta_xs = xi
        // By computing: delta_xs = Omega^-1 * xi
        //
        // The matrix inversion (Omega^-1) is performed on GPU using CUDA
        // - Matrix size: dim × dim (e.g., 2400×2400 for 800 timesteps)
        // - Algorithm: Gauss-Jordan elimination (parallelized)
        // - Speedup: Up to 19.7x compared to CPU baseline
        // ===================================================================

	// Mixed precision: cast Omega to float, invert on GPU in FP32, cast result back to FP64
        float *Omega_f   = (float*)malloc( sizeof(float) * dim * dim );
        float *inverse_f = (float*)malloc( sizeof(float) * dim * dim );
        for ( unsigned int j = 0; j < dim * dim; j++ )
            Omega_f[ j ] = (float)Omega[ j ];

	float *inverse_d;
	cudaMalloc((void **)&inverse_d, sizeof(float) * dim * dim);
        inv_cuda( inverse_d, Omega_f, dim );
	cudaMemcpy(inverse_f, inverse_d, sizeof(float) * dim * dim, cudaMemcpyDeviceToHost);
	cudaFree(inverse_d);
        free( Omega_f );

        for ( unsigned int j = 0; j < dim * dim; j++ )
            inverse[ j ] = (double)inverse_f[ j ];
        free( inverse_f );

        /*printf( "inverse: " );
        for ( unsigned int i = 0; i < dim; i ++ ) {
        	printf( "%f ", inverse[ i ] );
        }
        printf( "\n" );*/

        //printf( "inverse: " );
        //for ( unsigned int i = dim; i < dim * 2; i ++ ) {
        //	printf( "%f ", inverse[ i ] );
        //}
        //printf( "\n" );

        //printf( "xi: " );
        //for ( unsigned int i = 0; i < dim; i ++ ) {
        //	printf( "%f ", xi[ i ] );
        //}
        //printf( "\n" );

        //for ( unsigned int i = 0; i < dim; i ++ ) {
        //	xi[ i ] = 1.0;
        //}

        // STEP 5: Compute pose corrections
        // delta_xs = Omega^-1 * xi (matrix-vector multiplication)
        dot( delta_xs, inverse, dim, dim, xi, dim, 1 );

        //printf( "delta_xs: " );
        /*for ( unsigned int i = 0; i < dim; i ++ ) {
        	printf( "%f\n", delta_xs[ i ] );
        }*/
        //printf( "\n" );

        // STEP 6: Update robot poses
        // Apply corrections to each pose: hat_xs += delta_xs
        for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
            hat_xs[i].hat_x[0] = hat_xs[i].hat_x[0] + delta_xs[ i * 3 ];      // x
            hat_xs[i].hat_x[1] = hat_xs[i].hat_x[1] + delta_xs[ i * 3 + 1 ];  // y
            hat_xs[i].hat_x[2] = hat_xs[i].hat_x[2] + delta_xs[ i * 3 + 2 ];  // theta
        }

        // STEP 7: Compute convergence metric
        if ( toUseDiff )
            diff = vec_norm( delta_xs, dim, 1 ) / sqrt( (double)dim );  // normalized L2 norm

	    else
	        diff = fabs( cost - last_cost ) / cost;  // Relative cost change

	    printf( "%d rounds executed: %.17g\n", n, diff );

        free( Omega );
        free( xi );

        free( delta_xs );
        free( inverse );

        free( edges );
        free( size_of_edges );

        if ( toUseDiff ) {
            if ( diff < 0.01 ) {
                if ( toDraw >= 2 ) {
                    PyObject *SYS = PyImport_ImportModule( "sys" );
                    PyObject *PATH = PyObject_GetAttrString( SYS, "path" );
                    PyList_Append( PATH, PyUnicode_FromString( "." ) );

                    PyObject *NAME = PyUnicode_FromString( "GSLAM_draw" );
                    PyObject *MODULE = PyImport_Import( NAME );
                    PyObject *DICT = PyModule_GetDict( MODULE );
                    PyObject *FUNC = PyDict_GetItemString( DICT, "draw" );
                    PyObject *ARGS = PyTuple_New( 7 );

                    PyObject *HAT_Xs_STEPs = PyList_New( *size_of_hat_xs );
                    PyObject *HAT_Xs = PyList_New( *size_of_hat_xs * 3 );

                    PyObject *ZLIST_STEPs = PyList_New( *size_of_zlist );
                    PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
                    PyObject *Zs = PyList_New( *size_of_zlist * 2 );

                    PyObject *MSes_INDEXes = PyList_New( 0 );
                    PyObject *MSes = PyList_New( 0 );

                    for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
                        PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[ i ].step ) );

                        PyList_SetItem( HAT_Xs, i * 3, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 0 ] ) );
                        PyList_SetItem( HAT_Xs, i * 3 + 1, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 1 ] ) );
                        PyList_SetItem( HAT_Xs, i * 3 + 2, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 2 ] ) );
                    }

                    PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs );
                    PyTuple_SetItem( ARGS, 1, HAT_Xs );

                    for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
                        PyList_SetItem( ZLIST_STEPs, i, PyLong_FromUnsignedLong( zlist[ i ].step ) );

                        PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[ i ].landmark_id ) );

                        PyList_SetItem( Zs, i * 2, PyFloat_FromDouble( zlist[ i ].z[ 0 ] ) );
                        PyList_SetItem( Zs, i * 2 + 1, PyFloat_FromDouble( zlist[ i ].z[ 1 ] ) );
                    }

                    PyTuple_SetItem( ARGS, 2, ZLIST_STEPs );
                    PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
                    PyTuple_SetItem( ARGS, 4, Zs );

                    PyTuple_SetItem( ARGS, 5, MSes_INDEXes );
                    PyTuple_SetItem( ARGS, 6, MSes );

                    PyObject_CallObject( FUNC, ARGS );
                }

                break;
            }
            /* Plateau stop: track even/odd rounds separately for 2-cycle */
            if ( n % 2 == 0 ) {
                if ( diff < best_diff_even ) { best_diff_even = diff; no_improve_even = 0; }
                else { if ( ++no_improve_even >= 10 ) break; }
            } else {
                if ( diff < best_diff_odd ) { best_diff_odd = diff; no_improve_odd = 0; }
                else { if ( ++no_improve_odd >= 10 ) break; }
            }
        }

	    else {
            if( diff < 1e-4 ) {
	        if ( toDraw >= 2 ) {
                    PyObject *SYS = PyImport_ImportModule( "sys" );
                    PyObject *PATH = PyObject_GetAttrString( SYS, "path" );
                    PyList_Append( PATH, PyUnicode_FromString( "." ) );

                    PyObject *NAME = PyUnicode_FromString( "GSLAM_draw" );
                    PyObject *MODULE = PyImport_Import( NAME );
                    PyObject *DICT = PyModule_GetDict( MODULE );
                    PyObject *FUNC = PyDict_GetItemString( DICT, "draw" );
                    PyObject *ARGS = PyTuple_New( 7 );

                    PyObject *HAT_Xs_STEPs = PyList_New( *size_of_hat_xs );
                    PyObject *HAT_Xs = PyList_New( *size_of_hat_xs * 3 );

                    PyObject *ZLIST_STEPs = PyList_New( *size_of_zlist );
                    PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
                    PyObject *Zs = PyList_New( *size_of_zlist * 2 );

                    PyObject *MSes_INDEXes = PyList_New( 0 );
                    PyObject *MSes = PyList_New( 0 );

                    for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
                        PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[ i ].step ) );

                        PyList_SetItem( HAT_Xs, i * 3, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 0 ] ) );
                        PyList_SetItem( HAT_Xs, i * 3 + 1, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 1 ] ) );
                        PyList_SetItem( HAT_Xs, i * 3 + 2, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 2 ] ) );
                    }

                    PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs );
                    PyTuple_SetItem( ARGS, 1, HAT_Xs );

                    for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
                        PyList_SetItem( ZLIST_STEPs, i, PyLong_FromUnsignedLong( zlist[ i ].step ) );

                        PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[ i ].landmark_id ) );

                        PyList_SetItem( Zs, i * 2, PyFloat_FromDouble( zlist[ i ].z[ 0 ] ) );
                        PyList_SetItem( Zs, i * 2 + 1, PyFloat_FromDouble( zlist[ i ].z[ 1 ] ) );
                    }

                    PyTuple_SetItem( ARGS, 2, ZLIST_STEPs );
                    PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
                    PyTuple_SetItem( ARGS, 4, Zs );

                    PyTuple_SetItem( ARGS, 5, MSes_INDEXes );
                    PyTuple_SetItem( ARGS, 6, MSes );

                    PyObject_CallObject( FUNC, ARGS );
                }

                break;
            }
	    
	        printf( "last_cost: %.17g\n", last_cost );
	        printf( "cost: %.17g\n", cost );
	    
	        last_cost = cost;
	    }
    }

    /* _, zlist_landmark = make_edges(hat_xs, zlist)

    ms = {} ###graphbasedslam_2d_sensor_mapexec

    for landmark_id in zlist_landmark:
        edges = []
        head_z = zlist_landmark[landmark_id][0]

        for z in zlist_landmark[landmark_id]:
            edges.append(MapEdge(z[0], z[1][1], head_z[0], head_z[1][1], hat_xs))

        Omega = np.zeros((2,2)) #2x2に

        xi = np.zeros(2)                 #2次元に

        for e in edges:
            Omega += e.Omega
            xi += e.xi

        ms[landmark_id] = np.mean([e.m for e in edges], axis=0)

    draw(hat_xs, zlist, edges, ms) */

    struct Edge *edges = (struct Edge *)malloc( sizeof( struct Edge ) * 4194304 );//1048576
    struct LandmarkKeysZList *landmark_keys_zlist = (struct LandmarkKeysZList *)malloc( sizeof( struct LandmarkKeysZList ) * 4194304 );

    unsigned int *size_of_edges = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );
    unsigned int *size_of_landmark_keys_zlist = (unsigned int *)calloc ( 1, sizeof( unsigned int ) );

    make_edges( hat_xs, *size_of_hat_xs, zlist, *size_of_zlist, edges, size_of_edges, landmark_keys_zlist, size_of_landmark_keys_zlist );

    free( edges );
    free( size_of_edges );

    unsigned int number_of_landmarks = 0;
    unsigned int landmark_id = *size_of_landmark_keys_zlist;

    unsigned int *order_of_landmark_ids = (unsigned int *)calloc( 1024, sizeof( unsigned int ) );
    //unsigned int *count_of_landmark_ids = calloc( 1024, sizeof( unsigned int ) );

    struct MapEdge *map_edges = (struct MapEdge *)malloc( sizeof( struct MapEdge ) * 1048576 );
    unsigned int count_of_map_edges = 0;

    double *ms = (double *)malloc( sizeof( double ) * 2 * 131072 );

    struct Z head_z = { 0 };
    double snr[2] = { 0.14, 0.05 };

    for ( unsigned int i = 0; i < *size_of_landmark_keys_zlist; i ++ ) {
        if ( landmark_id != landmark_keys_zlist[ i ].landmark_id ) {
            landmark_id = landmark_keys_zlist[ i ].landmark_id;
            order_of_landmark_ids[ number_of_landmarks ] = landmark_id;

            number_of_landmarks ++;

            head_z = landmark_keys_zlist[ i ].z;
			count_of_map_edges = 0;
		}

		map_edges[ count_of_map_edges ] = *MapEdge_create ( landmark_keys_zlist[ i ].z, head_z, hat_xs, snr );
		//printf("map_edges[ %d ]: %e %e\n", count_of_map_edges, *map_edges[ count_of_map_edges ].m, *map_edges[ count_of_map_edges ].Omega);
		count_of_map_edges ++;

		//count_of_landmark_ids[ landmark_id ] ++;

		if ( i == *size_of_landmark_keys_zlist - 1 || ( i < *size_of_landmark_keys_zlist - 1 && landmark_keys_zlist[ i ].landmark_id != landmark_keys_zlist[ i + 1 ].landmark_id ) ) {
			//printf( "count_of_map_edges: %d\n", count_of_map_edges );

			double *mean = (double *)calloc( 2, sizeof( double ) );
			double *m = (double *)calloc( 2 * count_of_map_edges, sizeof( double ) );//128

			for ( unsigned int n = 0; n < count_of_map_edges; n ++ ) {
				m[ 2 * n ] = map_edges[ n ].m[ 0 ];
				m[ 2 * n + 1 ] = map_edges[ n ].m[ 1 ];

				//printf( "m: %f %f\n", m[ 2 * n ], m[ 2 * n + 1 ] );
			}

			vec_mean( mean, m, 2, count_of_map_edges );
			free( m );

			//printf( "mean: %f %f\n", mean[ 0 ], mean[ 1 ] );

			ms[ 2 * ( number_of_landmarks - 1 ) ] = mean[ 0 ];
			ms[ 2 * ( number_of_landmarks - 1 ) + 1 ] = mean[ 1 ];

			free( mean );
		}
	}

	//printf( "size_of_landmark_keys_zlist: %d\n", *size_of_landmark_keys_zlist );
	//printf( "number_of_landmarks: %d\n", number_of_landmarks );
	//printf( "order_of_landmark_ids: %d %d %d %d %d %d\n", order_of_landmark_ids[ 0 ], order_of_landmark_ids[ 1 ], order_of_landmark_ids[ 2 ], order_of_landmark_ids[ 3 ], order_of_landmark_ids[ 4 ], order_of_landmark_ids[ 5 ] );
	//printf( "count_of_landmark_ids: %d %d %d %d %d %d\n", count_of_landmark_ids[ order_of_landmark_ids[ 0 ] ], count_of_landmark_ids[ order_of_landmark_ids[ 1 ] ], count_of_landmark_ids[ order_of_landmark_ids[ 2 ] ], count_of_landmark_ids[ order_of_landmark_ids[ 3 ] ], count_of_landmark_ids[ order_of_landmark_ids[ 4 ] ], count_of_landmark_ids[ order_of_landmark_ids[ 5 ] ] );
	//printf( "ms: %f %f %f %f %f %f %f %f %f %f %f %f\n", ms[ 0 ], ms[ 1 ], ms[ 2 ], ms[ 3 ], ms[ 4 ], ms[ 5 ], ms[ 6 ], ms[ 7 ], ms[ 8 ], ms[ 9 ], ms[ 10 ], ms[ 11 ] );

    if ( toDraw >= 1 ) {
	    PyObject *SYS = PyImport_ImportModule( "sys" );
	    PyObject *PATH = PyObject_GetAttrString( SYS, "path" );
	    PyList_Append( PATH, PyUnicode_FromString( "." ) );

	    PyObject *NAME = PyUnicode_FromString( "GSLAM_draw" );
	    PyObject *MODULE = PyImport_Import( NAME );
	    PyObject *DICT = PyModule_GetDict( MODULE );
	    PyObject *FUNC = PyDict_GetItemString( DICT, "draw" );
	    PyObject *ARGS = PyTuple_New( 7 );

	    PyObject *HAT_Xs_STEPs = PyList_New( *size_of_hat_xs );
	    PyObject *HAT_Xs = PyList_New( *size_of_hat_xs * 3);

	    PyObject *ZLIST_STEPs = PyList_New( *size_of_zlist );
	    PyObject *ZLIST_LANDMARK_IDs = PyList_New( *size_of_zlist );
	    PyObject *Zs = PyList_New( *size_of_zlist * 2 );
   
	    PyObject *MSes_INDEXes = PyList_New( number_of_landmarks );
	    PyObject *MSes = PyList_New( number_of_landmarks * 2 );

	    for ( unsigned int i = 0; i < *size_of_hat_xs; i ++ ) {
	    	PyList_SetItem( HAT_Xs_STEPs, i, PyLong_FromUnsignedLong( hat_xs[ i ].step ) );

	    	PyList_SetItem( HAT_Xs, i * 3, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 0 ] ) );
	    	PyList_SetItem( HAT_Xs, i * 3 + 1, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 1 ] ) );
	    	PyList_SetItem( HAT_Xs, i * 3 + 2, PyFloat_FromDouble( hat_xs[ i ].hat_x[ 2 ] ) );
	    }

	    PyTuple_SetItem( ARGS, 0, HAT_Xs_STEPs );
	    PyTuple_SetItem( ARGS, 1, HAT_Xs );

	    for ( unsigned int i = 0; i < *size_of_zlist; i ++ ) {
	    	PyList_SetItem( ZLIST_STEPs, i, PyLong_FromUnsignedLong( zlist[ i ].step ) );

	    	PyList_SetItem( ZLIST_LANDMARK_IDs, i, PyLong_FromUnsignedLong( zlist[ i ].landmark_id ) );

	    	PyList_SetItem( Zs, i * 2, PyFloat_FromDouble( zlist[ i ].z[ 0 ] ) );
	    	PyList_SetItem( Zs, i * 2 + 1, PyFloat_FromDouble( zlist[ i ].z[ 1 ] ) );
	    }

	    PyTuple_SetItem( ARGS, 2, ZLIST_STEPs );
	    PyTuple_SetItem( ARGS, 3, ZLIST_LANDMARK_IDs );
	    PyTuple_SetItem( ARGS, 4, Zs );

	    for ( unsigned int i = 0; i < number_of_landmarks; i ++ ) {
	    	PyList_SetItem( MSes_INDEXes, i, PyLong_FromUnsignedLong( order_of_landmark_ids[ i ] ) );

	    	PyList_SetItem( MSes, i * 2, PyFloat_FromDouble( ms[ i * 2 ] ) );
	    	PyList_SetItem( MSes, i * 2 + 1, PyFloat_FromDouble( ms[ i * 2 + 1 ] ) );
	    }

	    PyTuple_SetItem( ARGS, 5, MSes_INDEXes );
	    PyTuple_SetItem( ARGS, 6, MSes );

	    PyObject_CallObject( FUNC, ARGS );
    }

    free( map_edges );
    free( ms );

    if ( toDraw >= 1 )
        Py_Finalize();

    free( delta );

    {
        const char *dump_path = getenv( "POSE_DUMP_FILE" );
        if ( dump_path ) {
            FILE *df = fopen( dump_path, "w" );
            if ( df ) {
                for ( unsigned int i = 0; i < *size_of_hat_xs; i++ ) {
                    fprintf( df, "%u %.17g %.17g %.17g\n",
                        hat_xs[i].step, hat_xs[i].hat_x[0],
                        hat_xs[i].hat_x[1], hat_xs[i].hat_x[2] );
                }
                fclose( df );
            }
        }
    }

    free( us );
	free( hat_xs );
    free( zlist );

    free( size_of_us );
    free( size_of_hat_xs );
    free( size_of_zlist );

    return 0;
}

/* import itertools
def make_edges(hat_xs, zlist):
    landmark_keys_zlist = {}

    for step in zlist:
        for z in zlist[step]:
            landmark_id = z[0]
            if landmark_id not in landmark_keys_zlist:
                landmark_keys_zlist[landmark_id] = []

            landmark_keys_zlist[landmark_id].append((step, z))

    edges = []
    for landmark_id in landmark_keys_zlist:
        step_pairs = list(itertools.combinations(landmark_keys_zlist[landmark_id], 2))
        edges += [ObsEdge(xz1[0], xz2[0], xz1[1], xz2[1], hat_xs) for xz1, xz2 in step_pairs]

    return edges, landmark_keys_zlist #ランドマークをキーにしたリストlandmark_keys_zlistも返す */

void make_edges( struct HAT_X *hat_xs, unsigned int size_of_hat_xs, struct Z *zlist, unsigned int size_of_zlist, struct Edge *edges, unsigned int *size_of_edges, struct LandmarkKeysZList *landmark_keys_zlist, unsigned int *size_of_landmark_keys_zlist ) {
	unsigned int max_id = 0;

	for ( unsigned int i = 0; i < size_of_zlist; i ++ ) { // Find max_id
		if ( zlist[ i ].landmark_id > max_id ) {
			max_id = zlist[ i ].landmark_id;
		}
	}

	//printf( "max_id: %u\n", max_id );

	unsigned int *id_count = (unsigned int *)calloc ( max_id + 1, sizeof( unsigned int ) );
	unsigned int *temp_landmark_ids = (unsigned int *)malloc ( sizeof( unsigned int ) * size_of_zlist );

	for ( unsigned int i = 0; i < size_of_zlist; i ++ ) {
		temp_landmark_ids[ i ] = zlist[ i ].landmark_id;
		id_count[ temp_landmark_ids[ i ] ] ++;
	}

	unsigned int *id_list = (unsigned int *)malloc ( sizeof( unsigned int ) * ( max_id + 1 ) );

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
		id_list[ i ] = max_id + 1;

		//printf( "%u id_count: %u\n", i, id_count[ i ] );
	}

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
		//printf( "id_list: %u\n", id_list[i] );

		for ( unsigned int j = 0; j < size_of_zlist; j ++ ) {
			if ( id_list[ i ] == max_id + 1 && id_list[ i ] != temp_landmark_ids[ j ] ) {
				id_list[ i ] = temp_landmark_ids[ j ];
				temp_landmark_ids[ j ] = max_id + 1;
			}

			else if ( id_list[ i ] != max_id + 1 && id_list[ i ] == temp_landmark_ids[ j ] ) {
				temp_landmark_ids[ j ] = max_id + 1;
			}

			else {
				continue;
			}
		}

		//printf( "id_list: %u\n\n", id_list[i] );
		//printf( "%u id_count: %u\n", i, id_count[ i ] );
	}

	*size_of_landmark_keys_zlist = size_of_zlist;

	//printf( "# of Landmark_keys_zlist: %d\n", *size_of_landmark_keys_zlist );

	unsigned int count = 0;

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
		for ( unsigned int j = 0; j < size_of_zlist; j ++ ) {
			if ( zlist[ j ].landmark_id == id_list[ i ] ) {
				landmark_keys_zlist[ count ].landmark_id = zlist[ j ].landmark_id;
				landmark_keys_zlist[ count ].z = zlist[ j ];

				count ++;
			}
		}
	}

	/*for ( unsigned int j = 0; j < size_of_zlist; j ++ ) {
		printf( "Landmark_keys_zlist: %d, ", landmark_keys_zlist[ j ].landmark_id );
		printf( "( %u,", landmark_keys_zlist[ j ].z.step );
		printf( " %u,", landmark_keys_zlist[ j ].z.landmark_id );
		printf( " %f,", landmark_keys_zlist[ j ].z.z[0] );
		printf( " %f )\n", landmark_keys_zlist[ j ].z.z[1] );
	}*/

	unsigned int sum = 0;

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
		//printf( "i : %u \n", i );
		//printf( "id_list: %u\n", id_list[i] );

		if ( id_list[ i ] != max_id + 1 && id_count[ id_list[ i ] ] >= 2 ) {
			//printf( "id_count : %u \n", id_count[ id_list[ i ] ] );
			//printf( "c : %lu \n", combination( id_count[ id_list[ i ] ], 2 ) );

			sum += combination( id_count[ id_list[ i ] ], 2 );
		}
	}

	//printf( "sum : %u \n", sum );

	*size_of_edges = sum;

	unsigned int position = 0;

	count = 0;

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
		if ( id_list[ i ] != max_id + 1 ) {
		for ( unsigned int j = 0; j < id_count[ id_list[ i ] ]; j ++ ) {
			for ( unsigned int k = 0; k < id_count[ id_list[ i ] ]; k ++ ) {
    			if ( j >= k ) {
					continue;
				}

				else {
					/*printf( "Step Pair: " );
					printf( "( ( %u,", landmark_keys_zlist[ position + j ].z.step );
					printf( " %u,", landmark_keys_zlist[ position + j ].z.landmark_id );
					printf( " %f,", landmark_keys_zlist[ position + j ].z.z[0] );
					printf( " %f ), ", landmark_keys_zlist[ position + j ].z.z[1] );
					
					printf( "( %u,", landmark_keys_zlist[ position + k ].z.step );
					printf( " %u,", landmark_keys_zlist[ position + k ].z.landmark_id );
					printf( " %f,", landmark_keys_zlist[ position + k ].z.z[0] );
					printf( " %f ) )\n", landmark_keys_zlist[ position + k ].z.z[1] );*/
					
					double snr[2] = { 0.14, 0.05 };
					edges[ count ] = *ObsEdge_create( landmark_keys_zlist[ position + j ].z, landmark_keys_zlist[ position + k ].z, hat_xs, snr );

					count ++;
					//printf( "count : %u \n", count );
				}
			}
		}

		/* count = 0;

		for ( unsigned int j = 0; j < id_count[ id_list[ i ] ]; j ++ ) {
			for ( unsigned int k = 0; k < id_count[ id_list[ i ] ]; k ++ ) {
				if ( j >= k ) {
					continue;
				}

				else {
					printf( "Step Pair: " );
					printf( "( ( %u,", step_pairs[ count ].xz1.step );
					printf( " %u,", step_pairs[ count ].xz1.landmark_id );
					printf( " %f,", step_pairs[ count ].xz1.z[0] );
					printf( " %f ), ", step_pairs[ count ].xz1.z[1] );

					printf( "( %u,", step_pairs[ count ].xz2.step );
					printf( " %u,", step_pairs[ count ].xz2.landmark_id );
					printf( " %f,", step_pairs[ count ].xz2.z[0] );
					printf( " %f ) )\n", step_pairs[ count ].xz2.z[1] );

					count ++;
				}
			}
		} */

		position += id_count[ id_list[ i ] ];
		} // end if id_list[i] != max_id + 1

		//printf( "position: %d\n", position );

		if ( count >= sum )
			break;
	}
	
	//printf( "position: %d\n", position );
	free( id_count );
	free( temp_landmark_ids );
	free( id_list );

}

/* def add_edge(edge, Omega, xi):
    f1, f2 = edge.t1*3, edge.t2*3
    t1 ,t2 = f1 + 3, f2 + 3
    Omega[f1:t1, f1:t1] += edge.omega_upperleft
    Omega[f1:t1, f2:t2] += edge.omega_upperright
    Omega[f2:t2, f1:t1] += edge.omega_bottomleft
    Omega[f2:t2, f2:t2] += edge.omega_bottomright
    xi[f1:t1] += edge.xi_upper
    xi[f2:t2] += edge.xi_bottom */

void add_edge( struct Edge edge, double *Omega, unsigned int Omega_y, unsigned int Omega_x, double *xi ) {
	unsigned int f1 = edge.t1 * 3;
	unsigned int f2 = edge.t2 * 3;

	//printf( "edge.xi_upper: %f %f %f\n", edge.xi_upper[0], edge.xi_upper[1], edge.xi_upper[2] );
	//printf( "edge.xi_bottom: %f %f %f\n", edge.xi_bottom[0], edge.xi_bottom[1], edge.xi_bottom[2] );

	Omega[ ( f1 + 0 ) * Omega_x + f1 + 0 ] += edge.omega_upperleft[ 0 ];
	Omega[ ( f1 + 0 ) * Omega_x + f1 + 1 ] += edge.omega_upperleft[ 1 ];
	Omega[ ( f1 + 0 ) * Omega_x + f1 + 2 ] += edge.omega_upperleft[ 2 ];
	Omega[ ( f1 + 1 ) * Omega_x + f1 + 0 ] += edge.omega_upperleft[ 3 ];
	Omega[ ( f1 + 1 ) * Omega_x + f1 + 1 ] += edge.omega_upperleft[ 4 ];
	Omega[ ( f1 + 1 ) * Omega_x + f1 + 2 ] += edge.omega_upperleft[ 5 ];
	Omega[ ( f1 + 2 ) * Omega_x + f1 + 0 ] += edge.omega_upperleft[ 6 ];
	Omega[ ( f1 + 2 ) * Omega_x + f1 + 1 ] += edge.omega_upperleft[ 7 ];
	Omega[ ( f1 + 2 ) * Omega_x + f1 + 2 ] += edge.omega_upperleft[ 8 ];

	Omega[ ( f1 + 0 ) * Omega_x + f2 + 0 ] += edge.omega_upperright[ 0 ];
	Omega[ ( f1 + 0 ) * Omega_x + f2 + 1 ] += edge.omega_upperright[ 1 ];
	Omega[ ( f1 + 0 ) * Omega_x + f2 + 2 ] += edge.omega_upperright[ 2 ];
	Omega[ ( f1 + 1 ) * Omega_x + f2 + 0 ] += edge.omega_upperright[ 3 ];
	Omega[ ( f1 + 1 ) * Omega_x + f2 + 1 ] += edge.omega_upperright[ 4 ];
	Omega[ ( f1 + 1 ) * Omega_x + f2 + 2 ] += edge.omega_upperright[ 5 ];
	Omega[ ( f1 + 2 ) * Omega_x + f2 + 0 ] += edge.omega_upperright[ 6 ];
	Omega[ ( f1 + 2 ) * Omega_x + f2 + 1 ] += edge.omega_upperright[ 7 ];
	Omega[ ( f1 + 2 ) * Omega_x + f2 + 2 ] += edge.omega_upperright[ 8 ];

	Omega[ ( f2 + 0 ) * Omega_x + f1 + 0 ] += edge.omega_bottomleft[ 0 ];
	Omega[ ( f2 + 0 ) * Omega_x + f1 + 1 ] += edge.omega_bottomleft[ 1 ];
	Omega[ ( f2 + 0 ) * Omega_x + f1 + 2 ] += edge.omega_bottomleft[ 2 ];
	Omega[ ( f2 + 1 ) * Omega_x + f1 + 0 ] += edge.omega_bottomleft[ 3 ];
	Omega[ ( f2 + 1 ) * Omega_x + f1 + 1 ] += edge.omega_bottomleft[ 4 ];
	Omega[ ( f2 + 1 ) * Omega_x + f1 + 2 ] += edge.omega_bottomleft[ 5 ];
	Omega[ ( f2 + 2 ) * Omega_x + f1 + 0 ] += edge.omega_bottomleft[ 6 ];
	Omega[ ( f2 + 2 ) * Omega_x + f1 + 1 ] += edge.omega_bottomleft[ 7 ];
	Omega[ ( f2 + 2 ) * Omega_x + f1 + 2 ] += edge.omega_bottomleft[ 8 ];

	Omega[ ( f2 + 0 ) * Omega_x + f2 + 0 ] += edge.omega_bottomright[ 0 ];
	Omega[ ( f2 + 0 ) * Omega_x + f2 + 1 ] += edge.omega_bottomright[ 1 ];
	Omega[ ( f2 + 0 ) * Omega_x + f2 + 2 ] += edge.omega_bottomright[ 2 ];
	Omega[ ( f2 + 1 ) * Omega_x + f2 + 0 ] += edge.omega_bottomright[ 3 ];
	Omega[ ( f2 + 1 ) * Omega_x + f2 + 1 ] += edge.omega_bottomright[ 4 ];
	Omega[ ( f2 + 1 ) * Omega_x + f2 + 2 ] += edge.omega_bottomright[ 5 ];
	Omega[ ( f2 + 2 ) * Omega_x + f2 + 0 ] += edge.omega_bottomright[ 6 ];
	Omega[ ( f2 + 2 ) * Omega_x + f2 + 1 ] += edge.omega_bottomright[ 7 ];
	Omega[ ( f2 + 2 ) * Omega_x + f2 + 2 ] += edge.omega_bottomright[ 8 ];

	xi[ f1 + 0 ] += edge.xi_upper[ 0 ];
	xi[ f1 + 1 ] += edge.xi_upper[ 1 ];
	xi[ f1 + 2 ] += edge.xi_upper[ 2 ];

	xi[ f2 + 0 ] += edge.xi_bottom[ 0 ];
	xi[ f2 + 1 ] += edge.xi_bottom[ 1 ];
	xi[ f2 + 2 ] += edge.xi_bottom[ 2 ];

	//printf( "Omega: %f\n", Omega[0] );
	//printf( "xi: %f\n", xi[0] );
}

/* class IdealRobot:
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0), nu*math.sin(t0), omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)), omega*time ] ) */

void state_transition( double *result, double nu, double omega, unsigned int time, struct HAT_X pose ) {
    double t0 = pose.hat_x[2];

    if ( fabs(omega) < 1e-10 ) {
        double vector[3] = { nu * cos( t0 ), nu * sin( t0 ), omega };

        double multiplication[ 3 ] = { 0.0 };
        mul( multiplication, vector, 3, 1, time );

        double addition[ 3 ] = { 0.0 };
        add( addition, pose.hat_x, 3, 1, multiplication, 3, 1 );

        for ( unsigned int i = 0; i < 3; i ++ ) {
			result[ i ] = addition[ i ];
		}
    }

    else {
        double vector[3] = { nu / omega * sin( t0 + omega * time ) - nu / omega * sin( t0 ), nu / omega * cos( t0 ) - nu / omega * cos( t0 + omega * time ), omega * time };

        double addition[ 3 ] = { 0.0 };

        add( addition, pose.hat_x, 3, 1, vector, 3, 1 );

        for ( unsigned int i = 0; i < 3; i ++ ) {
			result[ i ] = addition[ i ];
		}
    }
}

unsigned int combination( unsigned int n, unsigned int r ) {
	unsigned int p = n;
	unsigned int f = r;

	for ( unsigned int i = 1; i < r; i ++ ) {
		p = p * ( n - i );
		f = f * ( r - i );
	}


    return p / f;
}

/* def read_data():
    hat_xs = {}
    zlist = {}
    delta = 0.0
    us = {}

    with open("../data/log_edited.txt") as f:
    	for line in f.readlines():
    	    tmp = line.rstrip().split()

            step = int(tmp[1])
            if tmp[0] == "x":
                hat_xs[step] = np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]).T
            elif tmp[0] == "z":
                if step not in zlist:
                    zlist[step] = []
                zlist[step].append((int(tmp[2]), np.array([float(tmp[3]), float(tmp[4])]).T))
            elif tmp[0] == "delta":
                delta = float(tmp[1])
            elif tmp[0] == "u":
                us[step] = np.array([float(tmp[2]), float(tmp[3])]).T

        return hat_xs, zlist, us, delta */

unsigned int read_data( char *file_name, double *delta, struct U *us, unsigned int *size_of_us, struct HAT_X *hat_xs, unsigned int *size_of_hat_xs, struct Z *zlist, unsigned int *size_of_zlist  ) {
    char line[1024];
    FILE *input_file;

    if ( ( input_file = fopen( file_name, "r" ) ) == NULL ) {
        printf( "Error! Cannot open the input file!" );
        exit( 1 );
    }

    unsigned int step_index = 0;
    unsigned int num_of_us = 0;
    unsigned int num_of_xs = 0;
    unsigned int num_of_zs = 0;

    while (fgets(line, sizeof(line), input_file) != 0) {
        //printf("%s", line);

        char *token = strtok(line, " ");

        if ( strcmp( token, "u" ) == 0 ) {
            //printf( "%s\n", token );

            token = strtok(NULL, " ");
            step_index = atoi( token );
            us[ num_of_us ].step = step_index;

            token = strtok(NULL, " ");
            us[ num_of_us ].nu = atof( token );

            token = strtok(NULL, " ");
            us[ num_of_us ].omega = atof( token );

            //printf( "u[%d]: {", num_of_us );
            //printf( " %d, ", us[ num_of_us ].step );
            //printf( " %f, ", us[ num_of_us ].nu );
            //printf( "%f }\n", us[ num_of_us ].omega );

            num_of_us ++;
        }

        else if ( strcmp( token, "x" ) == 0 ) {
            //printf( "%s\n", token );

            token = strtok(NULL, " ");
            step_index = atoi( token );
            hat_xs[ num_of_xs ].step = step_index;

            token = strtok(NULL, " ");
            hat_xs[ num_of_xs ].hat_x[0] = atof( token );

            token = strtok(NULL, " ");
            hat_xs[ num_of_xs ].hat_x[1] = atof( token );

            token = strtok(NULL, " ");
            hat_xs[ num_of_xs ].hat_x[2] = atof( token );

            //printf( "hat_x[%d]: {", num_of_xs );
            //printf( " %d, ", hat_xs[ num_of_xs ].step );
            //printf( " %f, ", hat_xs[ num_of_xs ].hat_x[0] );
            //printf( "%f, ", hat_xs[ num_of_xs ].hat_x[1] );
            //printf( "%f }\n", hat_xs[ num_of_xs ].hat_x[2] );

            num_of_xs ++;
        }

        else if ( strcmp( token, "z" ) == 0 ) {
            //printf( "%s\n", token );

            token = strtok(NULL, " ");
            step_index = atoi( token );
            zlist[ num_of_zs ].step = step_index;

            token = strtok(NULL, " ");
            zlist[ num_of_zs ].landmark_id = atoi( token );

            token = strtok(NULL, " ");
            zlist[ num_of_zs ].z[0] = atof( token );

            token = strtok(NULL, " ");
            zlist[ num_of_zs ].z[1] = atof( token );

            //printf( "z[%d]: {", num_of_zs );
            //printf( " %d, ", zlist[ num_of_zs ].step );
            //printf( " %d, ", zlist[ num_of_zs ].landmark_id );
            //printf( "%f, ", zlist[ num_of_zs ].z[0] );
            //printf( "%f }\n", zlist[ num_of_zs ].z[1] );

            num_of_zs ++;
        }

        else { //delta
            //printf( "%s\n", token );

            token = strtok(NULL, " ");
            *delta = atof( token );

            //printf( "delta: %f\n", *delta );
        }
    }

    fclose(input_file);

    *size_of_us = num_of_us;
	*size_of_hat_xs = num_of_xs;
    *size_of_zlist = num_of_zs;

    //printf( "delta: %f ", *delta );
    //printf( "# of u: %u ", *size_of_us );
	//printf( "# of x: %u ", *size_of_hat_xs );
    //printf( "# of z: %u\n", *size_of_zlist );

    return num_of_us;
}
