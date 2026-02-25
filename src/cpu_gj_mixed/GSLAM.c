#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "GSLAM.h"
#include "MatrixOps.h"
#include "MotionEdge.h"
#include "ObsEdge.h"
#include "MapEdge.h"

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

    double delta = 0.0;

    unsigned int size_of_us = 0;
    unsigned int size_of_hat_xs = 0;
    unsigned int size_of_zlist = 0;

    struct U *us = (U*) malloc( sizeof( struct U ) * 131072 );
    struct HAT_X *hat_xs = (HAT_X*) malloc( sizeof( struct HAT_X ) * 131072 );
    struct Z *zlist = (Z*) malloc( sizeof( struct Z ) * 524288 );

    unsigned int dim = read_data( argv[5], &delta, us, &size_of_us, hat_xs, &size_of_hat_xs, zlist, &size_of_zlist ) * 3;

    printf( "delta: %f ", delta );
    printf( "size_of_us: %d ", size_of_us );
    printf( "size_of_hat_xs: %d ", size_of_hat_xs );
    printf( "size_of_zlist: %d ", size_of_zlist );
    printf( "dim: %d\n", dim );

    double last_cost = 0.0, cost = 0.0, diff = 0.0;
    double best_diff_even = 1e30, best_diff_odd = 1e30;
    unsigned int no_improve_even = 0, no_improve_odd = 0;

    for ( unsigned int n = 1; n <= numberOfRounds; n ++ ) {
        struct Edge *edges = (Edge*) malloc( sizeof( struct Edge ) * 4194304 );
        struct LandmarkKeysZList *landmark_keys_zlist = (LandmarkKeysZList*) malloc( sizeof( struct LandmarkKeysZList ) * 4194304 );

        unsigned int size_of_edges = 0;
        unsigned int size_of_landmark_keys_zlist = 0;

        make_edges( hat_xs, size_of_hat_xs, zlist, size_of_zlist, edges, &size_of_edges, landmark_keys_zlist, &size_of_landmark_keys_zlist );

        free( landmark_keys_zlist );

        double mns[4] = { 0.19, 0.001, 0.13, 0.2 };

        for ( unsigned int i = 0; i < size_of_hat_xs - 1; i ++ ) {
            MotionEdge_create( &edges[ size_of_edges + i ], i, i + 1, hat_xs, us, delta, lambda, mns );
        }

        size_of_edges = size_of_edges + size_of_hat_xs - 1;

        if ( !toUseDiff ) {
	    for( unsigned int i = 0; i < size_of_edges; i ++ ) {
                double transposeOFE[ 1 * 3 ] = { 0.0 };
                tra( transposeOFE, edges[ i ].hat_e, 3, 1 );

                double dot_productOFtEOmega[ 1 * 3 ] = { 0.0 };
                dot( dot_productOFtEOmega, transposeOFE, 1, 3, edges[ i ].Omega, 3, 3 );

                double dot_productOFtEOmegaE[ 1 * 1 ] = { 0.0 };
                dot( dot_productOFtEOmegaE, dot_productOFtEOmega, 1, 3, edges[ i ].hat_e, 3, 1 );

                cost += dot_productOFtEOmegaE[ 0 ] * edges[ i ].lambda;
            }
	    }

    	double *Omega = (double*) malloc( sizeof( double ) * dim * dim );

        for ( unsigned int j = 0; j < dim; j ++ ) {
            for ( unsigned int k = 0; k < dim; k ++ ) {
                if ( j == k && j < 3 ) {
                    Omega[ dim * j + k ] = 1000000.0;
                }

                else {
                    Omega[ dim * j + k ] = 0.0;
                }
            }
        }

        double *xi = (double*) malloc( sizeof( double ) * dim );

        for ( unsigned int i = 0; i < dim; i ++ ) {
            xi[ i ] = 0.0;
        }

        for ( unsigned int i = 0; i < size_of_edges; i ++ ) {
            add_edge( edges[ i ], Omega, dim, dim, xi );
        }

        double *delta_xs = (double*) malloc( sizeof( double ) * dim );
        double *inverse  = (double*) malloc( sizeof( double ) * dim * dim );

        // Mixed precision: cast Omega to float, invert in FP32, cast result back to FP64
        float *Omega_f   = (float*) malloc( sizeof( float ) * dim * dim );
        float *inverse_f = (float*) malloc( sizeof( float ) * dim * dim );

        for ( unsigned int j = 0; j < dim * dim; j++ )
            Omega_f[ j ] = (float)Omega[ j ];

        inv_float( inverse_f, Omega_f, dim );

        for ( unsigned int j = 0; j < dim * dim; j++ )
            inverse[ j ] = (double)inverse_f[ j ];

        free( Omega_f );
        free( inverse_f );

        dot( delta_xs, inverse, dim, dim, xi, dim, 1 );
        free( inverse );

        for ( unsigned int i = 0; i < size_of_hat_xs; i ++ ) {
            hat_xs[i].hat_x[0] = hat_xs[i].hat_x[0] + delta_xs[ i * 3 ];
            hat_xs[i].hat_x[1] = hat_xs[i].hat_x[1] + delta_xs[ i * 3 + 1 ];
            hat_xs[i].hat_x[2] = hat_xs[i].hat_x[2] + delta_xs[ i * 3 + 2 ];
        }

        if ( toUseDiff )
            diff = norm( delta_xs, dim, 1 ) / sqrt( (double)dim );

	    else
	        diff = fabs( cost - last_cost ) / cost;

	    printf( "%d rounds executed: %.17g\n", n, diff );

        free( Omega );
        free( xi );

        free( delta_xs );

        free( edges );

        if ( toUseDiff ) {
            if ( diff < 0.01 ) {
                break;
            }
            if ( n % 2 == 0 ) {
                if ( diff < best_diff_even ) { best_diff_even = diff; no_improve_even = 0; }
                else { if ( ++no_improve_even >= 10 ) break; }
            } else {
                if ( diff < best_diff_odd ) { best_diff_odd = diff; no_improve_odd = 0; }
                else { if ( ++no_improve_odd >= 10 ) break; }
            }
        }

	    else {
            if( diff < 1e-8 ) {
                break;
            }

	        last_cost = cost;
	    }
    }

    struct Edge *edges = (Edge*) malloc( sizeof( struct Edge ) * 4194304 );
    struct LandmarkKeysZList *landmark_keys_zlist = (LandmarkKeysZList*) malloc( sizeof( struct LandmarkKeysZList ) * 4194304 );

    unsigned int size_of_edges = 0;
    unsigned int size_of_landmark_keys_zlist = 0;

    make_edges( hat_xs, size_of_hat_xs, zlist, size_of_zlist, edges, &size_of_edges, landmark_keys_zlist, &size_of_landmark_keys_zlist );

    free( edges );

    unsigned int number_of_landmarks = 0;
    unsigned int landmark_id = size_of_landmark_keys_zlist;

    unsigned int order_of_landmark_ids[ 1024 ] = { 0 };

    struct MapEdge *map_edges = (MapEdge*) malloc( sizeof( struct MapEdge ) * 1048576 );
    unsigned int count_of_map_edges = 0;

    double *ms = (double*) malloc( sizeof( double ) * 2 * 131072 );

    struct Z head_z = { 0 };
    double snr[2] = { 0.14, 0.05 };

    for ( unsigned int i = 0; i < size_of_landmark_keys_zlist; i ++ ) {
        if ( landmark_id != landmark_keys_zlist[ i ].landmark_id ) {
            landmark_id = landmark_keys_zlist[ i ].landmark_id;
            order_of_landmark_ids[ number_of_landmarks ] = landmark_id;

            number_of_landmarks ++;

            head_z = landmark_keys_zlist[ i ].z;
			count_of_map_edges = 0;
		}

		MapEdge_create ( &map_edges[ count_of_map_edges ], landmark_keys_zlist[ i ].z, head_z, hat_xs, snr );
		count_of_map_edges ++;

		if ( i == size_of_landmark_keys_zlist - 1 || ( i < size_of_landmark_keys_zlist - 1 && landmark_keys_zlist[ i ].landmark_id != landmark_keys_zlist[ i + 1 ].landmark_id ) ) {
			double mean[ 2 ] = { 0.0 };
			double *m = (double*)calloc( 2 * count_of_map_edges, sizeof( double ) );

			for ( unsigned int n = 0; n < count_of_map_edges; n ++ ) {
				m[ 2 * n ] = map_edges[ n ].m[ 0 ];
				m[ 2 * n + 1 ] = map_edges[ n ].m[ 1 ];
			}

			vec_mean( mean, m, 2, count_of_map_edges );
			free( m );

			ms[ 2 * ( number_of_landmarks - 1 ) ] = mean[ 0 ];
			ms[ 2 * ( number_of_landmarks - 1 ) + 1 ] = mean[ 1 ];
		}
	}

    free( map_edges );
    free( ms );

    if ( toDraw >= 1 ) {
        const char *dump_path = getenv( "POSE_DUMP_FILE" );
        if ( dump_path ) {
            FILE *df = fopen( dump_path, "w" );
            if ( df ) {
                for ( unsigned int i = 0; i < size_of_hat_xs; i++ ) {
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

    return 0;
}

void make_edges( struct HAT_X *hat_xs, unsigned int size_of_hat_xs, struct Z *zlist, unsigned int size_of_zlist, struct Edge *edges, unsigned int *size_of_edges, struct LandmarkKeysZList *landmark_keys_zlist, unsigned int *size_of_landmark_keys_zlist ) {
	unsigned int max_id = 0;

	for ( unsigned int i = 0; i < size_of_zlist; i ++ ) { // Find max_id
		if ( zlist[ i ].landmark_id > max_id ) {
			max_id = zlist[ i ].landmark_id;
		}
	}

	unsigned int *id_count = (unsigned int*)calloc( max_id + 1, sizeof( unsigned int ) );

	unsigned int *temp_landmark_ids = (unsigned int*)malloc( sizeof( unsigned int ) * size_of_zlist );

	for ( unsigned int i = 0; i < size_of_zlist; i ++ ) {
		temp_landmark_ids[ i ] = zlist[ i ].landmark_id;
		id_count[ temp_landmark_ids[ i ] ] ++;
	}

	unsigned int *id_list = (unsigned int*)malloc( sizeof( unsigned int ) * ( max_id + 1 ) );

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
		id_list[ i ] = max_id + 1;
	}

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
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
	}

	*size_of_landmark_keys_zlist = size_of_zlist;

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

	unsigned int sum = 0;

	for ( unsigned int i = 0; i < max_id + 1; i ++ ) {
		if ( id_list[ i ] != max_id + 1 && id_count[ id_list[ i ] ] >= 2 ) {
			sum += combination( id_count[ id_list[ i ] ], 2 );
		}
	}

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
					double snr[2] = { 0.14, 0.05 };
					ObsEdge_create( &edges[ count ], landmark_keys_zlist[ position + j ].z, landmark_keys_zlist[ position + k ].z, hat_xs, snr );

					count ++;
				}
			}
		}

		position += id_count[ id_list[ i ] ];
		} // end if id_list[i] != max_id + 1

		if ( count >= sum )
			break;
	}
	free( id_count );
	free( temp_landmark_ids );
	free( id_list );
}

void add_edge( struct Edge edge, double *Omega, unsigned int Omega_y, unsigned int Omega_x, double *xi ) {
	unsigned int f1 = edge.t1 * 3;
	unsigned int f2 = edge.t2 * 3;

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
}

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
        char *token = strtok(line, " ");

        if ( strcmp( token, "u" ) == 0 ) {
            token = strtok(NULL, " ");
            step_index = atoi( token );
            us[ num_of_us ].step = step_index;

            token = strtok(NULL, " ");
            us[ num_of_us ].nu = atof( token );

            token = strtok(NULL, " ");
            us[ num_of_us ].omega = atof( token );

            num_of_us ++;
        }

        else if ( strcmp( token, "x" ) == 0 ) {
            token = strtok(NULL, " ");
            step_index = atoi( token );
            hat_xs[ num_of_xs ].step = step_index;

            token = strtok(NULL, " ");
            hat_xs[ num_of_xs ].hat_x[0] = atof( token );

            token = strtok(NULL, " ");
            hat_xs[ num_of_xs ].hat_x[1] = atof( token );

            token = strtok(NULL, " ");
            hat_xs[ num_of_xs ].hat_x[2] = atof( token );

            num_of_xs ++;
        }

        else if ( strcmp( token, "z" ) == 0 ) {
            token = strtok(NULL, " ");
            step_index = atoi( token );
            zlist[ num_of_zs ].step = step_index;

            token = strtok(NULL, " ");
            zlist[ num_of_zs ].landmark_id = atoi( token );

            token = strtok(NULL, " ");
            zlist[ num_of_zs ].z[0] = atof( token );

            token = strtok(NULL, " ");
            zlist[ num_of_zs ].z[1] = atof( token );

            num_of_zs ++;
        }

        else { //delta
            token = strtok(NULL, " ");
            *delta = atof( token );
        }
    }

    fclose(input_file);

    *size_of_us = num_of_us;
	*size_of_hat_xs = num_of_xs;
    *size_of_zlist = num_of_zs;

    return num_of_us;
}
