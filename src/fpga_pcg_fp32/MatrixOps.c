#include "MatrixOps.h"
#include "Edge.h"

double norm( double *mat, unsigned int mat_y, unsigned int mat_x ) {
    double result = 0.0;

    for ( unsigned int m = 0; m < mat_y; m ++ ) {
        for ( unsigned int n = 0; n < mat_x; n ++ ) {
			//printf( "mat[ n * mat_x + m ]: %f\n", mat[ n * mat_x + m ] );

            result += pow( mat[ n * mat_x + m ], 2.0 );
        }
    }

    return sqrt( result );
}

void tra( double *result, double *mat, unsigned int mat_y, unsigned int mat_x ) {
    unsigned int result_y = mat_x;
    unsigned int result_x = mat_y;

    for ( unsigned int m = 0; m < result_y; m ++ ) {
        for ( unsigned int n = 0; n < result_x; n ++ ) {
            result[ m * result_x + n ] = mat[ n * mat_x + m ];
        }
    }
}

void inv( double *result, double *mat, unsigned int dim ) {
    //printf( "dim: %d\n", dim );
	double temp[ dim * dim ];
	//double temp[ 3 * 3 ] = { 0.0 }; // dim * dim

	for ( unsigned int i = 0; i < dim * dim; i ++ ) {
		temp[ i ] = 0.0;
		//printf( "result[ %d ]: %f\n", i, temp[ i ] );
	}

	double ratio = 0.0;

	//printf( "dim: %d\n", dim );

	for ( unsigned int m = 0; m < dim; m ++ ) {
		for ( unsigned int n = 0; n < dim; n ++ ) {
			temp[ m * dim + n ] = mat[ m * dim + n ];

			if( m == n ) {
				result[ m * dim + n ] = 1.0;
			}

			else {
				result[ m * dim + n ] = 0.0;
			}
		}
	}

	//printf( "dim: %d\n", dim );

	for ( unsigned int i = 0; i < dim; i ++ ) {
		if ( temp[ i * dim + i ] == 0.0 ) { // Checking to see if an inverse exists
			for ( unsigned int m = 0; m < dim; m ++ ) {
				for ( unsigned int n = 0; n < dim; n ++ ) {
					result[ m * dim + n ] = 0.0; // Not a valid inverse, just return a zero matrix for testing only
				}
			}

			return;
		}

		for ( unsigned int j = 0; j < dim; j ++ ) {
			if ( i != j ) {
				ratio = temp[ j * dim + i ] / temp[ i * dim + i ];

				for( unsigned int k = 0; k < dim; k ++ ) {
					temp[ j * dim + k ] = temp[ j * dim + k ] - ratio * temp[ i * dim + k ];
					result[ j * dim + k ] = result[ j * dim + k ] - ratio * result[ i * dim + k ];
				}
			}
		}
	}

	for( unsigned i = 0; i < dim; i ++ ) {
		for( unsigned j = 0; j < dim; j ++ ) {
			result[ i * dim + j ] = result[ i * dim + j ] / temp[ i * dim + i ];
			//printf( "result[ %d ]: %f\n", i * dim + j, result[ i * dim + j ] );
		}
	}

	//printf( "result[ %d ]: %f\n", dim, result[ dim - 1  ] );
	//free( temp );
}

void mul( double *result, double *mat, unsigned int mat_y, unsigned int mat_x, double multiplicator ) {
    for ( unsigned int m = 0; m < mat_y; m ++ ) {
        for ( unsigned int n = 0; n < mat_x ; n ++ ) {
            result[ m * mat_x + n ] = mat[ m * mat_x + n ] * multiplicator;
        }
    }
}

void add( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x ) {
    assert( a_x == b_x );
    assert( a_y == b_y );

    for ( unsigned int m = 0; m < a_y; m ++ ) {
        for ( unsigned int n = 0; n < a_x ; n ++ ) {
            result[ m * a_x + n ] = a[ m * a_x + n ] + b[ m * a_x + n ];
        }
    }
}

void sub( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x ) {
    assert( a_x == b_x );
    assert( a_y == b_y );

    for ( unsigned int m = 0; m < a_y; m ++ ) {
        for ( unsigned int n = 0; n < a_x ; n ++ ) {
            result[ m * a_x + n ] = a[ m * a_x + n ] - b[ m * a_x + n ];
        }
    }
}

void dot( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x ) {
    assert( a_x == b_y );

    double a_row[ a_x ];

    for ( unsigned int i = 0; i < a_x; i ++ ) {
		a_row[ i ] = 0.0;
	}

    double b_column[ b_y ];

    for ( unsigned int i = 0; i < b_y; i ++ ) {
		b_column[ i ] = 0.0;
	}

    for ( unsigned int i = 0; i < a_y * b_x; i ++ ) {
        for ( unsigned int j = 0; j < a_x; j ++ ) {
        	a_row[ j ] = a[ i / b_x * a_x + j ];
        	b_column[ j ] = b[ j * b_x + i % b_x ];
        }

        result[ i ] = vec_dot( a_row, b_column, a_x );
        //printf( "result[ %d ]: %f\n", i, result[ i ] );
    }
}

double vec_dot( double *x, double *y, unsigned int size ) {
    double result = 0.0;

    for ( unsigned int i = 0; i < size; i ++ ) {
        result += x[ i ] * y[ i ];
	}

    //printf( "vec_result: %f\n", result );

    return result;
}

void vec_mean( double *result, double *vec, unsigned int size_of_vec, unsigned int num_of_vec ) {
	for ( unsigned int j = 0; j < num_of_vec; j ++ ) {
		for ( unsigned int k = 0; k < size_of_vec; k ++ ) {
	    	result[ k ] += vec[ size_of_vec * j + k ];
		}
	}

	for ( unsigned int k = 0; k < size_of_vec; k ++ ) {
		result[ k ] = result[ k ] / num_of_vec;
	}
}

// ---------------------------------------------------------------------------
// CSR helper: map (row, col) -> index into csr->values / csr->col_idx.
// Performs a linear scan over the row's entries — fine for small block sizes.
// Returns -1 if the entry is not present (should not happen after allocation).
// ---------------------------------------------------------------------------
static int csr_find_pos( CSRMatrix *csr, int row, int col ) {
    for ( int j = csr->row_ptr[row]; j < csr->row_ptr[row + 1]; j++ ) {
        if ( csr->col_idx[j] == col )
            return j;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// build_csr_from_edges_f
//
// Two-pass CSR construction:
//   Pass 1 — count the number of non-zero entries per row.
//   Pass 2 — fill col_idx and accumulate float values.
//
// Edge contributions:
//   rows/cols [f1..f1+2] x [f1..f1+2]  <- omega_upperleft   (3x3)
//   rows/cols [f1..f1+2] x [f2..f2+2]  <- omega_upperright  (3x3)
//   rows/cols [f2..f2+2] x [f1..f1+2]  <- omega_bottomleft  (3x3)
//   rows/cols [f2..f2+2] x [f2..f2+2]  <- omega_bottomright (3x3)
//
// Anchor constraint: diagonal entries [0..2] get +1000000.0f.
//
// xi is accumulated in double.
// ---------------------------------------------------------------------------
void build_csr_from_edges_f( struct Edge *edges, unsigned int n_edges,
                              unsigned int dim, CSRMatrix *csr, double *xi ) {
    // -----------------------------------------------------------------------
    // Pass 1: count distinct non-zero columns per row.
    // Use a dense boolean scratch array.
    // -----------------------------------------------------------------------
    unsigned char *present = (unsigned char*)calloc( (size_t)dim * dim, 1 );

    // Anchor: diagonal [0..2]
    for ( int i = 0; i < 3; i++ )
        present[ i * dim + i ] = 1;

    for ( unsigned int e = 0; e < n_edges; e++ ) {
        unsigned int f1 = edges[e].t1 * 3;
        unsigned int f2 = edges[e].t2 * 3;

        for ( int r = 0; r < 3; r++ ) {
            for ( int c = 0; c < 3; c++ ) {
                present[ (f1+r) * dim + (f1+c) ] = 1;
                present[ (f1+r) * dim + (f2+c) ] = 1;
                present[ (f2+r) * dim + (f1+c) ] = 1;
                present[ (f2+r) * dim + (f2+c) ] = 1;
            }
        }
    }

    // Build row_ptr from counts
    csr->dim     = (int)dim;
    csr->row_ptr = (int*)malloc( (dim + 1) * sizeof(int) );
    csr->row_ptr[0] = 0;
    for ( unsigned int i = 0; i < dim; i++ ) {
        int cnt = 0;
        for ( unsigned int j = 0; j < dim; j++ )
            if ( present[ i * dim + j ] ) cnt++;
        csr->row_ptr[i + 1] = csr->row_ptr[i] + cnt;
    }
    csr->nnz = csr->row_ptr[dim];

    // Allocate col_idx and values
    csr->col_idx = (int*)  malloc( csr->nnz * sizeof(int) );
    csr->values  = (float*)calloc( csr->nnz,  sizeof(float) );

    // Fill col_idx in column order for each row
    for ( unsigned int i = 0; i < dim; i++ ) {
        int pos = csr->row_ptr[i];
        for ( unsigned int j = 0; j < dim; j++ ) {
            if ( present[ i * dim + j ] ) {
                csr->col_idx[pos] = (int)j;
                pos++;
            }
        }
    }

    free( present );

    // -----------------------------------------------------------------------
    // Pass 2: accumulate float values from edges.
    // -----------------------------------------------------------------------

    // Anchor: +1000000.0f on diagonal [0..2]
    for ( int i = 0; i < 3; i++ ) {
        int pos = csr_find_pos( csr, i, i );
        csr->values[pos] += 1000000.0f;
    }

    // Edge contributions
    for ( unsigned int e = 0; e < n_edges; e++ ) {
        unsigned int f1 = edges[e].t1 * 3;
        unsigned int f2 = edges[e].t2 * 3;

        // omega_upperleft  -> block (f1, f1)
        for ( int r = 0; r < 3; r++ )
            for ( int c = 0; c < 3; c++ ) {
                int pos = csr_find_pos( csr, (int)(f1+r), (int)(f1+c) );
                csr->values[pos] += (float)edges[e].omega_upperleft[r*3+c];
            }

        // omega_upperright -> block (f1, f2)
        for ( int r = 0; r < 3; r++ )
            for ( int c = 0; c < 3; c++ ) {
                int pos = csr_find_pos( csr, (int)(f1+r), (int)(f2+c) );
                csr->values[pos] += (float)edges[e].omega_upperright[r*3+c];
            }

        // omega_bottomleft -> block (f2, f1)
        for ( int r = 0; r < 3; r++ )
            for ( int c = 0; c < 3; c++ ) {
                int pos = csr_find_pos( csr, (int)(f2+r), (int)(f1+c) );
                csr->values[pos] += (float)edges[e].omega_bottomleft[r*3+c];
            }

        // omega_bottomright -> block (f2, f2)
        for ( int r = 0; r < 3; r++ )
            for ( int c = 0; c < 3; c++ ) {
                int pos = csr_find_pos( csr, (int)(f2+r), (int)(f2+c) );
                csr->values[pos] += (float)edges[e].omega_bottomright[r*3+c];
            }

        // xi accumulation stays double
        for ( int r = 0; r < 3; r++ ) {
            xi[ f1+r ] += edges[e].xi_upper[r];
            xi[ f2+r ] += edges[e].xi_bottom[r];
        }
    }
}

// ---------------------------------------------------------------------------
// free_csr
// ---------------------------------------------------------------------------
void free_csr( CSRMatrix *csr ) {
    free( csr->row_ptr );
    free( csr->col_idx );
    free( csr->values  );
    csr->row_ptr = NULL;
    csr->col_idx = NULL;
    csr->values  = NULL;
    csr->dim = 0;
    csr->nnz = 0;
}

// ---------------------------------------------------------------------------
// build_diag_inv_f
//
// Extract each 3x3 diagonal block from the CSR matrix and invert it using
// the cofactor formula (all float arithmetic).
// diag_inv must be pre-allocated to (dim/3)*9 floats.
// ---------------------------------------------------------------------------
void build_diag_inv_f( CSRMatrix *csr, float *diag_inv, unsigned int dim ) {
    unsigned int n_blocks = dim / 3;

    for ( unsigned int blk = 0; blk < n_blocks; blk++ ) {
        unsigned int base = blk * 3;

        // Extract 3x3 diagonal block from CSR
        float a[9];
        for ( int r = 0; r < 3; r++ )
            for ( int c = 0; c < 3; c++ ) {
                int pos = csr_find_pos( csr, (int)(base+r), (int)(base+c) );
                a[r*3+c] = ( pos >= 0 ) ? csr->values[pos] : 0.0f;
            }

        // Invert 3x3 block analytically via cofactor formula (float)
        float det = a[0]*(a[4]*a[8]-a[5]*a[7])
                  - a[1]*(a[3]*a[8]-a[5]*a[6])
                  + a[2]*(a[3]*a[7]-a[4]*a[6]);
        if ( det == 0.0f ) det = 1e-30f;

        float inv[9];
        inv[0] = (a[4]*a[8]-a[5]*a[7])/det;
        inv[1] = (a[2]*a[7]-a[1]*a[8])/det;
        inv[2] = (a[1]*a[5]-a[2]*a[4])/det;
        inv[3] = (a[5]*a[6]-a[3]*a[8])/det;
        inv[4] = (a[0]*a[8]-a[2]*a[6])/det;
        inv[5] = (a[2]*a[3]-a[0]*a[5])/det;
        inv[6] = (a[3]*a[7]-a[4]*a[6])/det;
        inv[7] = (a[1]*a[6]-a[0]*a[7])/det;
        inv[8] = (a[0]*a[4]-a[1]*a[3])/det;

        for ( int k = 0; k < 9; k++ )
            diag_inv[ blk*9 + k ] = inv[k];
    }
}
