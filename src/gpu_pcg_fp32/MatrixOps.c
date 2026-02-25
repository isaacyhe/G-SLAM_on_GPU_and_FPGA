#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "Edge.h"
#include "MatrixOps.h"

float vec_norm( float *mat, unsigned int mat_y, unsigned int mat_x ) {
    float result = 0.0f;

    for ( unsigned int m = 0; m < mat_y; m ++ ) {
        for ( unsigned int n = 0; n < mat_x; n ++ ) {
			//printf( "mat[ n * mat_x + m ]: %f\n", mat[ n * mat_x + m ] );

            result += powf( mat[ n * mat_x + m ], 2.0f );
        }
    }

    return sqrtf( result );
}

void tra( float *result, float *mat, unsigned int mat_y, unsigned int mat_x ) {
    unsigned int result_y = mat_x;
    unsigned int result_x = mat_y;

    for ( unsigned int m = 0; m < result_y; m ++ ) {
        for ( unsigned int n = 0; n < result_x; n ++ ) {
            result[ m * result_x + n ] = mat[ n * mat_x + m ];
        }
    }
}

void inv( float *result, float *mat, unsigned int dim ) {
	float *temp = malloc ( sizeof( float ) * dim * dim ); // dim * dim

	for ( unsigned int i = 0; i < dim * dim; i ++ ) {
		temp[ i ] = 0.0f;
	}

	float ratio = 0.0f;

	for ( unsigned int m = 0; m < dim; m ++ ) {
		for ( unsigned int n = 0; n < dim; n ++ ) {
			temp[ m * dim + n ] = mat[ m * dim + n ];

			if( m == n ) {
				result[ m * dim + n ] = 1.0f;
			}

			else {
				result[ m * dim + n ] = 0.0f;
			}
		}
	}

   for(unsigned int i = 0; i < dim * dim; i ++)
   {
       unsigned int j = i % dim;
       unsigned int h = i / dim;

       if(temp[h * dim + h] == 0.0f)
       {
           for(unsigned int m = 0; m < dim * dim; m++)
           {
               result[m] = 0.0f;
           }

           return;
       }

       if(j != h)
       {
           ratio = temp[j * dim + h] / temp[h * dim + h];

           for(unsigned int k = 0; k < dim; k ++)
           {
               temp[j * dim + k] = temp[j * dim + k] - ratio * temp[h * dim + k];
               result[j * dim + k] = result[j * dim + k] - ratio * result[h * dim + k];
           }

       }
   }

	for( unsigned i = 0; i < dim; i ++ ) {
		for( unsigned j = 0; j < dim; j ++ ) {
			result[ i * dim + j ] = result[ i * dim + j ] / temp[ i * dim + i ];
		}
	}

	free( temp );
}

void mul( float *result, float *mat, unsigned int mat_y, unsigned int mat_x, float multiplicator ) {
    for ( unsigned int m = 0; m < mat_y; m ++ ) {
        for ( unsigned int n = 0; n < mat_x ; n ++ ) {
            result[ m * mat_x + n ] = mat[ m * mat_x + n ] * multiplicator;
        }
    }
}

void add( float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x ) {
    assert( a_x == b_x );
    assert( a_y == b_y );

    for ( unsigned int m = 0; m < a_y; m ++ ) {
        for ( unsigned int n = 0; n < a_x ; n ++ ) {
            result[ m * a_x + n ] = a[ m * a_x + n ] + b[ m * a_x + n ];
        }
    }
}

void sub( float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x ) {
    assert( a_x == b_x );
    assert( a_y == b_y );

    for ( unsigned int m = 0; m < a_y; m ++ ) {
        for ( unsigned int n = 0; n < a_x ; n ++ ) {
            result[ m * a_x + n ] = a[ m * a_x + n ] - b[ m * a_x + n ];
        }
    }
}

void dot( float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x ) {
    assert( a_x == b_y );

    float a_row[ a_x ];

    for ( unsigned int i = 0; i < a_x; i ++ ) {
		a_row[ i ] = 0.0f;
	}

    float b_column[ b_y ];

    for ( unsigned int i = 0; i < b_y; i ++ ) {
		b_column[ i ] = 0.0f;
	}

    for ( unsigned int i = 0; i < a_y * b_x; i ++ ) {
        for ( unsigned int j = 0; j < a_x; j ++ ) {
        	a_row[ j ] = a[ i / b_x * a_x + j ];
        	b_column[ j ] = b[ j * b_x + i % b_x ];
        }

        result[ i ] = vec_dot( a_row, b_column, a_x );
    }
}

float vec_dot( float *x, float *y, unsigned int size ) {
    float result = 0.0f;

    for ( unsigned int i = 0; i < size; i ++ ) {
        result += x[ i ] * y[ i ];
	}

    return result;
}

void vec_mean( float *result, float *vec, unsigned int size_of_vec, unsigned int num_of_vec ) {
	for ( unsigned int j = 0; j < num_of_vec; j ++ ) {
		for ( unsigned int k = 0; k < size_of_vec; k ++ ) {
	    	result[ k ] += vec[ size_of_vec * j + k ];
		}
	}

	for ( unsigned int k = 0; k < size_of_vec; k ++ ) {
		result[ k ] = result[ k ] / num_of_vec;
	}
}

void build_csr_from_edges_f(struct Edge *edges, unsigned int n_edges,
                              unsigned int dim, CSRMatrix *csr, double *xi) {
    // Step 1: use a boolean marker to identify unique (row, col) non-zero entries.
    // Each edge touches 4 blocks of 3x3 entries; anchor adds 3 diagonal entries.
    char *marker = (char*)calloc( (size_t)dim * dim, sizeof(char) );

    // anchor: diagonal entries [0,0], [1,1], [2,2]
    marker[0*(int)dim+0] = 1;
    marker[1*(int)dim+1] = 1;
    marker[2*(int)dim+2] = 1;

    for ( unsigned int e = 0; e < n_edges; e++ ) {
        unsigned int f1 = edges[e].t1 * 3;
        unsigned int f2 = edges[e].t2 * 3;
        for ( int r = 0; r < 3; r++ ) {
            for ( int c = 0; c < 3; c++ ) {
                marker[(f1+r)*(int)dim + (f1+c)] = 1;
                marker[(f1+r)*(int)dim + (f2+c)] = 1;
                marker[(f2+r)*(int)dim + (f1+c)] = 1;
                marker[(f2+r)*(int)dim + (f2+c)] = 1;
            }
        }
    }

    // Step 2: build row_ptr
    csr->dim = (int)dim;
    csr->row_ptr = (int*)malloc( (dim + 1) * sizeof(int) );
    csr->row_ptr[0] = 0;
    for ( unsigned int i = 0; i < dim; i++ ) {
        int count = 0;
        for ( unsigned int j = 0; j < dim; j++ )
            if ( marker[i * dim + j] ) count++;
        csr->row_ptr[i+1] = csr->row_ptr[i] + count;
    }
    csr->nnz = csr->row_ptr[dim];

    // Step 3: build col_idx and zero-initialize values
    csr->col_idx = (int*)malloc( csr->nnz * sizeof(int) );
    csr->values  = (float*)calloc( csr->nnz, sizeof(float) );

    for ( unsigned int i = 0; i < dim; i++ ) {
        int pos = csr->row_ptr[i];
        for ( unsigned int j = 0; j < dim; j++ ) {
            if ( marker[i * dim + j] ) {
                csr->col_idx[pos++] = (int)j;
            }
        }
    }
    free( marker );

    // Step 4: add anchor — 1e6 on diagonal entries [0,0], [1,1], [2,2]
    for ( int d = 0; d < 3; d++ ) {
        for ( int p = csr->row_ptr[d]; p < csr->row_ptr[d+1]; p++ ) {
            if ( csr->col_idx[p] == d ) {
                csr->values[p] += 1000000.0f;
                break;
            }
        }
    }

    // Step 5: accumulate edge contributions into CSR values (float) and xi (double)
    for ( unsigned int e = 0; e < n_edges; e++ ) {
        unsigned int f1 = edges[e].t1 * 3;
        unsigned int f2 = edges[e].t2 * 3;

        for ( int r = 0; r < 3; r++ ) {
            for ( int c = 0; c < 3; c++ ) {
                // omega_upperleft  -> block (f1+r, f1+c)
                for ( int p = csr->row_ptr[f1+r]; p < csr->row_ptr[f1+r+1]; p++ ) {
                    if ( csr->col_idx[p] == (int)(f1+c) ) {
                        csr->values[p] += (float)edges[e].omega_upperleft[r*3+c];
                        break;
                    }
                }
                // omega_upperright -> block (f1+r, f2+c)
                for ( int p = csr->row_ptr[f1+r]; p < csr->row_ptr[f1+r+1]; p++ ) {
                    if ( csr->col_idx[p] == (int)(f2+c) ) {
                        csr->values[p] += (float)edges[e].omega_upperright[r*3+c];
                        break;
                    }
                }
                // omega_bottomleft -> block (f2+r, f1+c)
                for ( int p = csr->row_ptr[f2+r]; p < csr->row_ptr[f2+r+1]; p++ ) {
                    if ( csr->col_idx[p] == (int)(f1+c) ) {
                        csr->values[p] += (float)edges[e].omega_bottomleft[r*3+c];
                        break;
                    }
                }
                // omega_bottomright -> block (f2+r, f2+c)
                for ( int p = csr->row_ptr[f2+r]; p < csr->row_ptr[f2+r+1]; p++ ) {
                    if ( csr->col_idx[p] == (int)(f2+c) ) {
                        csr->values[p] += (float)edges[e].omega_bottomright[r*3+c];
                        break;
                    }
                }
            }
            xi[f1+r] += edges[e].xi_upper[r];
            xi[f2+r] += edges[e].xi_bottom[r];
        }
    }
}

void free_csr(CSRMatrix *csr) {
    free( csr->row_ptr );
    free( csr->col_idx );
    free( csr->values );
}

void build_diag_inv_f(CSRMatrix *csr, float *diag_inv, unsigned int dim) {
    unsigned int n_blocks = dim / 3;

    for ( unsigned int blk = 0; blk < n_blocks; blk++ ) {
        unsigned int base = blk * 3;
        float a[9] = { 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f };
        // Extract diagonal 3x3 block entries from the float CSR
        for ( int r = 0; r < 3; r++ ) {
            for ( int p = csr->row_ptr[base+r]; p < csr->row_ptr[base+r+1]; p++ ) {
                int c = csr->col_idx[p] - (int)base;
                if ( c >= 0 && c < 3 )
                    a[r*3+c] = csr->values[p];
            }
        }
        // 3x3 analytical inverse via cofactor formula (float arithmetic)
        float det = a[0]*(a[4]*a[8]-a[5]*a[7])
                  - a[1]*(a[3]*a[8]-a[5]*a[6])
                  + a[2]*(a[3]*a[7]-a[4]*a[6]);
        if ( det == 0.0f ) det = 1e-30f;
        float *inv = diag_inv + blk * 9;
        inv[0] = (a[4]*a[8]-a[5]*a[7]) / det;
        inv[1] = (a[2]*a[7]-a[1]*a[8]) / det;
        inv[2] = (a[1]*a[5]-a[2]*a[4]) / det;
        inv[3] = (a[5]*a[6]-a[3]*a[8]) / det;
        inv[4] = (a[0]*a[8]-a[2]*a[6]) / det;
        inv[5] = (a[2]*a[3]-a[0]*a[5]) / det;
        inv[6] = (a[3]*a[7]-a[4]*a[6]) / det;
        inv[7] = (a[1]*a[6]-a[0]*a[7]) / det;
        inv[8] = (a[0]*a[4]-a[1]*a[3]) / det;
    }
}
