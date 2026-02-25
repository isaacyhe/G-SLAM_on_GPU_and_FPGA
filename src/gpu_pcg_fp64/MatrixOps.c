#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "MatrixOps.h"
#include "Edge.h"

double vec_norm( double *mat, unsigned int mat_y, unsigned int mat_x ) {
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
	//double temp[ dim * dim ];
	double *temp = malloc ( sizeof( double ) * dim * dim ); // dim * dim

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

	for(unsigned int i = 0; i < dim * dim; i++)
	{
		unsigned int j = i / dim;
		result[i] = result[i] / temp[j * dim + j];
	}

	free( temp );
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

/*
 * build_csr_from_edges:
 *
 * Constructs a CSR (Compressed Sparse Row) representation of the information
 * matrix Omega directly from the edge list, without forming the dense matrix.
 * Also accumulates the RHS vector xi.
 *
 * Steps:
 *   1. Use a char marker array (dim x dim) to find all unique (row, col) pairs
 *      contributed by each edge's four 3x3 sub-blocks, plus the anchor entries
 *      [0,0], [1,1], [2,2] for the first pose.
 *   2. Build row_ptr (count non-zeros per row, then prefix-sum).
 *   3. Build col_idx (sorted within each row by construction).
 *   4. Zero-initialize values array.
 *   5. Add anchor 1e6 to diagonal entries [0..2].
 *   6. Accumulate all edge contributions into values using linear search
 *      within each CSR row.
 *   7. Accumulate xi contributions from each edge.
 */
void build_csr_from_edges(struct Edge *edges, unsigned int n_edges,
                           unsigned int dim, CSRMatrix *csr, double *xi)
{
    /* --- Step 1: mark unique (row, col) pairs --- */
    char *marker = (char *)calloc((size_t)dim * dim, sizeof(char));

    /* Anchor entries: diagonal [0,0], [1,1], [2,2] */
    marker[0 * dim + 0] = 1;
    marker[1 * dim + 1] = 1;
    marker[2 * dim + 2] = 1;

    for (unsigned int e = 0; e < n_edges; e++) {
        unsigned int f1 = edges[e].t1 * 3;
        unsigned int f2 = edges[e].t2 * 3;

        /* omega_upperleft: rows f1..f1+2, cols f1..f1+2 */
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                marker[(f1 + r) * dim + (f1 + c)] = 1;

        /* omega_upperright: rows f1..f1+2, cols f2..f2+2 */
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                marker[(f1 + r) * dim + (f2 + c)] = 1;

        /* omega_bottomleft: rows f2..f2+2, cols f1..f1+2 */
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                marker[(f2 + r) * dim + (f1 + c)] = 1;

        /* omega_bottomright: rows f2..f2+2, cols f2..f2+2 */
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                marker[(f2 + r) * dim + (f2 + c)] = 1;
    }

    /* --- Step 2: count nnz per row and build row_ptr --- */
    int *row_ptr = (int *)calloc(dim + 1, sizeof(int));
    for (unsigned int row = 0; row < dim; row++) {
        int cnt = 0;
        for (unsigned int col = 0; col < dim; col++)
            if (marker[row * dim + col]) cnt++;
        row_ptr[row + 1] = cnt;
    }
    /* prefix sum */
    for (unsigned int row = 0; row < dim; row++)
        row_ptr[row + 1] += row_ptr[row];

    int nnz = row_ptr[dim];

    /* --- Step 3: fill col_idx (columns are naturally sorted 0..dim-1) --- */
    int *col_idx = (int *)malloc(nnz * sizeof(int));
    {
        int pos = 0;
        for (unsigned int row = 0; row < dim; row++)
            for (unsigned int col = 0; col < dim; col++)
                if (marker[row * dim + col])
                    col_idx[pos++] = (int)col;
    }

    free(marker);

    /* --- Step 4: zero-initialize values --- */
    double *values = (double *)calloc(nnz, sizeof(double));

    /* --- Step 5: anchor: add 1e6 to diagonal [0..2] --- */
    for (int d = 0; d < 3; d++) {
        /* find position of col d in row d */
        for (int k = row_ptr[d]; k < row_ptr[d + 1]; k++) {
            if (col_idx[k] == d) {
                values[k] += 1e6;
                break;
            }
        }
    }

    /* --- Step 6 & 7: accumulate edge contributions --- */
    /* Helper: find the CSR index for (row, col) using linear search */
    for (unsigned int e = 0; e < n_edges; e++) {
        unsigned int f1 = edges[e].t1 * 3;
        unsigned int f2 = edges[e].t2 * 3;

        /* omega_upperleft: rows f1..f1+2, cols f1..f1+2 */
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                int row = (int)(f1 + r);
                int col = (int)(f1 + c);
                for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
                    if (col_idx[k] == col) {
                        values[k] += edges[e].omega_upperleft[r * 3 + c];
                        break;
                    }
                }
            }
        }

        /* omega_upperright: rows f1..f1+2, cols f2..f2+2 */
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                int row = (int)(f1 + r);
                int col = (int)(f2 + c);
                for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
                    if (col_idx[k] == col) {
                        values[k] += edges[e].omega_upperright[r * 3 + c];
                        break;
                    }
                }
            }
        }

        /* omega_bottomleft: rows f2..f2+2, cols f1..f1+2 */
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                int row = (int)(f2 + r);
                int col = (int)(f1 + c);
                for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
                    if (col_idx[k] == col) {
                        values[k] += edges[e].omega_bottomleft[r * 3 + c];
                        break;
                    }
                }
            }
        }

        /* omega_bottomright: rows f2..f2+2, cols f2..f2+2 */
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                int row = (int)(f2 + r);
                int col = (int)(f2 + c);
                for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
                    if (col_idx[k] == col) {
                        values[k] += edges[e].omega_bottomright[r * 3 + c];
                        break;
                    }
                }
            }
        }

        /* xi accumulation */
        xi[f1 + 0] += edges[e].xi_upper[0];
        xi[f1 + 1] += edges[e].xi_upper[1];
        xi[f1 + 2] += edges[e].xi_upper[2];

        xi[f2 + 0] += edges[e].xi_bottom[0];
        xi[f2 + 1] += edges[e].xi_bottom[1];
        xi[f2 + 2] += edges[e].xi_bottom[2];
    }

    csr->row_ptr = row_ptr;
    csr->col_idx = col_idx;
    csr->values  = values;
    csr->dim     = (int)dim;
    csr->nnz     = nnz;
}

void free_csr(CSRMatrix *csr)
{
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    csr->row_ptr = NULL;
    csr->col_idx = NULL;
    csr->values  = NULL;
    csr->dim     = 0;
    csr->nnz     = 0;
}

/*
 * build_diag_inv:
 *
 * Extracts the 3x3 diagonal blocks from the CSR matrix and inverts each
 * analytically using the cofactor (adjugate) formula.
 *
 * Each 3x3 block occupying rows [blk*3 .. blk*3+2] and columns
 * [blk*3 .. blk*3+2] is extracted from the CSR values array, inverted,
 * and stored in diag_inv[blk*9 .. blk*9+8] in row-major order.
 */
void build_diag_inv(CSRMatrix *csr, double *diag_inv, unsigned int dim)
{
    int n_blocks = (int)dim / 3;

    for (int blk = 0; blk < n_blocks; blk++) {
        int base_row = blk * 3;
        int base_col = blk * 3;

        /* Extract 3x3 block B from CSR */
        double B[9] = { 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0 };

        for (int r = 0; r < 3; r++) {
            int row = base_row + r;
            for (int k = csr->row_ptr[row]; k < csr->row_ptr[row + 1]; k++) {
                int col = csr->col_idx[k];
                if (col >= base_col && col < base_col + 3) {
                    B[r * 3 + (col - base_col)] = csr->values[k];
                }
            }
        }

        /* Cofactor matrix (transpose of adjugate) and determinant */
        double det =
              B[0] * (B[4]*B[8] - B[5]*B[7])
            - B[1] * (B[3]*B[8] - B[5]*B[6])
            + B[2] * (B[3]*B[7] - B[4]*B[6]);

        double *inv_blk = diag_inv + blk * 9;

        if (det == 0.0) {
            /* Singular block: store identity as a safe fallback */
            for (int i = 0; i < 9; i++) inv_blk[i] = 0.0;
            inv_blk[0] = 1.0;
            inv_blk[4] = 1.0;
            inv_blk[8] = 1.0;
            continue;
        }

        double inv_det = 1.0 / det;

        inv_blk[0] =  (B[4]*B[8] - B[5]*B[7]) * inv_det;
        inv_blk[1] = -(B[1]*B[8] - B[2]*B[7]) * inv_det;
        inv_blk[2] =  (B[1]*B[5] - B[2]*B[4]) * inv_det;
        inv_blk[3] = -(B[3]*B[8] - B[5]*B[6]) * inv_det;
        inv_blk[4] =  (B[0]*B[8] - B[2]*B[6]) * inv_det;
        inv_blk[5] = -(B[0]*B[5] - B[2]*B[3]) * inv_det;
        inv_blk[6] =  (B[3]*B[7] - B[4]*B[6]) * inv_det;
        inv_blk[7] = -(B[0]*B[7] - B[1]*B[6]) * inv_det;
        inv_blk[8] =  (B[0]*B[4] - B[1]*B[3]) * inv_det;
    }
}
