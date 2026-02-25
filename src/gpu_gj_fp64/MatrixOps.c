#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "MatrixOps.h"

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


	/*for( unsigned i = 0; i < dim; i ++ ) {
		for( unsigned j = 0; j < dim; j ++ ) {
			result[ i * dim + j ] = result[ i * dim + j ] / temp[ i * dim + i ];
			//printf( "result[ %d ]: %f\n", i * dim + j, result[ i * dim + j ] );
		}
	}*/

	for(unsigned int i = 0; i < dim * dim; i++)
	{
		unsigned int j = i / dim;
		result[i] = result[i] / temp[j * dim + j];
		//printf("result[ %d ]: %f\n", i, result[i]);
	}
	
	//printf( "result[ %d ]: %f\n", dim, result[ dim - 1  ] );
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
