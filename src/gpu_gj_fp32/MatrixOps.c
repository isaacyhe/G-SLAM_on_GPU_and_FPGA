#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

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

	/*for ( unsigned int i = 0; i < dim; i ++ ) {
		if ( temp[ i * dim + i] == 0.0f ) { // Checking to see if an inverse exists
			for ( unsigned int m = 0; m < dim; m ++ ) {
				for ( unsigned int n = 0; n < dim; n ++ ) {
					result[ m * dim + n ] = 0.0f; // Not a valid inverse, just return a zero matrix for testing only
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
	}*/
 
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
        
            //printf( "result[ %d ]: %f\n", i * dim + j, result[ i * dim + j ] );
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
        //printf( "result[ %d ]: %f\n", i, result[ i ] );
    }
}

float vec_dot( float *x, float *y, unsigned int size ) {
    float result = 0.0f;

    for ( unsigned int i = 0; i < size; i ++ ) {
        result += x[ i ] * y[ i ];
	}

    //printf( "vec_result: %f\n", result );

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
