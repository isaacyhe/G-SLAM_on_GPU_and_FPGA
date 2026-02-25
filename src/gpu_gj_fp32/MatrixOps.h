#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#ifdef __cplusplus
extern "C"{
#endif

float vec_norm( float *mat, unsigned int mat_y, unsigned int mat_x );
void tra( float *result, float *mat, unsigned int mat_y, unsigned int mat_x );
void inv( float *result, float *mat, unsigned int dim );
void mul( float *result, float *mat, unsigned int mat_y, unsigned int mat_x, float multiplicator );
void add( float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x );
void sub( float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x );
void dot( float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x );

float vec_dot( float *x, float *y, unsigned int size );
void vec_mean( float *result, float *vec, unsigned int size_of_vec, unsigned int num_of_vec );


#ifdef __cplusplus
}
#endif
#endif
