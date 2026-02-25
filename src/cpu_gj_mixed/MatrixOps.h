#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

double norm( double *mat, unsigned int mat_y, unsigned int mat_x );
void tra( double *result, double *mat, unsigned int mat_y, unsigned int mat_x );
void inv( double *result, double *mat, unsigned int dim );
void mul( double *result, double *mat, unsigned int mat_y, unsigned int mat_x, double multiplicator );
void add( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );
void sub( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );
void dot( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );

double vec_dot( double *x, double *y, unsigned int size );
void vec_mean( double *result, double *vec, unsigned int size_of_vec, unsigned int num_of_vec );

// Mixed-precision GJ: invert Omega in FP32, cast result back to FP64
void inv_float( float *result, float *mat, unsigned int dim );

#endif
