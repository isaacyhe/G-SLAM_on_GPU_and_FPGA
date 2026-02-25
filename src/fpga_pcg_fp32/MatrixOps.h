#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "Edge.h"

double norm( double *mat, unsigned int mat_y, unsigned int mat_x );
void tra( double *result, double *mat, unsigned int mat_y, unsigned int mat_x );
void inv( double *result, double *mat, unsigned int dim );
void mul( double *result, double *mat, unsigned int mat_y, unsigned int mat_x, double multiplicator );
void add( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );
void sub( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );
void dot( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );

double vec_dot( double *x, double *y, unsigned int size );
void vec_mean( double *result, double *vec, unsigned int size_of_vec, unsigned int num_of_vec );

// Sparse CSR matrix with float values
typedef struct {
    int   *row_ptr;
    int   *col_idx;
    float *values;
    int    dim;
    int    nnz;
} CSRMatrix;

// Build a CSR matrix (float values) from edges; also fills xi in double.
// The anchor constraint (1000000.0f on diagonal entries [0..2]) is added here.
void build_csr_from_edges_f( struct Edge *edges, unsigned int n_edges,
                              unsigned int dim, CSRMatrix *csr, double *xi );

// Free memory owned by a CSRMatrix
void free_csr( CSRMatrix *csr );

// Build block-diagonal preconditioner inverse (float).
// diag_inv must be pre-allocated to (dim/3)*9 floats.
void build_diag_inv_f( CSRMatrix *csr, float *diag_inv, unsigned int dim );

#endif
