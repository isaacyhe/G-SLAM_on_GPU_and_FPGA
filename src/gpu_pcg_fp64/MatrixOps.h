#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#ifdef __cplusplus
extern "C"{
#endif

#include "Edge.h"

double vec_norm( double *mat, unsigned int mat_y, unsigned int mat_x );
void tra( double *result, double *mat, unsigned int mat_y, unsigned int mat_x );
void inv( double *result, double *mat, unsigned int dim );
void mul( double *result, double *mat, unsigned int mat_y, unsigned int mat_x, double multiplicator );
void add( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );
void sub( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );
void dot( double *result, double *a, unsigned int a_y, unsigned int a_x, double *b, unsigned int b_y, unsigned int b_x );

double vec_dot( double *x, double *y, unsigned int size );
void vec_mean( double *result, double *vec, unsigned int size_of_vec, unsigned int num_of_vec );

typedef struct {
    int    *row_ptr;
    int    *col_idx;
    double *values;
    int     dim;
    int     nnz;
} CSRMatrix;

void build_csr_from_edges(struct Edge *edges, unsigned int n_edges,
                           unsigned int dim, CSRMatrix *csr, double *xi);
void free_csr(CSRMatrix *csr);
void build_diag_inv(CSRMatrix *csr, double *diag_inv, unsigned int dim);

#ifdef __cplusplus
}
#endif
#endif
