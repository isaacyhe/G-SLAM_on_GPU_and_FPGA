#include "U.h"
#include "HAT_X.h"
#include "Z.h"
#include "Edge.h"
#include "LandmarkKeysZList.h"
#include "MatrixOps.h"

unsigned int read_data( char *file_name, double *delta, struct U *us, unsigned int *size_of_us, struct HAT_X *hat_xs, unsigned int *size_of_hat_xs, struct Z *zlist, unsigned int *size_of_zlist );

void make_edges( struct HAT_X *hat_xs, unsigned int size_of_hat_xs, struct Z *zlist, unsigned int size_of_zlist, struct Edge *edges, unsigned int *size_of_edges, struct LandmarkKeysZList *landmark_keys_zlist, unsigned int *size_of_landmark_keys_zlist );

void add_edge( struct Edge edge, double *Omega, unsigned int Omega_y, unsigned int Omega_x, double *xi );

void state_transition( double *result, double nu, double omega, unsigned int delta, struct HAT_X pose );

unsigned int combination( unsigned int n, unsigned int r );

void spmv_parallel( double *q, CSRMatrix *csr, double *p );
