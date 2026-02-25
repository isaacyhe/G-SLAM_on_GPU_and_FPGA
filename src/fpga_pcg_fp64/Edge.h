#ifndef EDGE_H
#define EDGE_H

#include "U.h"
#include "HAT_X.h"
#include "Z.h"

struct Edge {
    unsigned int t1, t2;
    struct HAT_X hat_x1, hat_x2;
    struct Z z1, z2;
    double Omega[ 3 * 3 ], omega_upperleft[ 3 * 3 ], omega_upperright[ 3 * 3 ], omega_bottomleft[ 3 * 3 ], omega_bottomright[ 3 * 3 ], xi_upper[ 3 * 1 ], xi_bottom[ 3 * 1 ];

    double lambda, hat_e[ 2 * 1 ];	// Needed when calculating cost
};

#endif
