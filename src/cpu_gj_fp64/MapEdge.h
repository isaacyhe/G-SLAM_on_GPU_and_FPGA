#ifndef MAPEDGE_H
#define MAPEDGE_H

#include "HAT_X.h"
#include "Z.h"

struct MapEdge {
    struct HAT_X x;
    struct Z z;
    double m[ 2 * 1 ], Omega[ 2 * 2 ], xi[ 2 * 1 ];
};

struct MapEdge *MapEdge_create ( struct MapEdge *map_edge_self, struct Z z, struct Z head_z, struct HAT_X *hat_xs, double *snr ); // The constructor

#endif
