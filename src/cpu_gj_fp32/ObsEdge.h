#ifndef OBSEDGE_H
#define OBSEDGE_H

#include "Edge.h"
#include "U.h"
#include "HAT_X.h"
#include "Z.h"

struct Edge* ObsEdge_create( struct Edge *obs_edge_self, struct Z z1, struct Z z2, struct HAT_X *hat_xs, double *snr ); // The constructor

#endif
