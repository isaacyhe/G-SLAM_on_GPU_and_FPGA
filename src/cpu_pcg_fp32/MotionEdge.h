#ifndef MOTIONEDGE_H
#define MOTIONEDGE_H


#include "Edge.h"
#include "U.h"
#include "HAT_X.h"
#include "Z.h"

struct Edge *MotionEdge_create ( struct Edge *motion_edge_self, unsigned int t1, unsigned int t2, struct HAT_X *hat_xs, struct U *us, double delta, double lambda, double *mns ); // The constructor

#endif
