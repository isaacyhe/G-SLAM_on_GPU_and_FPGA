#include <stdlib.h>
#include <math.h>

#include "MatrixOps.h"
#include "MapEdge.h"

struct MapEdge* MapEdge_create ( struct MapEdge *map_edge_self, struct Z z, struct Z head_z, struct HAT_X *hat_xs, double *snr ) { // The constructor
    //printf( "z[0]: %d\n", z.step );
    //printf( "head_z[0]: %d\n", head_z.step  );

    map_edge_self->x = hat_xs[ z.step ];
    map_edge_self->z = z;

    double m1[ 2 * 1 ] = {
	    map_edge_self->x.hat_x[0],
	    map_edge_self->x.hat_x[1]
    };

    double m2[ 2 * 1 ] = {
		z.z[0] * cos( map_edge_self->x.hat_x[2] + z.z[1] ),
		z.z[0] * sin( map_edge_self->x.hat_x[2] + z.z[1] )
    };

    double additionOfMM[ 2 * 1 ] = { 0.0 };
    add( additionOfMM, m1, 2, 1, m2, 2, 1 );

    //printf( "m1: %f %f\n", m1[0], m1[1] );
    //printf( "m2: %f %f\n", m2[0], m2[1] );
    //printf( "additionOfMM: %f %f\n", additionOfMM[0], additionOfMM[1] );

    for( unsigned int i = 0; i < 2 * 1; i ++ ) {
		map_edge_self->m[ i ] = additionOfMM[ i ];
    }

    double Q1[ 2 * 2 ] = {
			    map_edge_self->z.z[0] * pow( snr[0], 2 ) , 0.0,
			    0.0, pow( snr[1], 2 )
    };

    double s1 = sin( map_edge_self->x.hat_x[2] + map_edge_self->z.z[1] );
    double c1 = cos( map_edge_self->x.hat_x[2] + map_edge_self->z.z[1] );

    double R[ 2 * 2 ] = {
	    -c1, map_edge_self->z.z[0] * s1,
	    -s1, -map_edge_self->z.z[0] * c1
    };

    double dot_productOfRQ[ 2 * 2 ] = { 0.0 };
    dot( dot_productOfRQ, R, 2, 2, Q1, 2, 2 );

    double transposeOfR[ 2 * 2 ] = { 0.0 };
    tra(  transposeOfR, R, 2, 2 );

    double dot_productOfDPtR[ 2 * 2 ] = { 0.0 };
    dot( dot_productOfDPtR, dot_productOfRQ, 2, 2, transposeOfR, 2, 2 );

	for( unsigned int i = 0; i < 2 * 2; i ++ ) {
		map_edge_self->Omega[ i ] = dot_productOfDPtR[ i ];
    }

    double dot_productOfmOmm[ 2 * 1 ] = { 0.0 };
    dot( dot_productOfmOmm, map_edge_self->Omega, 2, 2, map_edge_self->m, 2, 1 );

    for( unsigned int i = 0; i < 2 * 1; i ++ ) {
		map_edge_self->xi[ i ] = dot_productOfmOmm[ i ];
    }

    return map_edge_self;
}

void MapEdge_destroy ( struct MapEdge *map_edge ) { // The destructor
    if ( map_edge )
        free( map_edge );
}
