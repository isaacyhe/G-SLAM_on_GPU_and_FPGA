#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "MatrixOps.h"
#include "ObsEdge.h"
#include "GSLAM.h"

struct Edge* ObsEdge_create( struct Edge *obs_edge_self, struct Z z1, struct Z z2, struct HAT_X *hat_xs, double *snr ) {
    assert( z1.landmark_id == z2.landmark_id );

    obs_edge_self->t1 = z1.step;
    obs_edge_self->t2 = z2.step;

    obs_edge_self->hat_x1 = hat_xs[ obs_edge_self->t1 ];
    obs_edge_self->hat_x2 = hat_xs[ obs_edge_self->t2 ];

    obs_edge_self->z1 = z1;
    obs_edge_self->z2 = z2;

    double s1 = sin( obs_edge_self->hat_x1.hat_x[2] + obs_edge_self->z1.z[1] );
    double c1 = cos( obs_edge_self->hat_x1.hat_x[2] + obs_edge_self->z1.z[1] );
    double s2 = sin( obs_edge_self->hat_x2.hat_x[2] + obs_edge_self->z2.z[1] );
    double c2 = cos( obs_edge_self->hat_x2.hat_x[2] + obs_edge_self->z2.z[1] );

    double hat_e[ 2 * 1 ] = {
        obs_edge_self->hat_x2.hat_x[0] - obs_edge_self->hat_x1.hat_x[0] + obs_edge_self->z2.z[0] * c2 - obs_edge_self->z1.z[0] * c1,
        obs_edge_self->hat_x2.hat_x[1] - obs_edge_self->hat_x1.hat_x[1] + obs_edge_self->z2.z[0] * s2 - obs_edge_self->z1.z[0] * s1
    };

    for( unsigned int i = 0; i < 2 * 1; i ++ ) {
		 obs_edge_self->hat_e[ i ] = hat_e[ i ];
    }

    double Q1[ 2 * 2 ] = {
        pow( obs_edge_self->z1.z[0] * snr[0], 2.0 ), 0.0,
        0.0, pow( snr[1], 2.0 )
    };

    double R1[ 2 * 2 ] = {
        -c1, obs_edge_self->z1.z[0] * s1,
        -s1, -obs_edge_self->z1.z[0] * c1
    };

    double Q2[ 2 * 2 ] = {
        pow( obs_edge_self->z2.z[0] * snr[0], 2.0 ), 0.0,
        0.0, pow( snr[1], 2.0 )
    };

    double R2[ 2 * 2 ] = {
        c2, -obs_edge_self->z2.z[0] * s2,
        s2, obs_edge_self->z2.z[0] * c2
    };

    double transposeOFR1[ 2 * 2 ] = { 0.0 };
    tra( transposeOFR1, R1, 2, 2 );

    double dot_productOFR1Q1[ 2 * 2 ] = { 0.0 };
    dot( dot_productOFR1Q1 , R1, 2, 2, Q1, 2, 2 );

    double dot_productOFR1Q1tR1[ 2 * 2 ] = { 0.0 };
    dot( dot_productOFR1Q1tR1, dot_productOFR1Q1, 2, 2, transposeOFR1, 2, 2 );

    double transposeOFR2[ 2 * 2 ] = { 0.0 };
    tra( transposeOFR2, R2, 2, 2 );

    double dot_productOFR2Q2[ 2 * 2 ] = { 0.0 };
    dot( dot_productOFR2Q2 , R2, 2, 2, Q2, 2, 2 );

    double dot_productOFR2Q2tR2[ 2 * 2 ] = { 0.0 };
    dot( dot_productOFR2Q2tR2, dot_productOFR2Q2, 2, 2, transposeOFR2, 2, 2 );

    double Sigma[ 2 * 2 ] = { 0.0 };
    add( Sigma, dot_productOFR1Q1tR1, 2, 2, dot_productOFR2Q2tR2, 2, 2 );

    double Omega[ 2 * 2 ] = { 0.0 };

    //printf( "Sigma: " );
	//for ( unsigned int i = 0; i < 4; i ++ ) {
	//	printf( "%f ", Sigma[ i ] );
	//}
	//printf( "\n" );

    inv( Omega, Sigma, 2 );

    //printf( "Omega: " );
	//for ( unsigned int i = 0; i < 4; i ++ ) {
	//	printf( "%f ", Omega[ i ] );
	//}
	//printf( "\n" );

    double B1[ 2 * 3 ] = {
        -1.0, -0.0, obs_edge_self->z1.z[0] * s1,
        -0.0, -1.0, -obs_edge_self->z1.z[0] * c1
    };

    //printf( "B1: " );
	//for ( unsigned int i = 0; i < 6; i ++ ) {
	//	printf( "%f ", B1[ i ] );
	//}
	//printf( "\n" );

    double B2[ 2 * 3 ] = {
        1.0, 0.0, -obs_edge_self->z2.z[0] * s2,
        0.0, 1.0, obs_edge_self->z2.z[0] * c2
    };

    //printf( "B2: " );
	//for ( unsigned int i = 0; i < 6; i ++ ) {
	//	printf( "%f ", B2[ i ] );
	//}
	//printf( "\n" );

    double transposeOFB1[ 3 * 2 ] = { 0.0 };
    tra( transposeOFB1, B1, 2, 3 );

    double transposeOFB2[ 3 * 2 ] = { 0.0 };
    tra( transposeOFB2, B2, 2, 3 );

    double dot_productOFtB1Omega[ 3 * 2 ] = { 0.0 };
    dot( dot_productOFtB1Omega, transposeOFB1, 3, 2, Omega, 2, 2 );

    double dot_productOFtB2Omega[ 3 * 2 ] = { 0.0 };
    dot( dot_productOFtB2Omega, transposeOFB2, 3, 2, Omega, 2, 2 );

    double dot_productOFtB1OmegaB1[ 3 * 3 ] = { 0.0 };
    dot( dot_productOFtB1OmegaB1, dot_productOFtB1Omega, 3, 2, B1, 2, 3 );
    for( unsigned int i = 0; i < 3 * 3; i ++ ) {
        obs_edge_self->omega_upperleft[ i ] = dot_productOFtB1OmegaB1[ i ];
    }

    //printf( "obs_edge_self->omega_upperleft: " );
	//for ( unsigned int i = 0; i < 9; i ++ ) {
	//	printf( "%f ", obs_edge_self->omega_upperleft[ i ] );
	//}
	//printf( "\n" );

    double dot_productOFtB1OmegaB2[ 3 * 3 ] = { 0.0 };
    dot( dot_productOFtB1OmegaB2, dot_productOFtB1Omega, 3, 2, B2, 2, 3 );
    for( unsigned int i = 0; i < 3 * 3; i ++ ) {
        obs_edge_self->omega_upperright[ i ] = dot_productOFtB1OmegaB2[ i ];
    }

    //printf( "obs_edge_self->omega_upperright: " );
	//for ( unsigned int i = 0; i < 9; i ++ ) {
	//	printf( "%f ", obs_edge_self->omega_upperright[ i ] );
	//}
	//printf( "\n" );

    double dot_productOFtB2OmegaB1[ 3 * 3 ] = { 0.0 };
    dot( dot_productOFtB2OmegaB1, dot_productOFtB2Omega, 3, 2, B1, 2, 3 );
    for( unsigned int i = 0; i < 3 * 3; i ++ ) {
        obs_edge_self->omega_bottomleft[ i ] = dot_productOFtB2OmegaB1[ i ];
    }

    //printf( "obs_edge_self->omega_bottomleft: " );
	//for ( unsigned int i = 0; i < 9; i ++ ) {
	//	printf( "%f ", obs_edge_self->omega_bottomleft[ i ] );
	//}
	//printf( "\n" );

    double dot_productOFtB2OmegaB2[ 3 * 3 ] = { 0.0 };
    dot( dot_productOFtB2OmegaB2, dot_productOFtB2Omega, 3, 2, B2, 2, 3 );
    for( unsigned int i = 0; i < 3 * 3; i ++ ) {
        obs_edge_self->omega_bottomright[ i ] = dot_productOFtB2OmegaB2[ i ];
    }

    //printf( "obs_edge_self->omega_bottomright: " );
	//for ( unsigned int i = 0; i < 9; i ++ ) {
	//	printf( "%f ", obs_edge_self->omega_bottomright[ i ] );
	//}
	//printf( "\n" );

    double dot_productOFtB1Omegahat_e[ 3 * 1 ] = { 0.0 };
    dot( dot_productOFtB1Omegahat_e, dot_productOFtB1Omega, 3, 2, hat_e, 2, 1 );
    for( unsigned int i = 0; i < 3 * 1; i ++ ) {
      obs_edge_self->xi_upper[ i ] = - dot_productOFtB1Omegahat_e[ i ];
    }

    //printf( "obs_edge_self->xi_upper: " );
	//for ( unsigned int i = 0; i < 3; i ++ ) {
	//	printf( "%f ", obs_edge_self->xi_upper[ i ] );
	//}
	//printf( "\n" );

    double dot_productOFtB2Omegahat_e[ 3 * 1 ] = { 0.0 };
    dot( dot_productOFtB2Omegahat_e, dot_productOFtB2Omega, 3, 2, hat_e, 2, 1 );
    for( unsigned int i = 0; i < 3 * 1; i ++ ) {
      obs_edge_self->xi_bottom[ i ] = - dot_productOFtB2Omegahat_e[ i ];
    }

    //printf( "obs_edge_self->xi_bottom: " );
	//for ( unsigned int i = 0; i < 3; i ++ ) {
	//	printf( "%f ", obs_edge_self->xi_bottom[ i ] );
	//}
	//printf( "\n" );

    for( unsigned int i = 0; i < 2 * 2; i ++ ) {
      obs_edge_self->Omega[ i ] = Omega[ i ]; // Not needed by ObsEdge
    }

    obs_edge_self->Omega[4] = 0.0; // Not needed by ObsEdge
    obs_edge_self->Omega[5] = 0.0; // Not needed by ObsEdge
    obs_edge_self->Omega[6] = 0.0; // Not needed by ObsEdge
    obs_edge_self->Omega[7] = 0.0; // Not needed by ObsEdge
    obs_edge_self->Omega[8] = 0.0; // Not needed by ObsEdge

    obs_edge_self->lambda = 1.0; // Not needed by ObsEdge

    return obs_edge_self;
}

void ObsEdge_destroy ( struct Edge *obs_edge ) {
    if ( obs_edge )
        free( obs_edge );
}
