/* class MotionEdge:
    def __init__(self, t1, t2, xs, us, delta, lmd=1.0, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}):
        self.t1, self.t2 = t1, t2                   #時刻の記録
        self.hat_x1, self.hat_x2 = xs[t1], xs[t2]    #各時刻の姿勢

        nu, omega = us[t2]
        if abs(omega) < 1e-5: omega = 1e-5 #ゼロにすると式が変わるので避ける

        M = matM(nu, omega, delta, motion_noise_stds)
        A = matA(nu, omega, delta, self.hat_x1[2])
        F = matF(nu, omega, delta, self.hat_x1[2])

        self.Omega = np.linalg.inv(A.dot(M).dot(A.T) + np.eye(3)*0.0001) #標準偏差0.01の雑音を足す

        self.omega_upperleft = F.T.dot(self.Omega).dot(F)*lmd
        self.omega_upperright = -F.T.dot(self.Omega)*lmd
        self.omega_bottomleft = - self.Omega.dot(F)*lmd
        self.omega_bottomright = self.Omega*lmd

        x2 = IdealRobot.state_transition(nu, omega, delta, self.hat_x1)
        self.xi_upper = F.T.dot(self.Omega).dot(self.hat_x2 - x2)*lmd
        self.xi_bottom = -self.Omega.dot(self.hat_x2 - x2)*lmd */

/* def matM(nu, omega, time, stds):
    return np.diag([stds["nn"]**2*abs(nu)/time + stds["no"]**2*abs(omega)/time,
                               stds["on"]**2*abs(nu)/time + stds["oo"]**2*abs(omega)/time])

def matA(nu, omega, time, theta):
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega*time), math.cos(theta + omega*time)
    return np.array([[(stw - st)/omega,    -nu/(omega**2)*(stw - st) + nu/omega*time*ctw],
                                 [(-ctw + ct)/omega, -nu/(omega**2)*(-ctw + ct) + nu/omega*time*stw],
                                 [0,                                time]] )

def matF(nu, omega, time, theta):
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = nu / omega * (math.cos(theta + omega * time) - math.cos(theta))
    F[1, 2] = nu / omega * (math.sin(theta + omega * time) - math.sin(theta))
    return F */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "MatrixOps.h"
#include "MotionEdge.h"
#include "GSLAM.h"

struct Edge *motion_edge_self;

struct Edge* MotionEdge_create ( unsigned int t1, unsigned int t2, struct HAT_X *hat_xs, struct U *us, unsigned int delta, double lambda, double *mns ) { // The constructor
    motion_edge_self = malloc ( sizeof( struct Edge ) );

    motion_edge_self->t1 = t1;
    motion_edge_self->t2 = t2;
//printf("motion_edge_self->t1: %d ", motion_edge_self->t1);
//printf("motion_edge_self->t2: %d ", motion_edge_self->t2);
    motion_edge_self->hat_x1 = hat_xs[t1];
    motion_edge_self->hat_x2 = hat_xs[t2];
//printf("motion_edge_self->t1: %d %f ", motion_edge_self->hat_x1.step, motion_edge_self->hat_x1.hat_x[2]);
    motion_edge_self->lambda = lambda;

    double nu = us[t2].nu;
    double omega = us[t2].omega;

    if ( fabs( omega ) < 1e-5 )
        omega = 1e-5;

    //printf( "%f ", nu );
    //printf( "omega: %f ", omega );

    double M[ 2 * 2 ] = {
        mns[ 0 ] * mns[ 0 ] * fabs( nu ) / delta + mns[ 1 ] * mns[ 1 ] * fabs( omega ) / delta, 0.0,
        0.0, mns[ 2 ] * mns[ 2 ] * fabs( nu ) / delta + mns[ 3 ] * mns[ 3 ] * fabs( omega ) / delta
    };

    //printf( "%f ", M[ 0 ] );
	//printf( "%f ", M[ 1 ] );
	//printf( "%f ", M[ 2 ] );
	//printf( "%f\n", M[ 3 ] );
//printf("motion_edge_self->hat_x1.hat_x[2]: %f", motion_edge_self->hat_x1.hat_x[2]);
    double st = sin( motion_edge_self->hat_x1.hat_x[2] );
    double ct = cos( motion_edge_self->hat_x1.hat_x[2] );

    double stw = sin( motion_edge_self->hat_x1.hat_x[2] + omega * delta );
    double ctw = cos( motion_edge_self->hat_x1.hat_x[2] + omega * delta );

    //printf( "%f ", st );
	//printf( "%f ", ct );
	//printf( "%f ", stw );
	//printf( "%f\n", ctw );

    //double A[ 3 * 2 ] = {
    //    ( stw - st ) / omega, nu / omega * delta * ctw - nu / powf( omega, 2 ) * ( stw - st ),
    //    ( ct - ctw ) / omega, nu / omega * delta * stw - nu / powf( omega, 2 ) * ( ct - ctw ),
    //    0.0, delta
    //};

    double A[ 3 * 2 ] = {
	        stw / omega - st / omega, nu / omega * delta * ctw - nu / omega / omega * stw + nu / omega / omega * st,
	        ct / omega - ctw / omega, nu / omega * delta * stw - nu / omega / omega * ct + nu / omega / omega * ctw,
	        0.0, delta
    };

    //printf( "%f ", A[ 0 ] );
	//printf( "%f ", A[ 1 ] );
	//printf( "%f ", A[ 2 ] );
	//printf( "%f ", A[ 3 ] );
	//printf( "%f ", A[ 4 ] );
	//printf( "%f\n", A[ 5 ] );

    double F[ 3 * 3 ] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    F[ 3 * 0 + 2 ] = nu / omega * cos( motion_edge_self->hat_x1.hat_x[2] + omega * delta ) - nu / omega * cos( motion_edge_self->hat_x1.hat_x[2] );
    F[ 3 * 1 + 2 ] = nu / omega * sin( motion_edge_self->hat_x1.hat_x[2] + omega * delta ) - nu / omega * sin( motion_edge_self->hat_x1.hat_x[2] );

    double NP_EYE[ 3 * 3 ] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    double transposeOfA[ 2 * 3 ] = { 0.0 };
    tra( transposeOfA, A, 3, 2 );

    double dot_productOfAM[ 3 * 2 ] = { 0.0 };
    dot( dot_productOfAM, A, 3, 2, M, 2, 2 );

    double dot_productOfDPtA[ 3 * 3 ] = { 0.0 };
    dot( dot_productOfDPtA, dot_productOfAM, 3, 2, transposeOfA, 2, 3 );

    double multiplicationOfN[ 3 * 3 ] = { 0.0 };
    mul( multiplicationOfN, NP_EYE, 3, 3, 0.0001 );

    double addition[ 3 * 3 ] = { 0.0 };
    add( addition, dot_productOfDPtA, 3, 3, multiplicationOfN, 3, 3 );

	double inverse[ 3 * 3 ] = { 0.0 };
	inv( inverse, addition, 3 );

	for( unsigned int i = 0; i < 3 * 3; i ++ ) {
		motion_edge_self->Omega[ i ] = inverse[ i ];
    }

    double transposeOfF[ 3 * 3 ] = { 0.0 };
    tra( transposeOfF, F, 3, 3 );

    double dot_productOftFmO[ 3 * 3 ] = { 0.0 };
    dot( dot_productOftFmO, transposeOfF, 3, 3, motion_edge_self->Omega, 3, 3 );

    double dot_productOfDPF[ 3 * 3 ] = { 0.0 };
    dot( dot_productOfDPF, dot_productOftFmO, 3, 3, F, 3, 3 );

    double multiplicationOfDPL_1[ 3 * 3 ] = { 0.0 };
    mul( multiplicationOfDPL_1, dot_productOfDPF, 3, 3, lambda );

	for( unsigned int i = 0; i < 3 * 3; i ++ ) {
		motion_edge_self->omega_upperleft[ i ] = multiplicationOfDPL_1[ i ];
    }

    double multiplicationOfDPL_2[ 3 * 3 ] = { 0.0 };
    mul( multiplicationOfDPL_2, dot_productOftFmO, 3, 3, -lambda );

	for( unsigned int i = 0; i < 3 * 3; i ++ ) {
		motion_edge_self->omega_upperright[ i ] = multiplicationOfDPL_2[ i ];
    }

    double dot_productOftmOF[ 3 * 3 ] = { 0.0 };
    dot( dot_productOftmOF, motion_edge_self->Omega, 3, 3, F, 3, 3 );

    double multiplicationOfDPL_3[ 3 * 3 ] = { 0.0 };
    mul( multiplicationOfDPL_3, dot_productOftmOF, 3, 3, -lambda );

	for( unsigned int i = 0; i < 3 * 3; i ++ ) {
		motion_edge_self->omega_bottomleft[ i ] = multiplicationOfDPL_3[ i ];
    }

    double multiplicationOfmOL[ 3 * 3 ] = { 0.0 };
    mul( multiplicationOfmOL, motion_edge_self->Omega, 3, 3, lambda );

	for( unsigned int i = 0; i < 3 * 3; i ++ ) {
		motion_edge_self->omega_bottomright[ i ] = multiplicationOfmOL[ i ];
    }

    double x2[ 3 * 1 ] = { 0.0 };
    state_transition( x2, nu, omega, delta, motion_edge_self->hat_x1 );

    double subtractionOfmHX2[ 3 * 1 ] = { 0.0 };
    sub( subtractionOfmHX2, motion_edge_self->hat_x2.hat_x, 3, 1, x2, 3, 1 );

    double dot_productOfDPS[ 3 * 1 ] = { 0.0 };
    dot( dot_productOfDPS, dot_productOftFmO, 3, 3, subtractionOfmHX2, 3, 1 );

    double multiplicationOfDPL_4[ 3 * 1 ] = { 0.0 };
    mul( multiplicationOfDPL_4, dot_productOfDPS, 3, 1, lambda );

	for( unsigned int i = 0; i < 3 * 1; i ++ ) {
		motion_edge_self->xi_upper[ i ] = multiplicationOfDPL_4[ i ];
    }

    double dot_productOfmOS[ 3 * 1 ] = { 0.0 };
    dot( dot_productOfmOS, motion_edge_self->Omega, 3, 3, subtractionOfmHX2, 3, 1 );

    double multiplicationOfDPL_5[ 3 * 1 ] = { 0.0 };
    mul( multiplicationOfDPL_5, dot_productOfmOS, 3, 1, -lambda );

	for( unsigned int i = 0; i < 3 * 1; i ++ ) {
		motion_edge_self->xi_bottom[ i ] = multiplicationOfDPL_5[ i ];
    }

    for( unsigned int i = 0; i < 2 * 1; i ++ ) {
		motion_edge_self->hat_e[ i ] = 0.0; // Not needed by MotionEdge
    }

    return motion_edge_self;
}

void MotionEdge_destroy ( struct Edge *motion_edge ) { // The destructor
    if ( motion_edge )
        free( motion_edge );
}
