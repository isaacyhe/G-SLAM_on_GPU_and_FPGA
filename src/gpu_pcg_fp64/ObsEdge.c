/* class ObsEdge:   ###graphbasedslam_2d_sensor_obsedge
    def __init__(self, t1, t2, z1, z2, xs, sensor_noise_rate=[0.14, 0.05]):  #ψの標準偏差を削除
        assert z1[0] == z2[0]

        self.t1, self.t2 = t1, t2
        self.x1, self.x2 = xs[t1], xs[t2]
        self.z1, self.z2 = z1[1], z2[1]

        s1 = math.sin(self.x1[2] + self.z1[1])
        c1 = math.cos(self.x1[2] + self.z1[1])
        s2 = math.sin(self.x2[2] + self.z2[1])
        c2 = math.cos(self.x2[2] + self.z2[1])

        ##誤差の計算##
        hat_e = self.x2[0:2] - self.x1[0:2] + np.array([       #self.x2とself.x1は上の2行だけを使う
            self.z2[0]*c2 - self.z1[0]*c1,
            self.z2[0]*s2 - self.z1[0]*s1
        ])                                                                   #ψに関する行列の行と、正規化していた行を削除。

        ##精度行列の作成##
        Q1 = np.diag([(self.z1[0]*sensor_noise_rate[0])**2, sensor_noise_rate[1]**2]) #ψの分散を削除
        R1 = - np.array([[c1, -self.z1[0]*s1],
                                     [s1,  self.z1[0]*c1]])    #3行目、3列目を削除

        Q2 = np.diag([(self.z2[0]*sensor_noise_rate[0])**2, sensor_noise_rate[1]**2]) #ψの分散を削除
        R2 = np.array([[c2, -self.z2[0]*s2],
                                  [s2,  self.z2[0]*c2]])    #3行目、3列目を削除

        Sigma = R1.dot(Q1).dot(R1.T) + R2.dot(Q2).dot(R2.T)
        Omega = np.linalg.inv(Sigma)                                                  #2x2行列になる

        ##大きな精度行列と係数ベクトルの各部分を計算##
        B1 = - np.array([[1, 0, -self.z1[0]*s1],
                                    [0, 1, self.z1[0]*c1]])                                  #3行目を削除
        B2 = np.array([[1, 0,  -self.z2[0]*s2],
                                   [0, 1,   self.z2[0]*c2]])                                  #3行目を削除

        self.omega_upperleft = B1.T.dot(Omega).dot(B1)         #ここは計算すると3x3行列のままになる
        self.omega_upperright = B1.T.dot(Omega).dot(B2)
        self.omega_bottomleft = B2.T.dot(Omega).dot(B1)
        self.omega_bottomright = B2.T.dot(Omega).dot(B2)

        self.xi_upper = - B1.T.dot(Omega).dot(hat_e)         #ここも計算すると3次元縦ベクトルのままになる
        self.xi_bottom = - B2.T.dot(Omega).dot(hat_e) */

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

    inv( Omega, Sigma, 2 );

    double B1[ 2 * 3 ] = {
        -1.0, -0.0, obs_edge_self->z1.z[0] * s1,
        -0.0, -1.0, -obs_edge_self->z1.z[0] * c1
    };

    double B2[ 2 * 3 ] = {
        1.0, 0.0, -obs_edge_self->z2.z[0] * s2,
        0.0, 1.0, obs_edge_self->z2.z[0] * c2
    };

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

    double dot_productOFtB1OmegaB2[ 3 * 3 ] = { 0.0 };
    dot( dot_productOFtB1OmegaB2, dot_productOFtB1Omega, 3, 2, B2, 2, 3 );
    for( unsigned int i = 0; i < 3 * 3; i ++ ) {
        obs_edge_self->omega_upperright[ i ] = dot_productOFtB1OmegaB2[ i ];
    }

    double dot_productOFtB2OmegaB1[ 3 * 3 ] = { 0.0 };
    dot( dot_productOFtB2OmegaB1, dot_productOFtB2Omega, 3, 2, B1, 2, 3 );
    for( unsigned int i = 0; i < 3 * 3; i ++ ) {
        obs_edge_self->omega_bottomleft[ i ] = dot_productOFtB2OmegaB1[ i ];
    }

    double dot_productOFtB2OmegaB2[ 3 * 3 ] = { 0.0 };
    dot( dot_productOFtB2OmegaB2, dot_productOFtB2Omega, 3, 2, B2, 2, 3 );
    for( unsigned int i = 0; i < 3 * 3; i ++ ) {
        obs_edge_self->omega_bottomright[ i ] = dot_productOFtB2OmegaB2[ i ];
    }

    double dot_productOFtB1Omegahat_e[ 3 * 1 ] = { 0.0 };
    dot( dot_productOFtB1Omegahat_e, dot_productOFtB1Omega, 3, 2, hat_e, 2, 1 );
    for( unsigned int i = 0; i < 3 * 1; i ++ ) {
      obs_edge_self->xi_upper[ i ] = - dot_productOFtB1Omegahat_e[ i ];
    }

    double dot_productOFtB2Omegahat_e[ 3 * 1 ] = { 0.0 };
    dot( dot_productOFtB2Omegahat_e, dot_productOFtB2Omega, 3, 2, hat_e, 2, 1 );
    for( unsigned int i = 0; i < 3 * 1; i ++ ) {
      obs_edge_self->xi_bottom[ i ] = - dot_productOFtB2Omegahat_e[ i ];
    }

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
