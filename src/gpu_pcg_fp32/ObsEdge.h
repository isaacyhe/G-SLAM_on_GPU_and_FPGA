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

#ifndef OBSEDGE_H
#define OBSEDGE_H

#ifdef __cplusplus
extern "C"{
#endif

#include "Edge.h"
#include "U.h"
#include "HAT_X.h"
#include "Z.h"

struct Edge* ObsEdge_create( struct Edge *obs_edge_self, struct Z z1, struct Z z2, struct HAT_X *hat_xs, float *snr ); // The constructor
void ObsEdge_destroy ( struct Edge *obs_edge ); // The destructor


#ifdef __cplusplus
}
#endif
#endif
