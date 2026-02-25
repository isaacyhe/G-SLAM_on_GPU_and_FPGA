/* class MapEdge: ###graphbasedslam_2d_sensor_mapedge
    def __init__(self, t, z, head_t, head_z, xs, sensor_noise_rate=[0.14, 0.05]):  #センサの雑音モデルを削除
        self.x = xs[t]
        self.z = z

        self.m = self.x[0:2] + np.array([z[0]*math.cos(self.x[2] + z[1]), z[0]*math.sin(self.x[2] + z[1])]).T  #3行目削除

        ##精度行列の計算##
        Q1 = np.diag([(self.z[0]*sensor_noise_rate[0])**2, sensor_noise_rate[1]**2]) #ψの分散を削除

        s1 = math.sin(self.x[2] + self.z[1])
        c1 = math.cos(self.x[2] + self.z[1])
        R = np.array([[-c1, self.z[0]*s1],
                                [-s1,-self.z[0]*c1]]) #3行目、3列目を削除

        self.Omega = R.dot(Q1).dot(R.T) #2x2行列になる
        self.xi = self.Omega.dot(self.m)   #2次元ベクトルになる */

#ifndef MAPEDGE_H
#define MAPEDGE_H

#include "HAT_X.h"
#include "Z.h"

#ifdef __cplusplus
extern "C"{
#endif

struct MapEdge {
    struct HAT_X x;
    struct Z z;
    float m[ 2 * 1 ], Omega[ 2 * 2 ], xi[ 2 * 1 ];
};

extern struct MapEdge *map_edge_self;

struct MapEdge *MapEdge_create ( struct Z z, struct Z head_z, struct HAT_X *hat_xs, float *snr ); // The constructor
void MapEdge_destroy ( struct MapEdge *map_edge ); // The destructor


#ifdef __cplusplus
}
#endif
#endif
