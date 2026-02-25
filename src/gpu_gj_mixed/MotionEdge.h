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

#ifndef MOTIONEDGE_H
#define MOTIONEDGE_H

#ifdef __cplusplus
extern "C"{
#endif

#include "Edge.h"
#include "U.h"
#include "HAT_X.h"
#include "Z.h"
#include "GSLAM.h"

	extern void state_transition( double *result, double nu, double omega, unsigned int delta, struct HAT_X pose );
extern struct Edge *motion_edge_self;

struct Edge *MotionEdge_create ( unsigned int t1, unsigned int t2, struct HAT_X *hat_xs, struct U *us, unsigned int delta, double lambda, double *mns ); // The constructor
void MotionEdge_destroy ( struct Edge *motion_edge ); // The destructor

#ifdef __cplusplus
}
#endif
#endif
