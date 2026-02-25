#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import math
import numpy as np

import matplotlib as mpl  # IYH

mpl.use('TkAgg')  # IYH
import matplotlib.pyplot as plt  # IYH


# In[2]:

def make_ax():  # axisの準備
    # fig = plt.figure(figsize=(4,4))  ##做一个长为4宽为4的图
    fig = plt.figure()  ##图像标题默认为Figure 1
    ax = fig.add_subplot(111)  ##画布分割成1行1列，图像画在从左到右从上到下第1块
    # ax.set_aspect('equal')
    # ax.set_xlim(-5,5)  #设置坐标轴的显示范围
    # ax.set_ylim(-5,5)
    ax.set_xlabel("X", fontsize=10)  #为子图设置x轴标题，字体大小为10
    ax.set_ylabel("Y", fontsize=10)
    return ax


def draw_trajectory(xs, ax):  # 軌跡の描画
    poses = [xs[s] for s in range(len(xs))]
    ax.scatter([e[0] for e in poses], [e[1] for e in poses], s=5, marker=".", color="black")  ##绘制散点图，参数s为点的大小，marker表示的是标记的样式“.”点
    ax.plot([e[0] for e in poses], [e[1] for e in poses], linewidth=0.5, color="black")  ##Axes.plot用于绘制XY坐标系的点、线或其他标记形状


def draw_observations(xs, zlist, ax):  # センサ値の描画  ##绘制传感器值
    for s in range(len(xs)):  ##返回一系列连续增加的整数，range()函数内只有一个参数，则表示会产生从0开始计数的整数列表
        if s not in zlist:
            continue

        for obs in zlist[s]:
            x, y, theta = xs[s]
            ell, phi = obs[1][0], obs[1][1]
            mx = x + ell * math.cos(theta + phi)
            my = y + ell * math.sin(theta + phi)
            ax.plot([x, mx], [y, my], color="pink", alpha=0.5) ##alpha，表示透明度，浮点类型


# def draw_edges(edges, ax):
#    for e in edges:
#        ax.plot([e.hat_x1[0], e.hat_x2[0]], [e.hat_x1[1] ,e.hat_x2[1]], color="red", alpha=0.5)

def draw_landmarks(ms, ax):  ##绘制路标
    ax.scatter([ms[k][0] for k in ms], [ms[k][1] for k in ms], s=100, marker="*", color="blue", zorder=100)


def draw(xs, zlist, edges, ms={}):  # ms追加
    ax = make_ax()
    draw_observations(xs, zlist, ax)
    draw_trajectory(xs, ax)
    draw_landmarks(ms, ax)  # 追加
    plt.show()


def state_transition(nu, omega, time, pose):
    t0 = pose[2]
    # print(pose)

    if math.fabs(omega) < 1e-10:
        return pose + np.array([nu * math.cos(t0),
                                nu * math.sin(t0),
                                omega]) * time
    else:
        return pose + np.array([nu / omega * (math.sin(t0 + omega * time) - math.sin(t0)),
                                nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)),
                                omega * time])


def matM(nu, omega, time, stds):
    return np.diag([stds["nn"] ** 2 * abs(nu) / time + stds["no"] ** 2 * abs(omega) / time,
                    stds["on"] ** 2 * abs(nu) / time + stds["oo"] ** 2 * abs(omega) / time])


def matA(nu, omega, time, theta):
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega * time), math.cos(theta + omega * time)
    # print(st)
    # print(ct)
    # print(stw)
    # print(ctw)
    # print(-nu/(omega**2))
    # print((stw - st))
    return np.array([[(stw - st) / omega, -nu / (omega ** 2) * (stw - st) + nu / omega * time * ctw],
                     [(-ctw + ct) / omega, -nu / (omega ** 2) * (-ctw + ct) + nu / omega * time * stw],
                     [0, time]])


def matF(nu, omega, time, theta):
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = nu / omega * (math.cos(theta + omega * time) - math.cos(theta))
    F[1, 2] = nu / omega * (math.sin(theta + omega * time) - math.sin(theta))
    return F


# In[3]:

def read_data():  ###graphbasedslam_2d_sensor_readdata
    hat_xs = {}
    zlist = {}
    delta = 0.0
    us = {}

    if len(sys.argv) != 3:
        print('Usage:\npyhton3', sys.argv[0], '0|1|2|3 path/to/input/file')
        sys.exit()

    with open(sys.argv[2]) as f:  # log2.txtに変えておく
        for line in f.readlines():
            tmp = line.rstrip().split()

            step = int(float(tmp[1]))
            if tmp[0] == "x":
                # print(np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]))
                # print(np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]).T)
                hat_xs[step] = np.array([float(tmp[2]), float(tmp[3]), float(tmp[4])]).T
            elif tmp[0] == "z":
                if step not in zlist:
                    zlist[step] = []
                # print(step)
                # print(int(tmp[2]))
                # print(np.array([float(tmp[3]), float(tmp[4])]))
                # print(np.array([float(tmp[3]), float(tmp[4])]).T)
                zlist[step].append((int(tmp[2]), np.array([float(tmp[3]), float(tmp[4])]).T))  # 変更。ψを読み込まないように
            elif tmp[0] == "delta":
                delta = float(tmp[1])
            elif tmp[0] == "u":
                # print(np.array([float(tmp[2]), float(tmp[3])]))
                # print(np.array([float(tmp[2]), float(tmp[3])]).T)
                us[step] = np.array([float(tmp[2]), float(tmp[3])]).T
                # print(us[step])

        return hat_xs, zlist, us, delta  # us, deltaも返す


# In[4]:

class ObsEdge:  ###graphbasedslam_2d_sensor_obsedge
    def __init__(self, t1, t2, z1, z2, xs, sensor_noise_rate=[0.14, 0.05]):  # ψの標準偏差を削除
        assert z1[0] == z2[0]

        # print(z1)
        # print(z2)
        # print("\n")

        self.t1, self.t2 = t1, t2
        self.x1, self.x2 = xs[t1], xs[t2]
        self.z1, self.z2 = z1[1], z2[1]

        s1 = math.sin(self.x1[2] + self.z1[1])
        c1 = math.cos(self.x1[2] + self.z1[1])
        s2 = math.sin(self.x2[2] + self.z2[1])
        c2 = math.cos(self.x2[2] + self.z2[1])

        ##誤差の計算##
        hat_e = self.x2[0:2] - self.x1[0:2] + np.array([  # self.x2とself.x1は上の2行だけを使う
            self.z2[0] * c2 - self.z1[0] * c1,
            self.z2[0] * s2 - self.z1[0] * s1
        ])  # ψに関する行列の行と、正規化していた行を削除。

        ##精度行列の作成##
        Q1 = np.diag([(self.z1[0] * sensor_noise_rate[0]) ** 2, sensor_noise_rate[1] ** 2])  # ψの分散を削除
        R1 = - np.array([[c1, -self.z1[0] * s1],
                         [s1, self.z1[0] * c1]])  # 3行目、3列目を削除

        Q2 = np.diag([(self.z2[0] * sensor_noise_rate[0]) ** 2, sensor_noise_rate[1] ** 2])  # ψの分散を削除
        R2 = np.array([[c2, -self.z2[0] * s2],
                       [s2, self.z2[0] * c2]])  # 3行目、3列目を削除

        Sigma = R1.dot(Q1).dot(R1.T) + R2.dot(Q2).dot(R2.T)
        # print(Sigma)
        Omega = np.linalg.inv(Sigma)  # 2x2行列になる
        # print(Omega)
        # print("\n")

        ##大きな精度行列と係数ベクトルの各部分を計算##
        B1 = - np.array([[1, 0, -self.z1[0] * s1],
                         [0, 1, self.z1[0] * c1]])  # 3行目を削除
        # print(B1)
        B2 = np.array([[1, 0, -self.z2[0] * s2],
                       [0, 1, self.z2[0] * c2]])  # 3行目を削除
        # print(B2)

        self.omega_upperleft = B1.T.dot(Omega).dot(B1)  # ここは計算すると3x3行列のままになる
        self.omega_upperright = B1.T.dot(Omega).dot(B2)
        self.omega_bottomleft = B2.T.dot(Omega).dot(B1)
        self.omega_bottomright = B2.T.dot(Omega).dot(B2)

        # print(self.omega_upperleft)
        # print(self.omega_upperright)
        # print(self.omega_bottomleft)
        # print(self.omega_bottomright)
        # print("\n")

        self.xi_upper = - B1.T.dot(Omega).dot(hat_e)  # ここも計算すると3次元縦ベクトルのままになる
        self.xi_bottom = - B2.T.dot(Omega).dot(hat_e)

        # print(self.xi_upper)
        # print(self.xi_bottom)
        # print("\n")


# In[5]:

class MotionEdge:
    def __init__(self, t1, t2, xs, us, delta, lmd=1.0,
                 motion_noise_stds={"nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2}):
        self.t1, self.t2 = t1, t2  # 時刻の記録
        self.hat_x1, self.hat_x2 = xs[t1], xs[t2]  # 各時刻の姿勢

        nu, omega = us[t2]
        if abs(omega) < 1e-5: omega = 1e-5  # ゼロにすると式が変わるので避ける

        # print(nu)
        # print(omega)
        # print(np.array([float(tmp[2]), float(tmp[3])]))
        # print(np.array([float(tmp[2]), float(tmp[3])]))

        M = matM(nu, omega, delta, motion_noise_stds)
        # print(M)
        A = matA(nu, omega, delta, self.hat_x1[2])
        # print(A)
        F = matF(nu, omega, delta, self.hat_x1[2])
        # print(F)

        self.Omega = np.linalg.inv(A.dot(M).dot(A.T) + np.eye(3) * 0.0001)  # 標準偏差0.01の雑音を足す
        # print(self.Omega)
        # print("\n")

        self.omega_upperleft = F.T.dot(self.Omega).dot(F) * lmd
        self.omega_upperright = -F.T.dot(self.Omega) * lmd
        self.omega_bottomleft = -self.Omega.dot(F) * lmd
        self.omega_bottomright = self.Omega * lmd

        x2 = state_transition(nu, omega, delta, self.hat_x1)

        self.xi_upper = F.T.dot(self.Omega).dot(self.hat_x2 - x2) * lmd
        self.xi_bottom = -self.Omega.dot(self.hat_x2 - x2) * lmd

        # print(F.T.dot(self.Omega).dot(self.hat_x2 - x2))
        # print(-self.Omega.dot(self.hat_x2 - x2))

        # print(self.xi_upper)
        # print(self.xi_bottom)
        # print("\n")


# In[6]:

import itertools


def make_edges(hat_xs, zlist):
    landmark_keys_zlist = {}

    for step in zlist:
        # print(step)
        for z in zlist[step]:
            # print(z)
            # print(zlist[step])
            # print("\n")
            # print(z[0])
            landmark_id = z[0]
            if landmark_id not in landmark_keys_zlist:
                landmark_keys_zlist[landmark_id] = []

            landmark_keys_zlist[landmark_id].append((step, z))

    # for landmark_id in landmark_keys_zlist:
    # print(landmark_id)
    # print(landmark_keys_zlist[landmark_id])
    # print("\n")

    edges = []
    for landmark_id in landmark_keys_zlist:
        # print(len(landmark_keys_zlist[landmark_id]))
        # print(landmark_keys_zlist[landmark_id])
        # print("\n")
        step_pairs = list(itertools.combinations(landmark_keys_zlist[landmark_id], 2))
        # print(len(step_pairs))
        # print(len(step_pairs[0]))
        # print(step_pairs[0])
        edges += [ObsEdge(xz1[0], xz2[0], xz1[1], xz2[1], hat_xs) for xz1, xz2 in step_pairs]
        # print(len(edges))
        # print(edges)

    # count = 0
    # for xz1, xz2 in step_pairs:
    # count = count + 1
    # print(count)
    # print(xz1)
    # print(xz2)
    # print(xz1[0])
    # print(xz1[1])
    # print(xz2[0])
    # print(xz2[1])
    # print("\n")

    # print("make_edges: ")
    # print(len(edges))
    return edges, landmark_keys_zlist  # ランドマークをキーにしたリストlandmark_keys_zlistも返す


# In[7]:

def add_edge(edge, Omega, xi):
    # print("before:")
    # print(edge)
    # print("\n")
    # print(Omega)
    # print("\n")
    # print(xi)

    f1, f2 = edge.t1 * 3, edge.t2 * 3
    t1, t2 = f1 + 3, f2 + 3

    # print(f1)
    # print(t1)
    # print(f2)
    # print(t2)
    # print("\n")

    # print(Omega)
    # print("\n")
    # print(f1:t1)
    # print("\n")
    # print(Omega[f1:t1, f1:t1])
    # print("\n")
    # print(edge.omega_upperleft)
    # print("\n")

    Omega[f1:t1, f1:t1] += edge.omega_upperleft
    Omega[f1:t1, f2:t2] += edge.omega_upperright
    Omega[f2:t2, f1:t1] += edge.omega_bottomleft
    Omega[f2:t2, f2:t2] += edge.omega_bottomright

    # print(xi)
    # print("\n")
    # print(edge.xi_upper)
    # print(edge.xi_bottom)
    # print("\n")

    xi[f1:t1] += edge.xi_upper
    xi[f2:t2] += edge.xi_bottom

    # print("after:")
    # print(edge)
    # print("\n")
    # print(Omega)
    # print("\n")
    # print(xi)


# In[8]:

hat_xs, zlist, us, delta = read_data()
dim = len(hat_xs) * 3

# print(dim)

for n in range(1, 51):
    # print(n)
    ##エッジ、大きな精度行列、係数ベクトルの作成##
    edges, _ = make_edges(hat_xs, zlist)  # 返す変数が2つになるので「_」で合わせる
    # edges = []

    # print(edges)
    # print(len(edges))
    # print("\n")

    for i in range(len(hat_xs) - 1):  # 行動エッジの追加
        # print(us[i])
        edges.append(MotionEdge(i, i + 1, hat_xs, us, delta, 10.0))  # lambda=100

    # print(len(hat_xs))
    # print(len(zlist))
    # print(len(edges))

    if int(sys.argv[1]) >= 3:
        draw(hat_xs, zlist, edges)

    Omega = np.zeros((dim, dim))
    xi = np.zeros(dim)

    # print(Omega[0:3, 0:3])
    # print(Omega)
    Omega[0:3, 0:3] += np.eye(3) * 1000000
    # print(Omega[0:3, 0:3])
    # print(Omega)

    # print(len(Omega))
    # print(len(Omega[0]))

    ##軌跡を動かす量（差分）の計算##
    for e in edges:
        # print(e.t1)
        # print(e.t2)
        add_edge(e, Omega, xi)

        # print(Omega)
    # print(Omega[185:186,0:186])
    # print(xi)
    # print(len(edges))
    # print("\n")

    delta_xs = np.linalg.inv(Omega).dot(xi)
    # print(np.linalg.inv(Omega)[0:2,0:186])
    # print(delta_xs[0:3])
    # print(delta_xs)

    ##推定値の更新##
    for i in range(len(hat_xs)):
        # print(hat_xs[i])
        # print(delta_xs[i*3:(i+1)*3])
        hat_xs[i] += delta_xs[i * 3:(i + 1) * 3]
        # print(hat_xs[i])

    ##終了判定##
    diff = np.linalg.norm(delta_xs)
    print("{} rounds executed: {}".format(n, diff))
    if diff < 0.01:
        if int(sys.argv[1]) >= 2:
            draw(hat_xs, zlist, edges)
        break

# In[9]:

_, zlist_landmark = make_edges(hat_xs, zlist)


# print(zlist_landmark)
# print("\n")
# print(zlist_landmark[0])
# print("\n")
# print(zlist_landmark[0][0])
# print("\n")
# print(zlist_landmark[0][1][1])

# In[10]:

class MapEdge:  ###graphbasedslam_2d_sensor_mapedge
    def __init__(self, t, z, head_t, head_z, xs, sensor_noise_rate=[0.14, 0.05]):  # センサの雑音モデルを削除
        self.x = xs[t]
        self.z = z

        self.m = self.x[0:2] + np.array(
            [z[0] * math.cos(self.x[2] + z[1]), z[0] * math.sin(self.x[2] + z[1])]).T  # 3行目削除
        # print(self.x[0:2])
        # print(np.array([z[0]*math.cos(self.x[2] + z[1]), z[0]*math.sin(self.x[2] + z[1])]).T)
        # print(self.x[0:2] + np.array([z[0]*math.cos(self.x[2] + z[1]), z[0]*math.sin(self.x[2] + z[1])]).T)

        ##精度行列の計算##
        Q1 = np.diag([(self.z[0] * sensor_noise_rate[0]) ** 2, sensor_noise_rate[1] ** 2])  # ψの分散を削除

        s1 = math.sin(self.x[2] + self.z[1])
        c1 = math.cos(self.x[2] + self.z[1])
        R = np.array([[-c1, self.z[0] * s1],
                      [-s1, -self.z[0] * c1]])  # 3行目、3列目を削除

        self.Omega = R.dot(Q1).dot(R.T)  # 2x2行列になる
        self.xi = self.Omega.dot(self.m)  # 2次元ベクトルになる


# In[11]:

ms = {}  ###graphbasedslam_2d_sensor_mapexec
for landmark_id in zlist_landmark:
    # print( landmark_id )
    edges = []
    head_z = zlist_landmark[landmark_id][0]
    # print(zlist_landmark[landmark_id])
    for z in zlist_landmark[landmark_id]:
        # print(z)
        # print(z[1][1])
        # print(head_z)
        # print(head_z[1][1])
        # print("\n")
        edges.append(MapEdge(z[0], z[1][1], head_z[0], head_z[1][1], hat_xs))

    Omega = np.zeros((2, 2))  # 2x2に
    xi = np.zeros(2)  # 2次元に

    for e in edges:
        Omega += e.Omega
        xi += e.xi

    ms[landmark_id] = np.mean([e.m for e in edges], axis=0)
    # print([e.m for e in edges])
    # print("\n")

# print(ms)
# print(len(edges))
# print(ms[0][0])

if int(sys.argv[1]) >= 1:
    draw(hat_xs, zlist, edges, ms)

# In[12]:

zlist

# In[13]:

hat_xs
