#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('./scripts/')
from kf import *  # 誤差楕円を描くのに利用

import matplotlib as mpl  # IYH

mpl.use('TkAgg')  # IYH
import matplotlib.pyplot as plt  # IYH


def make_ax():  # axisの準備
    # fig = plt.figure(figsize=(4,4))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    # ax.set_xlim(-5,5)
    # ax.set_ylim(-5,5)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    return ax


def draw_trajectory(xs, ax):  # 軌跡の描画
    poses = [xs[s] for s in range(len(xs))]
    ax.scatter([e[0] for e in poses], [e[1] for e in poses], s=5, marker=".", color="black")
    ax.plot([e[0] for e in poses], [e[1] for e in poses], linewidth=0.5, color="black")


def draw_observations(xs, zlist, ax):  # センサ値の描画
    for s in range(len(xs)):
        if s not in zlist:
            continue

        for obs in zlist[s]:
            x, y, theta = xs[s]
            ell, phi = obs[1][0], obs[1][1]
            mx = x + ell * math.cos(theta + phi)
            my = y + ell * math.sin(theta + phi)
            ax.plot([x, mx], [y, my], color="pink", alpha=0.5)


# Not used
##def draw_edges(edges, ax):
##    for e in edges:
##        ax.plot([e.hat_x1[0], e.hat_x2[0]], [e.hat_x1[1] ,e.hat_x2[1]], color="red", alpha=0.5)

def draw_landmarks(ms, ax):
    ax.scatter([ms[k][0] for k in ms], [ms[k][1] for k in ms], s=100, marker="*", color="blue", zorder=100)


def draw(hat_xs_steps, hat_xs, zlist_steps, zlist_landmark_ids, zs, ms_indexes, mses):
    xs = {}
    for i in range(len(hat_xs_steps)):
        xs[hat_xs_steps[i]] = np.array([hat_xs[i * 3], hat_xs[i * 3 + 1], hat_xs[i * 3 + 2]]).T

    zlist = {}
    for i in range(len(zlist_steps)):
        if zlist_steps[i] not in zlist:
            zlist[zlist_steps[i]] = []
        zlist[zlist_steps[i]].append((zlist_landmark_ids[i], np.array([zs[i * 2], zs[i * 2 + 1]]).T))

    ms = {}
    if len(ms_indexes) != 0:
        for i in range(len(ms_indexes)):
            ms[ms_indexes[i]] = np.array([mses[i * 2], mses[i * 2 + 1]])

    ax = make_ax()
    draw_observations(xs, zlist, ax)
    draw_trajectory(xs, ax)
    draw_landmarks(ms, ax)  # 追加
    plt.show()
