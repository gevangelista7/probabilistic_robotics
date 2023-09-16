import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, ConnectionPatch, Ellipse
import shutil

import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, ConnectionPatch
import shapely.geometry as shp
import seaborn as sns
from scipy.stats import norm

from utils import stochastic_universal_sampler, Map, normalize_angle, HGrid, PGrid, smaller_arc_between_angles, clear_directory
from motion_model import motion_model_velocity, sample_model_velocity
from percep_model import landmark_model_known_correspondence
from plot_utils import plot_robot_v3, plot_MCL_scene, plot_scene_with_confidence, make_movie


def EKF_SLAM_known_correspondences(mu_tp, Sigma_tp, u_t, z_t, c_t,
                                   N, seen_cits,
                                   R = None, deltat=1,
                                   sigmas_percep=(.1, .1, .1)):

    ### verify
    assert z_t.shape[0] == 3
    assert z_t.shape[1] == len(c_t)

    ### unzip
    mu_tp_x = mu_tp[0].item()
    mu_tp_y = mu_tp[1].item()
    mu_tp_theta = mu_tp[2].item()

    v = u_t[0]
    w = u_t[1]

    if abs(w) < 1e-6:
        w = 1e-6

    r = v/w
    r = np.clip(r, -1e8, 1e8)

    if R is None:
        R = np.eye(3)/100

    ### Motion update
    F_x = np.hstack((np.eye(3), np.zeros((3, 3 * N))))

    mov0 = - r * sin(mu_tp_theta) + r * sin(mu_tp_theta + w * deltat)
    mov1 =   r * cos(mu_tp_theta) - r * cos(mu_tp_theta + w * deltat)
    mov2 = w * deltat

    mov = np.array([
        [mov0],
        [mov1],
        [mov2]
    ])

    mubar_t = mu_tp + F_x.T @ mov

    mubar_tx = mubar_t[0].item()
    mubar_ty = mubar_t[1].item()
    mubar_ttheta = mubar_t[2].item()

    gt = np.array([[0, 0, -mov1],
                   [0, 0,  mov0],
                   [0, 0,  0]])


    G_t = np.eye(3*N + 3) + F_x.T @ gt @ F_x

    Sigmabar_t = G_t @ Sigma_tp @ G_t.T + F_x.T @ R @ F_x


    ### perception update
    Q_t = np.diag(sigmas_percep)

    for i, z in enumerate(z_t.T):
        cit = c_t[i]

        loc = 3 + (cit-1)*3

        rit = z[0]
        phiit = z[1]
        sit = z[2]
        z = z[:, None]

        # line 9
        if cit not in seen_cits:
            mubar_jx = mubar_tx + rit * cos(phiit + mubar_ttheta)
            mubar_jy = mubar_ty + rit * sin(phiit + mubar_ttheta)
            mubar_js = sit

            mubar_t[loc] = mubar_jx
            mubar_t[loc+1] = mubar_jy
            mubar_t[loc+2] = mubar_js
            #
            # seen_cits.append(cit)
        else:
            mubar_jx = mubar_t[loc].item()
            mubar_jy = mubar_t[loc + 1].item()
            mubar_js = mubar_t[loc + 2].item()

        deltax = mubar_jx - mubar_tx
        deltay = mubar_jy - mubar_ty

        # line 12
        delta = np.array([[deltax],
                          [deltay]])
        # line 13
        q = (delta.T @ delta).item()

        # line 14
        zhat = np.array([[q**.5],
                         [smaller_arc_between_angles(mubar_ttheta, arctan2(deltay, deltax))],
                         [mubar_js]])

        # line 16
        sq = sqrt(q)
        Haux_pos = np.array([[-sq*deltax, -sq*deltay,   0],
                              [    deltay,    -deltax, -q],
                              [         0,          0,  0]]) / q

        Haux_map = np.array([[sq * deltax, sq * deltay, 0],
                             [    -deltay,      deltax, 0],
                             [          0,           0, q]]) / q

        Hit = np.hstack((Haux_pos, np.zeros((3, 3 * cit - 3)), Haux_map, np.zeros((3, 3 * (N - cit)))))

        # line 17
        Kit = Sigmabar_t @ Hit.T @ np.linalg.inv(Hit @ Sigmabar_t @ Hit.T + Q_t)

        # line 18
        mubar_t = mubar_t + Kit @ (z - zhat)

        # line 19
        # print(f"cit: {cit}, fit:{z}")
        # print('Sigmabar_t após da atualização', Sigmabar_t[loc:loc+3, loc:loc+3])
        Sigmabar_t = (np.eye(3 + 3*N) - Kit @ Hit) @ Sigmabar_t
        # print('Sigmabar_t antes da atualização', Sigmabar_t[loc:loc+3, loc:loc+3])
        # print(f"Kit @ Hit: {(Kit @ Hit)[loc:loc+3, loc:loc+3]}")
        # print()
        # print("===============================")

    # lines 21, 22, and 23.
    mu_t, Sigma_t = mubar_t, Sigmabar_t

    return mu_t, Sigma_t #, seen_cits
