import numpy as np
from numpy import sin, cos, sqrt, arctan2
from utils import smaller_arc_between_angles


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
    # line 2 -> F_x not used
    # F_x = np.hstack((np.eye(3), np.zeros((3, 3 * N))))

    mov0 = - r * sin(mu_tp_theta) + r * sin(mu_tp_theta + w * deltat)
    mov1 =   r * cos(mu_tp_theta) - r * cos(mu_tp_theta + w * deltat)
    mov2 = w * deltat

    # line 3
    # mubar_t = mu_tp + F_x.T @ mov
    mubar_tx = mu_tp_x + mov0
    mubar_ty = mu_tp_y + mov1
    mubar_ttheta = mu_tp_theta + mov2

    mubar_t = mu_tp
    mubar_t[0] = mubar_tx
    mubar_t[1] = mubar_ty
    mubar_t[2] = mubar_ttheta

    # line 4
    # gt = np.array([[0, 0, -mov1],
    #                [0, 0,  mov0],
    #                [0, 0,  0]])
    # G_t = np.eye(3 * N + 3) + F_x.T @ gt @ F_x
    G_t = np.eye(3*N + 3)
    G_t[0, 2] = -mov1
    G_t[1, 2] = mov0

    # line 5
    # Sigmabar_t = G_t @ Sigma_tp @ G_t.T + F_x.T @ R @ F_x
    R_ext = np.zeros((3*N + 3, 3*N + 3))
    R_ext[:3, :3] = R
    Sigmabar_t = G_t @ Sigma_tp @ G_t.T + R_ext

    # line 6
    ### perception update
    Q_t = np.diag(sigmas_percep)

    # lines 7 and 8
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

        # to do: cut the zeros multiply
        Hit = np.hstack((Haux_pos, np.zeros((3, 3 * cit - 3)), Haux_map, np.zeros((3, 3 * (N - cit)))))

        # line 17
        Kit = Sigmabar_t @ Hit.T @ np.linalg.inv(Hit @ Sigmabar_t @ Hit.T + Q_t)

        # line 18
        mubar_t = mubar_t + Kit @ (z - zhat)

        # line 19
        Sigmabar_t = (np.eye(3 + 3*N) - Kit @ Hit) @ Sigmabar_t

    # lines 21, 22, and 23.
    mu_t, Sigma_t = mubar_t, Sigmabar_t

    return mu_t, Sigma_t
