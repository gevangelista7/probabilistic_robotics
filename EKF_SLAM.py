import numpy as np
from numpy import sin, cos, sqrt, arctan2
from utils import smaller_arc_between_angles, expand_sigma


def EKF_SLAM(mu_tp, Sigma_tp, u_t, z_t, Nt, alpha_ML,
            R=None, deltat=1,
            sigmas_percep=(.1, .1, .1)):

    ### verify
    assert z_t.shape[0] == 3

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
        R = np.eye(3)/10

    ### Motion update
    F_x = np.hstack((np.eye(3), np.zeros((3, 3 * Nt))))

    mov0 = - r * sin(mu_tp_theta) + r * sin(mu_tp_theta + w * deltat)
    mov1 =   r * cos(mu_tp_theta) - r * cos(mu_tp_theta + w * deltat)
    mov2 = w * deltat

    # mov = np.array([
    #     [mov0],
    #     [mov1],
    #     [mov2]
    # ])

    # mubar_t = mu_tp + F_x.T @ mov
    mubar_tx = mu_tp_x + mov0
    mubar_ty = mu_tp_y + mov1
    mubar_ttheta = mu_tp_theta + mov2

    mubar_t = mu_tp
    mubar_t[0] = mubar_tx
    mubar_t[1] = mubar_ty
    mubar_t[2] = mubar_ttheta

    gt = np.array([[0, 0, -mov1],
                   [0, 0,  mov0],
                   [0, 0,  0]])

    G_t = np.eye(3*Nt + 3) + F_x.T @ gt @ F_x

    Sigmabar_t = G_t @ Sigma_tp @ G_t.T + F_x.T @ R @ F_x


    ### perception update
    Q_t = np.diag(sigmas_percep)

    for i, z in enumerate(z_t.T):
        rit = z[0]
        phiit = z[1]
        sit = z[2]
        z = z[:, None]

        mu_Nppx = mubar_tx + rit * cos(phiit + mubar_ttheta)
        mu_Nppy = mubar_ty + rit * sin(phiit + mubar_ttheta)
        mu_Npps = sit

        # line 9
        mu_nf = np.array([[mu_Nppx],
                          [mu_Nppy],
                          [mu_Npps]])

        # start cache
        pi_lkhd = np.zeros(Nt+1)
        Haux_pos_cache = []
        Haux_map_cache = []
        psi_cache = []
        zhat_cache = []

        # get the extended representation
        Sigmabar_ext = expand_sigma(Sigmabar_t)
        mubar_ext = np.vstack((mubar_t, mu_nf))
        for k in range(Nt+1):
            loc = 3 + k * 3

            mubar_kx = mubar_ext[loc].item()
            mubar_ky = mubar_ext[loc + 1].item()
            mubar_ks = mubar_ext[loc + 2].item()

            # line 11
            deltax = mubar_kx - mubar_tx
            deltay = mubar_ky - mubar_ty

            delta = np.array([[deltax],
                              [deltay]])

            # lines 12 and 13
            q = (delta.T @ delta).item()
            zhat = np.array([[q**.5],
                             [smaller_arc_between_angles(mubar_ttheta, arctan2(deltay, deltax))],
                             [mubar_ks]])

            # lines 14 and 15
            sq = sqrt(q)
            Haux_pos = np.array([[-sq*deltax, -sq*deltay,   0],
                                  [    deltay,    -deltax, -q],
                                  [         0,          0,  0]]) / q
            Haux_map = np.array([[sq * deltax, sq * deltay, 0],
                                 [    -deltay,      deltax, 0],
                                 [          0,           0, q]]) / q

            Hkt = np.zeros((3, 3 * (Nt + 1) + 3))
            Hkt[:, :3] = Haux_pos
            Hkt[:, 3+k:6+k] = Haux_map

            # lines 16 and 17
            psi_k = Hkt @ Sigmabar_ext @ Hkt.T + Q_t
            pi_lkhd[k] = (z - zhat).T @ psi_k @ (z - zhat)

            # put in cache
            psi_cache.append(psi_k)
            Haux_pos_cache.append(Haux_pos)
            Haux_map_cache.append(Haux_map)
            zhat_cache.append(zhat)

        # line 19
        pi_lkhd[Nt] = alpha_ML

        # line 20
        j = np.argmin(pi_lkhd)


        # print(f'Início iter: Nt={Nt}, i={i}, k={k}, j={j} Sigmbar_t.shape={Sigmabar_t.shape},  pi={pi_lkhd}')
        # print(f'Nt={Nt}, j+1={j+1} , max(Nt, j+1)={max(Nt, j+1)}, Condition={(j+1) == Nt}')
        # line 21
        N0 = Nt
        Nt = max(Nt, j+1)

        if not Nt == N0:
            Sigmabar_t = Sigmabar_ext
            mubar_t = mubar_ext


        # get elements from cache
        Haux_pos = Haux_pos_cache[j]
        Haux_map = Haux_map_cache[j]
        psi_j = psi_cache[j]
        zhatj = zhat_cache[j]

        # prepare the Hjt
        Hjt = np.zeros((3, 3 * Nt + 3))
        Hjt[:, :3] = Haux_pos
        Hjt[:, 3 + j:6 + j] = Haux_map

        # line 22
        print(f"Nt={Nt}, Sigmabar_t: {Sigmabar_t.shape}. Hjt.T: {Hjt.T.shape}, np.linalg.inv(psi_j): {psi_j.shape}")
        Kit = Sigmabar_t @ Hjt.T @ np.linalg.inv(psi_j)

        # line 23
        mubar_t = mubar_t + Kit @ (z - zhatj)

        # line 24
        # print(f"fit:{z}")
        # print('Sigmabar_t após da atualização', Sigmabar_t[-3:,-3:])
        Sigmabar_t = (np.eye(3 + 3*Nt) - Kit @ Hjt) @ Sigmabar_t
        # print('Sigmabar_t antes da atualização', Sigmabar_t[-3:, -3:])
        # print(f"Kit @ Hit: {(Kit @ Hjt)[-3:,-3:]}")
        # # print(psi_j)
        # # print(f'Final iter: Nt={Nt}, i={i}, k={k}, j={j} Sigmbar_t.shape={Sigmabar_t.shape},  pi={pi_lkhd}')
        # print("===============================")

    # lines 26 and 27
    mu_t, Sigma_t = mubar_t, Sigmabar_t

    return mu_t, Sigma_t, Nt

