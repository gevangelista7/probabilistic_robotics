import numpy as np
from numpy import sqrt, pi
from scipy.stats import norm

from utils import stochastic_universal_sampler, Map, normalize_angle, HGrid
from motion_model import sample_model_velocity
from percep_model import landmark_model_known_correspondence


def KLD_Sampling_MCL_vel_camera(X_tp, u_t, m,
                                cit, fit, percep_sigmas,
                                motion_alphas, motion_deltat=1,
                                M_X_min=100, eps=0.05, delta=0.1,
                                min_scales=(0, 0, -pi), max_scales=(15, 15, pi), res=(0.5, 0.5, pi/18)):
    """
    input:
        X_t: np.array, 3 x S -> São tratadas S amostras. Os estados são colunas [x, y, theta].T dispostas lado a lado.
        u_: np.array, 2 x 1 -> controle considerando velocidade   componentes: [v, w]
        m: Map

        # perception model
        cit: list, L -> lista dos identificadores dos landmarks no mapa.
        fit: np.array, 2 x L -> A leitura do landmark corresponde a colunas [r, phi].T.
        percep_sigmas: 3 x 1 -> desvios padrão das medições do modelo de percepção.

        # motion model
        motion_alphas:
        motion_dt

    output:
        p np.array, 1 x S -> likelihood de cada uma das amostras
    """
    assert type(m) == Map
    _, n_samples = X_tp.shape

    w_tp = X_tp[3]
    X_t = np.array([[], [], [], []])
    H = HGrid(min_scales, max_scales, res)
    M, M_x, k = 0, 0, 0
    min_x, max_x, min_y, max_y = m.get_map_limits()

    while M < M_x or M < M_X_min:
        i_p = stochastic_universal_sampler(w_tp, 1)
        x_M = sample_model_velocity(u_t=u_t, x_tp=X_tp[:, i_p], n_samples=1,
                                    alphas=motion_alphas, deltat=motion_deltat)
        x_M[0] = np.clip(x_M[0], a_min=min_x, a_max=max_x)
        x_M[1] = np.clip(x_M[1], a_min=min_y, a_max=max_y)
        x_M[2] = normalize_angle(x_M[2])

        w_t = np.ones(1)
        for i in range(len(cit)):
            w_t = w_t * landmark_model_known_correspondence(fit=fit.T[i], cit=cit[i], m=m, xt=x_M, sigmas=percep_sigmas)

        x_M = np.vstack((x_M, w_t))
        X_t = np.hstack((X_t, x_M))
        M += 1

        if H.value[H.state_to_idx(x_M)] == 0:
            k += 1
            H.value[H.state_to_idx(x_M)] += 1
            if k > 1:
                zdelta = norm.ppf(1 - delta)
                M_x = (k-1)/2/eps*(1 - 2 / 9 / (k - 1) + sqrt(2 / 9 / (k - 1)) * zdelta)**3

    print(f"KLDMCL iter end M: {M}, M_x: {M_x}, M_X_min: {M_X_min}")
    return X_t
