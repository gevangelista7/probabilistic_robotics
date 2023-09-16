import numpy as np
from numpy import sin, cos, sqrt, arctan2
from utils import smaller_arc_between_angles, expand_sigma
from motion_model import sample_model_velocity

def UFK_localization(mu_tp, Sigma_tp, u_t, z_t, Nt, m, sigmas_percep,
                     alphas, gamma, beta=2, alpha_sig2=0.25, deltat=1):
    mu_tp_x = mu_tp[0].item()
    mu_tp_y = mu_tp[1].item()
    mu_tp_theta = mu_tp[2].item()
    v = u_t[0]
    w = u_t[1]

    Mt = np.diag((alphas[0]*v**2 + alphas[1]*w**2, alphas[2]*v**2 + alphas*w**2))

    Qt = np.diag((sigmas_percep[0]**2, sigmas_percep[1]**2))

    mu_a_tp = np.vstack((mu_tp, np.zeros((4, 1))))

    Sigma_a_tp = np.block([
        [Sigma_tp,         np.zeros((3, 2)), np.zeros((2, 2))],
        [np.zeros((3, 2)), Mt,               np.zeros((2, 2))],
        [np.zeros((3, 2)), np.zeros((2, 2)), Qt              ]
    ])

    chi_a_tp = np.hstack((mu_a_tp, mu_a_tp + gamma * Sigma_a_tp ** .5,  gamma * Sigma_a_tp ** .5))

    wm = np.array(((gamma**2-3)/gamma**2, 1/(2*gamma**2 - 4), 1/(2*gamma**2 - 4)))
    wc = np.array([wm[0] + (1 - alpha_sig2 + beta), 1/(2*gamma**2 - 4), 1/(2*gamma**2 - 4) ])

    chi_x_t = []

    for i, sigpoint in enumerate(chi_a_tp.T):
        u_pt = sigpoint[3:5] + u_t
        X_pt = sigpoint[:3]

        X_pt = sample_model_velocity(u_t=u_pt, x_tp=X_pt, deltat=deltat, alphas=np.zeros(6))
        chi_x_t.append(X_pt)

        Z_pt =

    chi_x_t = np.array(chi_x_t)

    mu_t_bar = (wm * chi_x_t).sum(axis=1)
    Sigma_t_bar = wc * (chi_x_t - mu_t_bar) @ (chi_x_t - mu_t_bar).T







