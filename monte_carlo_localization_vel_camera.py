import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, ConnectionPatch
import shapely.geometry as shp
import seaborn as sns
from scipy.stats import norm

from utils import stochastic_universal_sampler, Map, normalize_angle, HGrid, PGrid, get_random_samples
from motion_model import motion_model_velocity, sample_model_velocity
from percep_model import landmark_model_known_correspondence
from plot_utils import plot_robot_v3, plot_MCL_scene, plot_robots_in_map

np.random.seed(7)

def monte_carlo_localization_vel_camera(X_tp, u_t, m,
                                        cit, fit, percep_sigmas,
                                        motion_alphas, motion_deltat=1, mult_samples=1):
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

    # aplicação do modelo de velocidade para todas as amostras.
    X_t = sample_model_velocity(u_t=u_t, x_tp=X_tp, n_samples=mult_samples, alphas=motion_alphas, deltat=motion_deltat)

    # atribuição dos pesos para cada uma das amostras. Os pesos são iniciados todos unitários e,
    # para cada landmark detectado, os pesos multiplicados pela likelihood obtida com o modelo de percepção.
    w_t = np.ones_like(X_t[0])
    for i in range(len(cit)):
        w_t = w_t * landmark_model_known_correspondence(fit=fit.T[i], cit=cit[i], m=m, xt=X_t, sigmas=percep_sigmas)

    # os indices das amostras são obtidas pelo SUS.
    survivors_index = stochastic_universal_sampler(w_array=w_t, n_samples=n_samples)

    return X_t[:, survivors_index]