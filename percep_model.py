from copy import deepcopy
import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, ConnectionPatch
import shapely.geometry as shp
import seaborn as sns
from utils import prob, normalize_angle, relative_bearing, Map


def landmark_model_known_correspondence(fit, cit, xt, m, sigmas):
    """
    inputs:
        cit: int -> indice do landmark no mapa.
        xt: 3 x S -: São tratadas S amostras. Cada uma correspondente a uma coluna [x, y, theta]^T.
        fit: 3 x 1 -> A leitura do landmark corresponde a colunas [r, phi, s].T.
        sigmas: 3 x 1 -> são obtidas os desvios padrão para cada uma das medições relativas dos landmarks.
        m: Map -> mapa do cenário conforme classe implementada.

    output:
        q: 1 x S -> vetor com as likelihoods de cada uma das amostras considerando a leitura realizada.
    """
    assert type(m) == Map
    mjx, mjy = m.landmarks[cit]  # lookup da posição do landmark no map

    x = xt[0, :]  # estado hipotético do robô
    y = xt[1, :]
    theta = xt[2, :]

    rit = fit[0]
    phiit = fit[1]
    sit = fit[2]                    # não utlizado -> considerando a leitura perfeita dos atributos.

    sigma_r = sigmas[0]             # obtenção variância de r e phi
    sigma_phi = sigmas[1]
    sigma_s = sigmas[2]             # não utlizado -> considerando a leitura perfeita dos atributos.

    rhat = sqrt((mjx - x) ** 2 + (mjy - y) ** 2)

    phihat = np.zeros_like(theta)
    for i, theta_i in enumerate(theta):
        phihat[i] = relative_bearing(observer_angle=theta_i,
                                     target_angle=arctan2(mjy - y[i], mjx - x[i]))

    # phihat = arctan2(mjy - y, mjx - x) - theta

    q = prob(rit - rhat, sigma_r) * prob(phiit - phihat, sigma_phi)

    return q
