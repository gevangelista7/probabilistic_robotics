from copy import deepcopy
import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, ConnectionPatch
import shapely.geometry as shp
import seaborn as sns
from utils import prob, normalize_angle, relative_bearing, Map

def sample_model_velocity(u_t, x_tp, alphas, deltat=1):
    """
        input:
            x_tp -> estado em tempo previo            componentes: [x, y, theta, n]
            u_t -> controle considerando velocidade   componentes: [v, w]
        output:
            x_t -> estado em tempo corrente.          componentes: [xl, yl, thetal, n * n_sample]
    """
    x = x_tp[0, :]
    y = x_tp[1, :]
    theta = x_tp[2, :]

    v = u_t[0]
    w = u_t[1]

    # x = np.tile(x, (1, n_samples))
    # y = np.tile(y, (1, n_samples))
    # theta = np.tile(theta, (1, n_samples))

    vhat = np.random.normal(v, (alphas[0] * v ** 2 + alphas[1] * w ** 2) ** .5)
    what = np.random.normal(w, (alphas[2] * v ** 2 + alphas[3] * w ** 2) ** .5)
    gammahat = np.random.normal(np.zeros_like(theta), (alphas[4] * v ** 2 + alphas[5] * w ** 2) ** .5)

    if abs(what) < 1e-6:
        what += 1e-6

    rhat = vhat / what
    thetahat = theta + what * deltat

    rhat = np.clip(rhat, -1e8, 1e8)

    xl = x - rhat * (sin(theta) - sin(thetahat))
    yl = y + rhat * (cos(theta) - cos(thetahat))
    thetal = thetahat + gammahat * deltat

    res = np.vstack((xl, yl, thetal))

    return res


def motion_model_velocity(x_t, u_t, x_tp, alphas, deltat):
    """
      input:
        x_t -> estado em tempo corrente.          componentes: [xl, yl, thetal]
        x_tp -> estado em tempo previo            componentes: [x, y, theta]
        u_t -> controle considerando velocidade   componentes: [v, w]
      output:
        p -> distribuiçao (p_1 * p_2 * p_3 )
    """
    xl = x_t[0, :]
    yl = x_t[1, :]
    thetal = x_t[2, :]

    x = x_tp[0, :]
    y = x_tp[1, :]
    theta = x_tp[2, :]

    v = u_t[0]
    w = u_t[1]

    if abs(((y - yl) * cos(theta) - (x - xl) * sin(theta))) < 1e-6:
        mu = 1e5
    else:
        mu = 0.5 * ((x - xl) * cos(theta) + (y - yl) * sin(theta)) / ((y - yl) * cos(theta) - (x - xl) * sin(theta))

    xstar = (x + xl) / 2 + mu * (y - yl)
    ystar = (y + yl) / 2 + mu * (xl - x)
    rstar = sqrt((x - xstar) ** 2 + (y - ystar) ** 2)

    # deltatheta = np.arctan2(yl - ystar, xl - xstar) - np.arctan2(y - ystar, x - xstar)

    deltatheta = relative_bearing(observer_angle=np.arctan2(yl - ystar, xl - xstar),
                                  target_angle=np.arctan2(y - ystar, x - xstar))

    what = deltatheta / deltat
    vhat = what * rstar
    gammahat = (thetal - theta) / deltat - what

    p1 = prob(v - vhat, alphas[0] * v ** 2 + alphas[1] * w ** 2)
    p2 = prob(w - what, alphas[2] * v ** 2 + alphas[3] * w ** 2)
    p3 = prob(gammahat, alphas[4] * v ** 2 + alphas[5] * w ** 2)

    return p1 * p2 * p3


