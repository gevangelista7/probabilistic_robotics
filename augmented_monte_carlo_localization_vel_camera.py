import numpy as np
from numpy import pi
from utils import stochastic_universal_sampler, Map
from motion_model import sample_model_velocity
from percep_model import landmark_model_known_correspondence


def augmented_monte_carlo_localization_vel_camera(X_tp, u_t, m,
                                                  cit, fit, percep_sigmas,
                                                  motion_alphas, motion_deltat=1, mult_samples=1,
                                                  wslow=.1, wfast=.1, alphaslow=0.01, alphafast=10.):
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
    min_x, max_x, min_y, max_y = m.get_map_limits()

    _, n_samples = X_tp.shape

    # amostra a partir das amostras de t-1 com o modelo de movimento.
    X_t = sample_model_velocity(u_t=u_t, x_tp=X_tp, n_samples=mult_samples,
                                alphas=motion_alphas, deltat=motion_deltat)

    # likelihood das amostras advindas do modelo de movimento.
    w_t = np.ones_like(X_t[0])
    for i in range(len(cit)):
        w_t = w_t * landmark_model_known_correspondence(fit=fit.T[i], cit=cit[i], m=m,
                                                        xt=X_t, sigmas=percep_sigmas)

    X_t = np.vstack((X_t, w_t))

    # calculo da dinâmica dos pesos e número de amostras esperado,
    # regulado pelo número original de amostras
    wavg = w_t.mean()
    wslow = wslow + alphaslow * (wavg - wslow)
    wfast = wfast + alphafast * (wavg - wfast)

    n_rand_samp = int(max(0, 1 - wfast / wslow) * n_samples)
    n_rand_samp = min(n_rand_samp, n_samples)

    # caso seja necssário, são amostradas aleatóriamente o número de amostras
    # preconizado pelo algoritmo
    if n_rand_samp > 0:
        rand_x = np.random.uniform(min_x, max_x, n_rand_samp)
        rand_y = np.random.uniform(min_y, max_x, n_rand_samp)
        rand_theta = np.random.uniform(-pi, pi, n_rand_samp)
        X_rand = np.vstack((rand_x, rand_y, rand_theta))

        # pesos das amostras aleatórias.
        w_rand = np.ones_like(X_rand[0])
        for i in range(len(cit)):
            w_rand = w_rand * landmark_model_known_correspondence(fit=fit.T[i], cit=cit[i], m=m,
                                                                  xt=X_rand, sigmas=percep_sigmas)

        X_rand = np.vstack((X_rand, w_rand))

        # unem-se as amostras aleatórias e do modelo de movimento
        X_t = np.hstack((X_t, X_rand))
        w_t = X_t[3, :]

    # selecionam-se os sobreviventes entra as amostras do modelo e as aleatórias
    survivors_index = stochastic_universal_sampler(w_array=w_t, n_samples=n_samples)

    X_t = X_t[:3, survivors_index]

    return X_t, wslow, wfast