import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, ConnectionPatch
import shapely.geometry as shp
import seaborn as sns
from scipy.stats import norm

from utils import stochastic_universal_sampler, Map, normalize_angle, HGrid, PGrid
from motion_model import motion_model_velocity, sample_model_velocity
from percep_model import landmark_model_known_correspondence
from plot_utils import plot_robot_v3, plot_MCL_scene

np.random.seed(7)


def grid_localization(ptp, ut, fit, cit, m, motion_alphas, motion_dt, percep_sigmas):
    """
    ptp: p_grid: p para um conjunto específicos de indices para (x, y, e theta)
    u_t -> controle considerando velocidade   componentes: [v, w]
    z_t -> foi substituído por fit e cit, pois será utlizado o landmark_model_known_correspondence para medição

    cit: int -> indice do landmark no mapa.
    fit: 2 x 1 -> A leitura do landmark corresponde a colunas [r, phi].T.
    sigmas: 3 x S -> são obtidas os desvios padrão para cada uma das medições relativas dos landmarks.
    """
    assert type(ptp) == PGrid
    assert type(m) == Map

    # inicializa-se o grid de probabilidades do tempor final da iteração
    pt = PGrid(ptp.min_scales, ptp.max_scales, ptp.res)

    # para cada posição do p_grid é realizado o somatório do algorigmo,
    # pkt é um acumulador para receber o somatório das likelihoods do modelo
    # ponderadas pela probabilidade da iteração anterior.
    for k in pt.value:
        pkt = 0
        # k = (i_x_t, i_y_t, i_theta_t)
        # o for a seguir vai iterandos sobre as parcelas da soma
        for i in ptp.value:
            # i = (i_x_tp, i_y_tp, i_theta_tp)

            # cálculo dos centros das células para cada um dos indices.
            xk_mean = pt.cell_center(*k)
            xi_mean = ptp.cell_center(*i)

            # aplicação do modelo de movimento ponderado pelo p_grid prévio.
            # desta forma realiza-se se o modelo de movimento
            # partindo de cada uma das células para cada uma das células
            pkt += ptp.value[i] * motion_model_velocity(x_t=xk_mean, u_t=ut, x_tp=xi_mean,
                                                        alphas=motion_alphas, deltat=motion_dt)

        # após o término do somatório o valor é armazenado no p_grid de tempo presente.
        pt.value[k] = pkt

    pt.plot_p_grid("Distribuição considerando apenas o motion_model_velocity")
    # incorporando a medição:
    # novamente itera-se por todas as células do grid, obtém-se novamente os centros das células e
    # obtém-se as likelihoods considerando a medição passada. foi utilizado um modelo de câmera e com isso
    # o modelo de percepção utlilizado foi o landmark_model_known_correspondence.
    for cit_idx in range(len(cit)):
        for k in pt.value:
            # k = (i_x_t, i_y_t, i_theta_t)

            xk_mean = pt.cell_center(*k)

            pt.value[k] = pt.value[k] * \
                          landmark_model_known_correspondence(fit=fit.T[cit_idx], cit=cit[cit_idx], m=m, xt=xk_mean,
                                                              sigmas=percep_sigmas)

    # normalização: inicialmente obtém-se o fator eta e depois itera-se sobre todas as células multiplicando-o.
    eta = 1 / sum([*pt.value.values()])
    for k in pt.value:
        pt.value[k] = eta * pt.value[k]

    return pt



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

    X_t = sample_model_velocity(u_t=u_t, x_tp=X_tp, n_samples=mult_samples, alphas=motion_alphas, deltat=motion_deltat)
    w_t = np.ones_like(X_t[0])
    for i in range(len(cit)):
        w_t = w_t * landmark_model_known_correspondence(fit=fit.T[i], cit=cit[i], m=m, xt=X_t, sigmas=percep_sigmas)

    survivors_index = stochastic_universal_sampler(w_array=w_t, n_samples=n_samples)

    return X_t[:, survivors_index]


def augmented_monte_carlo_localization_vel_camera(X_tp, u_t, m,
                                                  cit, fit, percep_sigmas,
                                                  motion_alphas, motion_deltat=1, mult_samples=1,
                                                  wslow=.1, wfast=.1, alphaslow=0.01, alphafast=10):
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

    X_t = sample_model_velocity(u_t=u_t, x_tp=X_tp, n_samples=mult_samples, alphas=motion_alphas, deltat=motion_deltat)
    w_t = np.ones_like(X_t[0])
    for i in range(len(cit)):
        w_t = w_t * landmark_model_known_correspondence(fit=fit.T[i], cit=cit[i], m=m, xt=X_t, sigmas=percep_sigmas)

    wavg = w_t.mean()
    wslow = wslow + alphaslow * (wavg - wslow)
    wfast = wfast + alphafast * (wavg - wfast)

    n_rand_samp = int(max(0, 1 - wfast / wslow) * n_samples)
    n_rand_samp = min(n_rand_samp, n_samples)

    # get the random samples
    rand_x = np.random.uniform(min_x, max_x, n_rand_samp)
    rand_y = np.random.uniform(min_y, max_x, n_rand_samp)
    rand_theta = np.random.uniform(-pi, pi, n_rand_samp)
    rand_samples = np.vstack((rand_x, rand_y, rand_theta))

    if n_rand_samp < n_samples:
        survivors_index = stochastic_universal_sampler(w_array=w_t, n_samples=n_samples-n_rand_samp)
    else:
        survivors_index = []

    X_t = np.hstack((rand_samples, X_t[:, survivors_index]))

    return X_t


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




landmarks = {1: (0, 0), 2: (0, 3), 3: (0, 6),
             4: (0, 9), 5: (0, 12), 6: (0, 15),
             7: (15, 0), 8: (15, 3), 9: (15, 6),
             10: (15, 9), 11: (15, 12), 12: (15, 15),
             13: (5, 0), 14: (5, 3), 15: (5, 6),
             16: (5, 9), 17: (5, 12), 18: (5, 15),
             19: (10, 0), 20: (10, 3), 21: (10, 6),
             22: (10, 9), 23: (10, 12), 24: (10, 15)}
area_limit = [(0, 0), (0, 15), (15, 15), (15, 0)]
obstacles = [[(0, 0), (15, 0), (15, 0.1), (0, .1)],  # paredes
             [(0, 0), (0, 15), (.1, 15), (.1, 0)],
             [(0, 14.9), (0, 14.9), (15, 14.9), (15, 15)],
             [(14.9, 15), (15, 15), (15, 0), (14.9, 0)]]

test_area_map = Map(landmarks=landmarks, obstacles=obstacles, area_limit=area_limit)

camera_fov = pi / 2  # 90 degree centrado em theta
camera_range = 5  # meters para reconhecimento adequado
motion_alphas = np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1)) * 0
percep_sigmas = (.1, .1, 0.01)

#### ut
ut = np.array([[0, 2, 5, 0, 0, 3, 0, 4],
               [0, pi / 2, 0, pi / 2, -pi / 2, 0, -pi / 2, 0]])

xt_real0 = np.array([[2], [2], [0]])

### comparativo distâncias médias dos três métodos de MCL
vanMCL_dist_mean_real = []
augMCL_dist_mean_real = []
KLDMCL_dist_mean_real = []
KLD_n_samples = []

xt_real = xt_real0

xti_real = xt_real0[:, -1:]
### cenário real:
for uti in ut.T:
    # move and read camera
    xti_real = sample_model_velocity(u_t=uti.T, x_tp=xti_real, n_samples=1, deltat=1, alphas=np.zeros(6))
    real_robot = plot_robot_v3(xti_real.T[0], color='red', zorder=3)
    xt_real = np.hstack((xt_real, xti_real))

# plot_robots_in_map(xt_real, test_area_map, real_robot, title=f'Cenário Real')


X0 = []
xti_real = xt_real[:, 0]
min_x, max_x, min_y, max_y = test_area_map.get_map_limits()
while len(X0) < 500:
    x = np.random.uniform(low=min_x, high=max_x)
    y = np.random.uniform(low=min_y, high=max_y)
    theta = np.random.uniform(-pi / 2, pi / 2)
    if not test_area_map.point_is_occupied((x, y)):
        X0.append([x, y, theta])

Xt = np.array(X0).T
w0 = np.ones_like(Xt[0])
Xt = np.vstack((Xt, w0))

if __name__ == '__main__':
    i = 0
    for uti in ut.T:
        Xtp = Xt

        # read camera on with the real position
        xti_real = xt_real[:, i]
        citi, fiti = test_area_map.get_visual_landmark(xti_real, camera_fov=camera_fov,
                                                       camera_range=camera_range, sigma_cam=percep_sigmas)

        ### ========================= MCL ========================= ###
        Xt = monte_carlo_localization_vel_camera(X_tp=Xt, u_t=uti, m=test_area_map,
                                                 cit=citi, fit=fiti, percep_sigmas=percep_sigmas,
                                                 motion_alphas=motion_alphas, mult_samples=1)

        meanx = Xt[0].mean()
        meany = Xt[1].mean()
        dist = ((xti_real[0] - meanx) ** 2 + (xti_real[1] - meany) ** 2) ** .5
        vanMCL_dist_mean_real.append(dist)
        ### ========================= MCL ========================= ###

        i += 1
        plot_MCL_scene(Xtp, Xt, xt_real, i, camera_range, camera_fov)
        print('ut:', uti)
