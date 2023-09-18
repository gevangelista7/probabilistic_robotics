import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
from scipy.stats import norm

from utils import stochastic_universal_sampler, Map, normalize_angle, HGrid, PGrid
from motion_model import motion_model_velocity, sample_model_velocity
from percep_model import landmark_model_known_correspondence
from plot_utils import plot_robot_v3, plot_MCL_scene

from augmented_monte_carlo_localization_vel_camera import augmented_monte_carlo_localization_vel_camera
from KLD_Sampling_MCL_vel_camera import KLD_Sampling_MCL_vel_camera
from grid_localization import grid_localization
from monte_carlo_localization_vel_camera import monte_carlo_localization_vel_camera

np.random.seed(7)

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
motion_alphas = np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
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

wslow = .1
wfast = 10.

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

        ### ======================= Aug-MCL ======================= ###
        Xt, wslow, wfast = augmented_monte_carlo_localization_vel_camera(X_tp=Xt, u_t=uti, m=test_area_map,
                                                                         cit=citi, fit=fiti,
                                                                         percep_sigmas=percep_sigmas,
                                                                         motion_alphas=motion_alphas, mult_samples=1,
                                                                         wslow=wslow,
                                                                         wfast=wfast,
                                                                         alphaslow=0.1,
                                                                         alphafast=.8)
        meanx = Xt[0].mean()
        meany = Xt[1].mean()
        dist = ((xti_real[0] - meanx) ** 2 + (xti_real[1] - meany) ** 2) ** .5
        augMCL_dist_mean_real.append(dist)
        ### ======================= Aug-MCL ======================= ###






        i += 1
        plot_MCL_scene(Xtp, Xt, xt_real, i, camera_range, camera_fov)
        print('ut:', uti)
