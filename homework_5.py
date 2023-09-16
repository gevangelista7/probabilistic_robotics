import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
from utils import stochastic_universal_sampler, Map, normalize_angle, HGrid, PGrid, smaller_arc_between_angles, clear_directory
from motion_model import motion_model_velocity, sample_model_velocity
from plot_utils import plot_robot_v3, plot_MCL_scene, plot_scene_with_confidence, make_movie

from EKF_SLAM import EKF_SLAM
from EKF_SLAM_known_correspondences import EKF_SLAM_known_correspondences


np.random.seed(7)

obstacles = [[(0, 0), (30, 0), (30, 0.01), (0, .01)],  # paredes
             [(0, 0), (0, 16), (.01, 16), (.01, 0)],
             [(0, 15.99), (0, 15.99), (30, 15.99), (30, 16)],
             [(29.99, 16), (30, 16), (30, 0), (29.99, 0)]]

landmarks = {1: (0, 16), 2: (2, 16), 3: (12, 16), 4: (15, 16), 5: (17, 16), 6: (30, 16),
             7: (0, 14), 8: (30, 14),
             9: (4, 12), 10: (6, 12), 11: (8, 12), 12: (10, 12), 13: (12, 12), 14: (14, 12), 15: (18, 12), 16: (20, 12), 17: (24, 12),
             18: (0, 8),
             19: (17, 7),
             20: (2, 6),
             21: (15, 5), 22: (19, 5),
             23: (0, 4), 24: (4, 4),
             25: (17, 3),
             26: (2, 2), 27: (30, 2),
             29: (30, 1),
             28: (4, 0), 30: (30, 0), 31: (10, 0)}

area_limit = [(-1, -1), (-1, 17), (31, 17), (31, -1)]

test_area_map = Map(landmarks=landmarks, obstacles=obstacles, area_limit=area_limit)
# test_area_map.plot_map()

camera_fov = pi / 2     # 90 degree centrado em theta
camera_range = 7        # meters para reconhecimento adequado
percep_sigmas = (0.1, 0.01, 0.01)
hr = -pi / 2  # comando de rot horario
ah = pi / 2  # comando de rot anti-horario


### CONTROL PANEL ###

# algo_name = 'EKF_SLAM_kc'
algo_name = 'EKF_SLAM'
xt0 = np.array([[2], [1], [pi / 2]])

ut_seq = np.array(
    ([0] + 12 * [1] + [0] + [1]  + 23 * [1] + [1]  + 8 * [1] + 1 * [1]  + 19 * [1] + 10 * [0] + 1 * [0]  + 3 * [1] + 10 * [0],
     [0] + 12 * [0] + [0] + [hr] + 23 * [0] + [hr] + 8 * [0] + 1 * [hr] + 19 * [0] + 10 * [0] + 1 * [hr] + 3 * [0] + 10 * [0]))
    # subida                # direita         # descida            # esquerda                       # subida

### CONTROL PANEL ###
if __name__ == "__main__":


    N_ldmk = len(test_area_map.landmarks)
    if algo_name == 'EKF_SLAM_kc':
        N = N_ldmk
        Sigma_t = np.diag([0]*3 + [1e8]*3*N)
        mu_t = np.vstack((xt0, np.zeros(3 * N)[:, None]))
    elif algo_name == 'EKF_SLAM':
        N = 0
        Sigma_t = np.diag([0]*3)
        mu_t = xt0
    else: raise 'algo_name err'

    xti = mu_t[:3]
    seen_cits = []

    dir = f'./movies/{algo_name}_movie/'
    clear_directory(dir)

    plot_scene_with_confidence(xt=xti, mu=mu_t, Sigma=Sigma_t, seen_cits=seen_cits, area_map=test_area_map,
                               camera_on=True, camera_fov=camera_fov, camera_range=camera_range,
                               title=f"Condição inicial. Existem {N_ldmk} landmarks...", save=True,
                               figname=dir+'iter_{0:03d}'.format(0))

    for i, ut in enumerate(ut_seq.T):
        xti = sample_model_velocity(u_t=ut, x_tp=xti, alphas=np.zeros(6), deltat=1)
        citi, fiti = test_area_map.get_visual_landmark(xti, camera_fov=camera_fov,
                                                       camera_range=camera_range,
                                                       sigma_cam=percep_sigmas)

        # algorithm execution
        if algo_name == 'EKF_SLAM_kc':
            mu_t, Sigma_t = EKF_SLAM_known_correspondences(mu_tp=mu_t, Sigma_tp=Sigma_t,
                                                           u_t=ut, z_t=fiti, c_t=citi,
                                                           N=N, seen_cits=seen_cits,
                                                           sigmas_percep=percep_sigmas)

            if len(citi) > 0:
                ldmks_string = f'Ldmks {citi} detected!'
            else:
                ldmks_string = 'No Ldmks detected...'
            title = f"Uncertainty update iter={i}. " + ldmks_string

        elif algo_name == 'EKF_SLAM':
            mu_t, Sigma_t, N = EKF_SLAM(mu_tp=mu_t, Sigma_tp=Sigma_t,
                                        u_t=ut, z_t=fiti, Nt=N,
                                        alpha_ML=1,
                                        R=np.eye(3)/10, deltat=1,
                                        sigmas_percep=percep_sigmas)

            if len(citi) > 0:
                ldmks_string = f'{len(citi)} ldmks detected!'
            else:
                ldmks_string = 'No Ldmks detected...'
            title = f"Uncertainty update i={i}, N={N}, {ldmks_string}"

        for c in citi:
            if c not in seen_cits:
                seen_cits.append(c)

        plot_scene_with_confidence(xt=xti, mu=mu_t, Sigma=Sigma_t, seen_cits=seen_cits, area_map=test_area_map,
                                   camera_on=True, camera_fov=camera_fov, camera_range=camera_range,
                                   title=title, save=True, figname=dir+'iter_{0:03d}'.format(i))

        print(f'Fim da iteração {i} de {len(ut_seq.T)-1}')

    make_movie(dir, f'./movies/{algo_name}_movie.gif')



