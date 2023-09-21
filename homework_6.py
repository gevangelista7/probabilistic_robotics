import numpy as np
from numpy import pi
from utils import Map, clear_directory, get_random_samples, init_SLAM_population
from motion_model import sample_model_velocity
from plot_utils import plot_FastSLAM_scene, make_movie

from FastSLAM_1 import FastSLAM_1
# from FastSLAM_2 import FastSLAM_2


np.random.seed(7)

obstacles = [[(0, 0), (30, 0), (30, 0.01), (0, .01)],  # just walls
             [(0, 0), (0, 16), (.01, 16), (.01, 0)],
             [(0, 15.99), (0, 15.99), (30, 15.99), (30, 16)],
             [(29.99, 16), (30, 16), (30, 0), (29.99, 0)]]
landmarks = {1: (0, 16), 2: (2, 16), 3: (12, 16), 4: (15, 16), 5: (17, 16), 6: (30, 16),    # ordered by y decreasing
             7: (0, 14), 8: (30, 14),
             9: (4, 12), 10: (6, 12), 11: (8, 12), 12: (10, 12), 13: (12, 12),              # next line also with y=12
             14: (14, 12), 15: (18, 12), 16: (20, 12), 17: (24, 12),
             18: (0, 8),
             19: (17, 7),
             20: (2, 6),
             21: (15, 5), 22: (19, 5),
             23: (0, 4), 24: (4, 4),
             25: (17, 3),
             26: (2, 2), 27: (30, 2),
             29: (30, 1),
             28: (4, 0), 30: (30, 0), 31: (10, 0)}
area_limit = [(-1, -1), (-1, 17), (31, 17), (31, -1)]       # some extra space to better show the ellipses

test_area_map = Map(landmarks=landmarks, obstacles=obstacles, area_limit=area_limit)

hr = -pi / 2            # clockwise rot command
ah = pi / 2             # anticlockwise rot command

camera_fov = pi / 2     # 90 degree theta centered
camera_range = 7        # meters for proper recognize


# == CONTROL PANEL == #

algo_name = 'FastSLAM_1'
# algo_name = 'FastSLAM_2'
xt0 = np.array([[2], [1], [pi / 2]])

ut_seq = np.array(
    ([0] + 12 * [1] + [0] + [1]  + 23 * [1] + [1]  + 8 * [1] + 1 * [1]  + 19 * [1] + 10 * [0] + 1 * [0]  + 3 * [1] + 10 * [0],
     [0] + 12 * [0] + [0] + [hr] + 23 * [0] + [hr] + 8 * [0] + 1 * [hr] + 19 * [0] + 10 * [0] + 1 * [hr] + 3 * [0] + 10 * [0]))
    # subida                # direita         # descida            # esquerda                       # subida

n_particles = 100
percep_sigmas = (0.3, 0.3, 0.1)
motion_alphas = np.array((0.01, 0.001, 0.01, 0.001, 0.01, 0.001))
p0 = 1e-1

# good values for visualization:
# percep_sigmas = (0.3, 0.01, 0.1)
# R = np.eye(3)/100

# == CONTROL PANEL == #
if __name__ == "__main__":

    X = np.tile(xt0, n_particles)
    Y = init_SLAM_population(X[:3])
    xti = xt0

    images_dir = f'./movies/{algo_name}_movie/'
    clear_directory(images_dir)

    for i, ut in enumerate(ut_seq.T):
        # real robot for plot and camera reading
        xti = sample_model_velocity(u_t=ut, x_tp=xti, alphas=np.zeros(6), deltat=1)
        citi, fiti = test_area_map.get_visual_landmark(xti, camera_fov=camera_fov,
                                                       camera_range=camera_range,
                                                       sigma_cam=percep_sigmas)

        # algorithm execution
        if algo_name == 'FastSLAM_1':

            Y, w = FastSLAM_1(Y=Y, u_t=ut, z_t=fiti,
                              p0=p0, alphas_motion=motion_alphas, deltat=1,
                              camera_range=camera_range, sigmas_percep=percep_sigmas[:2])

            if len(citi) > 0:
                ldmks_string = f'{len(citi)} ldmks detected!'
            else:
                ldmks_string = 'No Ldmks detected...'
            title = f"Uncertainty update i={i}, N={0}, {ldmks_string}"
        else: raise 'algo_name err'

        plot_FastSLAM_scene(xt=xti, Y=Y, w=w, area_map=test_area_map, camera_fov=camera_fov,
                            camera_range=camera_range, title=title, save=True,
                            figname=images_dir + 'iter_{0:03d}'.format(i))

        print(f'Fim da iteração {i} de {len(ut_seq.T)-1}')


    make_movie(images_dir, f'./movies/{algo_name}_movie.gif')
