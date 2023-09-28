from copy import deepcopy
from operator import itemgetter
import numpy as np
from numpy import sin, cos, sqrt, arctan2, pi
from motion_model import sample_model_velocity
from utils import SLAMParticle, stochastic_universal_sampler, calc_map_jacobian, relative_bearing, calc_pos_jacobian


def FastSLAM_1(Y, u_t, z_t, p0, alphas_motion, camera_range, camera_fov,
               deltat=1, sigmas_percep=(.1, .1, .1)):

    ### verify
    assert z_t.shape[0] == 3

    Q_percep = np.diag(sigmas_percep)

    w_particles = np.ones((len(z_t[0]), len(Y)))

    ### line 2
    for k, particle in enumerate(Y):  # iter over particles
        assert type(particle) == SLAMParticle
        # line 4
        particle.x = sample_model_velocity(x_tp=particle.x, u_t=u_t, alphas=alphas_motion, deltat=deltat)

        for z_idx, z in enumerate(z_t.T):                       # iter over readings
            s = z[2]
            z = z[:2][:, None]                                  # ignore s and make vertical again

            Qinv_cache = []

            w_features = np.zeros(particle.N + 1)

            # line 5
            for j in range(particle.N):                 # iter over features to get likelihoods
                muj, Sigmaj, _ = particle.access_feature(j)

                # line 6
                zhat = particle.compute_expected_measurement(j)

                # lines 7
                Hj = calc_map_jacobian(particle.pos(), muj.squeeze())

                # lines 8
                Qj = Hj @ Sigmaj @ Hj.T + Q_percep

                Qj_inv = np.linalg.inv(Qj)
                Qinv_cache.append(Qj_inv)

                # line 9
                w_features[j] = (2 * pi * np.linalg.det(Qj))**-.5 * np.exp(-.5 * (z - zhat).T @ Qj_inv @ (z - zhat))

                # print('zhat', zhat)
                # print('z', z)
                # print('pos', particle.pos())
                # print('muj', muj.squeeze())
                # print('s', s)
                # print('w', w_features[j])
                #
                # print(f'fim da associação, p = {k}, j = {j}')


            # line 11
            w_features[particle.N] = p0

            # line 12
            w_particles[z_idx, k] *= np.max(w_features)

            # line 13
            c = np.argmax(w_features)

            # line 14, and hold the current N for further comparisons
            Ntp = particle.N
            Nt = max(Ntp, c+1)

            delete_list = []
            for j in range(Nt):
                if j == c and j == Ntp:                                      # new feature
                    # line 17
                    mux, muy = particle.compute_expected_landmark_pos(z)
                    mu = np.array([[mux], [muy]])

                    # line 18
                    Hj = calc_map_jacobian(particle.pos(), mu.squeeze())
                    Hj_inv = np.linalg.inv(Hj)
                    Sigma = Hj_inv.T @ Q_percep @ Hj_inv

                    particle.insert_new_feature(mu, Sigma)

                elif (j == c) and j <= (Ntp - 1):                           # observed feature
                    mu, Sigma, _ = particle.access_feature(j)
                    zhat = particle.compute_expected_measurement(j)

                    Hj = calc_map_jacobian(particle.pos(), mu.squeeze())

                    # line 21
                    # Qj_inv = Qinv_cache[j]
                    Qj = Hj @ Sigma @ Hj.T + Q_percep
                    Qj_inv = np.linalg.inv(Qj)

                    K = Sigma @ Hj.T @ Qj_inv

                    # line 22
                    mu = mu + K @ (z - zhat)

                    # line 23
                    Sigma = (np.eye(2) - K @ Hj) @ Sigma

                    # update (increment into the particle function
                    particle.update_map(j, mu, Sigma)

                else:                                                           # all other features
                    # does not necessary to update the feature
                    # check if into the perception range
                    # line 28
                    dist = np.linalg.norm(particle.mu_features[j] - particle.x[:2])
                    rel_bearing = relative_bearing(observer_angle=particle.x[2],
                                                   target_angle=arctan2(particle.mu_features[j][0] - particle.x[0],
                                                                        particle.mu_features[j][1] - particle.x[1]))

                    if dist < camera_range and abs(rel_bearing) < camera_fov:
                        # line 31
                        particle.counter_features[j] -= 1

                        # lines 32, and 33
                        if particle.counter_features[j] < 0:
                            delete_list.append(j)

            delete_list.reverse()
            for j in delete_list:
                particle.delete_feature(j)

    # particles likelihoods considering all readings
    w_particles = w_particles.prod(axis=0)

    # lines 40-45 -> resampling
    survivors_idx = stochastic_universal_sampler(w_particles, len(Y))
    Yaux = []
    for i in survivors_idx:
        Yaux.append(deepcopy(Y[i]))

    w_particles = w_particles[survivors_idx]

    return Yaux, w_particles
