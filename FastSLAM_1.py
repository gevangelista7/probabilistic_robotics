from copy import deepcopy
import numpy as np
from numpy import sin, cos, sqrt, arctan2, pi
from motion_model import sample_model_velocity
from utils import SLAMParticle, stochastic_universal_sampler, calc_map_jacobian, relative_bearing, calc_pos_jacobian


def FastSLAM_1(Y, u_t, z_t, p0, alphas_motion, camera_range, camera_fov,
               deltat=1, sigmas_percep=(.1, .1, .1)):

    ### verify
    assert z_t.shape[0] == 3

    Q_t = np.diag(sigmas_percep)

    w_particles = np.ones(len(Y))

    ### line 2
    for k, particle in enumerate(Y):  # iter over particles
        assert type(particle) == SLAMParticle
        # line 4
        particle.x = sample_model_velocity(x_tp=particle.x, u_t=u_t, alphas=alphas_motion, deltat=deltat)

        dubious_feat = set()
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
                Qj = Hj @ Sigmaj @ Hj.T + Q_t

                Qj_inv = np.linalg.inv(Qj)
                Qinv_cache.append(Qj_inv)

                # line 9
                w_features[j] = (2 * pi * np.linalg.det(Qj))**-.5 * np.exp(-.5 * (z - zhat).T @ Qj_inv @ (z - zhat))

            # line 11
            w_features[particle.N] = p0

            # line 12
            w_particles[k] *= np.max(w_features)

            # line 13
            c = np.argmax(w_features)

            # line 14, and hold the current N for further comparisons
            Ntp = particle.N
            Nt = max(Ntp, c+1)

            for j in range(Nt):
                if j == c and j == Ntp:                                      # new feature
                    # line 17
                    mux, muy = particle.compute_expected_landmark_pos(z)
                    mu = np.array([[mux], [muy]])

                    # line 18
                    Hj = calc_map_jacobian(particle.pos(), mu.squeeze())
                    Hj_inv = np.linalg.inv(Hj)
                    Sigma = Hj_inv.T @ Q_t @ Hj_inv

                    particle.insert_new_feature(mu, Sigma)

                elif j == c and j < Ntp:                                    # update observed feature
                    mu, Sigma, _ = particle.access_feature(j)
                    zhat = particle.compute_expected_measurement(j)

                    Hj = calc_map_jacobian(particle.pos(), mu.squeeze())

                    # line 21
                    # Qj_inv = Qinv_cache[j]
                    Qj = Hj @ Sigma @ Hj.T + Q_t
                    Qj_inv = np.linalg.inv(Qj)

                    K = Sigma @ Hj.T @ Qj_inv

                    # line 22
                    mu = mu + K @ (z - zhat)

                    # line 23
                    Sigma = (np.eye(2) - K @ Hj) @ Sigma

                    # update (increment into the particle function
                    particle.update_map(j, mu, Sigma)

                else:                                                           # all other features
                    # check if into the perception range
                    # line 28
                    dist = np.linalg.norm(particle.mu_features[j] - particle.x[:2])
                    rel_bearing = relative_bearing(observer_angle=particle.x[2],
                                                   target_angle=arctan2(particle.mu_features[j][0] - particle.x[0],
                                                                        particle.mu_features[j][1] - particle.x[1]))
                    if dist < camera_range and abs(rel_bearing) < camera_fov/2:
                        dubious_feat.add(j)

        dubious_feat = list(dubious_feat)
        dubious_feat.sort()
        dubious_feat.reverse()
        for j in dubious_feat:
            particle.counter_features[j] -= 1

            # lines 32, and 33
            if particle.counter_features[j] < 0:
                particle.delete_feature(j)

    # lines 40-45 -> resampling
    survivors_idx = stochastic_universal_sampler(w_particles, len(Y))
    Yaux = []
    for i in survivors_idx:
        Yaux.append(deepcopy(Y[i]))

    w_particles = w_particles[survivors_idx]

    return Yaux, w_particles
