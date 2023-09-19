from operator import itemgetter
import numpy as np
from numpy import sin, cos, sqrt, arctan2, pi
from motion_model import sample_model_velocity
from utils import SLAMParticle, stochastic_universal_sampler, calc_map_jacobian


def FastSLAM_1(Y, u_t, z_t, p0, alphas_motion, camera_range,
               deltat=1, sigmas_percep=(.1, .1, .1)):

    ### verify
    assert z_t.shape[0] == 3

    Q_percep = np.diag(sigmas_percep)

    w_particles = np.ones((len(z_t[0]), len(Y)))

    ### line 2
    for z_idx, z in enumerate(z_t.T):                       # iter over readings
        z = z[:2][:, None]                                  # ignore s and make vertical again
        for k, particle in enumerate(Y):                    # iter over particles
            assert type(particle) == SLAMParticle

            # line 4
            x = sample_model_velocity(x_tp=particle.x, u_t=u_t, alphas=alphas_motion, deltat=deltat)
            particle.update_pos(x)

            w_features = np.zeros(particle.N + 1)

            # line 5
            Qinv_cache = []
            for j in range(particle.N):                 # iter over features to get likelihoods
                # line 6
                zhat = particle.measurement_prediction(j)

                # lines 7
                muj, Sigmaj, _ = particle.access_feature(j)
                Hj = calc_map_jacobian(particle.x[:2, 0], muj.squeeze())

                # lines 8
                Qj = Hj @ Sigmaj @ Hj.T + Q_percep

                Qj_inv = np.linalg.inv(Qj)
                Qinv_cache.append(Qj_inv)

                # line 9
                w_features[j] = (2 * pi * np.linalg.det(Qj))**-.5 * np.exp(-.5 * (z - zhat).T @ Qj_inv @ (z - zhat))


            # line 11
            w_features[-1] = p0

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
                    mux, muy = particle.inverse_measure(z)
                    mu = np.array([mux, muy])[:, None]
                    # line 18
                    Hj = calc_map_jacobian(particle.x[:2, 0], [mux, muy])
                    Hj_inv = np.linalg.inv(Hj)
                    Sigma = Hj_inv.T @ Q_percep @ Hj_inv

                    particle.insert_new_feature(mu, Sigma)

                elif (j == c) and j <= (Ntp - 1):                                        # observed feature
                    mu, Sigma, _ = particle.access_feature(j)
                    Hj = calc_map_jacobian(particle.x[:2, 0], mu.squeeze())
                    zhat = particle.measurement_prediction(j)

                    # line 21
                    Qj_inv = Qinv_cache[j]
                    K = Sigma @ Hj @ Qj_inv

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

                    if dist < camera_range:
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
    Y = itemgetter(*survivors_idx)(Y)

    return Y
