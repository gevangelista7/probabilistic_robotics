import numpy as np
from numpy import sin, cos, sqrt, arctan2, pi
from motion_model import sample_model_velocity
from utils import SLAMParticle, stochastic_universal_sampler


def FastSLAM_1(Y, u_t, z_t, p0, alphas_motion, camera_range,
               deltat=1, sigmas_percep=(.1, .1, .1)):

    ### verify
    assert z_t.shape[0] == 3

    Q_percep = np.diag(sigmas_percep)

    w_particles = np.ones((len(Y), len(z_t[0])))

    ### line 2
    for z_idx, z in enumerate(z_t.T):                       # iter over readings
        for k, particle in enumerate(Y):                    # iter over particles
            assert type(particle) == SLAMParticle

            # line 4
            x = sample_model_velocity(x_tp=particle.x, u_t=u_t, alphas=alphas_motion, deltat=deltat)
            particle.update_pos(x)

            w_features = np.zeros(particle.N + 1)

            # line 5
            for j in range(particle.N):                 # iter over features to get likelihoods
                # line 6
                zhat = particle.measurement_prediction(j)

                # lines 7 and 8
                Qj = particle.get_measurement_covariance(j=j, Qt=Q_percep)

                # line 9
                w_features[j] = (2 * pi * np.linalg.det(Qj))**-.5 * np.exp(-.5 * (z - zhat).T @ np.linalg.inv(Qj) @ (z - zhat))

            # line 11
            w_features[-1] = p0

            # line 12
            w_particles[k] *= np.max(w_features[z_idx])

            # line 13
            c = np.argmax(w_features[z_idx])

            # line 14, and hold the current N for further comparisons
            N0 = particle.N
            N = max(N0, c)

            for j in range(N):
                if j == c and j == N0 + 1:                                      # new feature
                    x_hat, y_hat, s = particle.measure_inverse(z)

                    # line 17
                    mu = np.array([x_hat, y_hat, s])[:, None]

                    # line 18
                    Hj_inv = np.linalg.inv(particle.get_jacobian(j))
                    Sigma = Hj_inv.T @ Q_percep @ Hj_inv

                    particle.insert_new_feature(mu, Sigma)

                elif j == c and j <= N0:                                        # observed feature
                    # prepare, cache used into the particle functions
                    mu, Sigma, _ = particle.access_feature(j)
                    Hj = particle.get_jacobian(j)
                    Qj = particle.get_measurement_covariance(j=j, Qt=Q_percep)
                    zhat = particle.measurement_prediction(j)

                    # line 21
                    K = Sigma @ Hj @ np.linalg.inv(Qj)

                    # line 22
                    mu = mu + K @ (z - zhat)

                    # line 23
                    Sigma = (np.eye(3 + 3*N) - K @ Hj) @ Sigma

                    # update (increment into the particle function
                    particle.update_map(j, mu, Sigma)

                else:                                                           # all other features
                    # does not necessary to update the feature
                    # check if into the perception range
                    # line 28
                    mu, _, _ = particle.access_feature(j)
                    x_f, y_f = mu[0], mu[1]
                    x_pos, y_pos = particle.x[0], particle.x[1]
                    dist = sqrt((x_f - x_pos)**2 + (y_f - y_pos)**2)

                    if dist < camera_range:
                        # line 31
                        particle.counter_features[j] -= 1

                        # lines 32, and 33
                        if particle.counter_features[j] < 0:
                            particle.delete_feature(j)

    # particles likelihoods considering all readings
    w_particles = w_particles.prod(axis=0)

    # lines 40-45 -> resampling
    survivors_idx = stochastic_universal_sampler(w_particles, len(Y))
    Y = Y[survivors_idx]

    return Y
