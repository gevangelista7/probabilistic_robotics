import numpy as np
from numpy import sin, cos, sqrt, arctan2, pi
from motion_model import sample_model_velocity
from utils import SLAMParticle, stochastic_universal_sampler



def FastSLAM_1(Y, u_t, z_t, alphas_motion, Q_percep, p0, camera_range,
               R=None, deltat=1, sigmas_percep=(.1, .1, .1)):

    ### verify
    assert z_t.shape[0] == 3

    if R is None:
        R = np.eye(3)/10

    w_particles = np.ones(len(Y))
    Y_aux = []


    ### line 2
    for k, particle in enumerate(Y):
        assert type(particle) == SLAMParticle

        # line 4
        x = sample_model_velocity(x_tp=particle.x, u_t=u_t, alphas=alphas_motion, deltat=deltat)
        particle.update_pos(x)

        # todo organize the w management
        w = np.zeros((len(z_t), particle.N + 1))

        # line 5
        # TODO verify where to put the readings iter
        for z_idx, z in enumerate(z_t.T):
            for j in range(particle.N):
                # line 6
                zhat = particle.measurement_prediction(j)

                # lines 7 and 8
                Qj = particle.get_measurement_covariance(j=j, Qt=Q_percep)

                # line 9
                w[z_idx, j] = (2 * pi * np.linalg.det(Qj))**-.5 * np.exp(-.5 * (z - zhat).T @ np.linalg.inv(Qj) @ (z - zhat))

            # line 11
            w[z_idx, -1] = p0

            # lines 12, 13, and 14
            c = np.argmax(w[z_idx])
            w = np.max(w[z_idx])

            N0 = particle.N
            N = max(N0, c+1)

            for j in range(N0 + 1):
                if j == c and j == N0 + 1:
                    x_hat, y_hat, s = particle.measure_inverse(z)
                    mu = np.array([x_hat, y_hat, s])[:, None]

                    Hj_inv = np.linalg.inv(particle.get_jacobian(j))
                    Sigma = Hj_inv.T @ Q_percep @ Hj_inv

                    particle.insert_new_feature(mu, Sigma)

                elif j == c and j <= N0:

                    mu, Sigma, _ = particle.access_feature(j)
                    Hj = particle.get_jacobian(j)
                    Qj = particle.get_measurement_covariance(j=j, Qt=Q_percep)
                    zhat = particle.measurement_prediction(j)

                    K = Sigma @ Hj @ np.linalg.inv(Qj)
                    mu = mu + K @ (z - zhat)
                    Sigma = (np.eye(3 + 3*N) - K @ Hj) @ Sigma

                    particle.update_map(j, mu, Sigma)

                else:
                    mu, _, _ = particle.access_feature(j)
                    x_f, y_f = mu[0], mu[1]
                    x_pos, y_pos = particle.x[0], particle.x[1]

                    dist = sqrt((x_f - x_pos)**2 + (y_f - y_pos)**2)
                    if dist < camera_range:
                        particle.counter_features[j] -= 1
                        if particle.counter_features[j] < 0:
                            particle.delete_feature(j)

            # TODO w_particles management!

        survivors_idx = stochastic_universal_sampler(w_particles, len(Y))

        Y = Y[survivors_idx]

    return Y

