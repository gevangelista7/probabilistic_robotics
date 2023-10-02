from copy import deepcopy
import numpy as np
from numpy import arctan2, pi
from motion_model import sample_model_velocity
from utils import SLAMParticle, stochastic_universal_sampler, calc_map_jacobian, relative_bearing, calc_pos_jacobian


def FastSLAM_2(Y, u_t, z_t, p0, alphas_motion, camera_range, camera_fov,
               deltat=1, R=None, sigmas_percep=(.1, .1, .1)):
    ### verify
    assert z_t.shape[0] == 3

    Q_t = np.diag(sigmas_percep)

    if R is None:
        R = np.eye(3)/10

    R_inv = np.linalg.inv(R)

    w_particles = np.ones(len(Y))

    ### line 2
    for k, particle in enumerate(Y):  # iter over particles
        assert type(particle) == SLAMParticle

        ### line 4
        for z_idx, z in enumerate(z_t.T):                       # iter over readings
            s = z[2]
            z = z[:2][:, None]                                  # ignore s and make vertical again

            w_features = np.zeros(particle.N + 1)

            # Caches
            xt_feat = np.zeros(particle.N)
            Hmj_cache = []
            Hxj_cache = []
            Qj_cache = []
            Qjinv_cache = []

            for j in range(particle.N):
                muj, Sigmaj, _ = particle.access_feature(j)

                ### line 5
                xhatj = sample_model_velocity(x_tp=particle.x, u_t=u_t, alphas=np.zeros(6), deltat=deltat)

                ### line 6
                zbarj = particle.compute_expected_measurement(j)

                ### lines 7 and 8
                Hxj = calc_pos_jacobian(x=xhatj, mu=muj)
                Hmj = calc_map_jacobian(x=xhatj, mu=muj)

                ### line 9
                Qj = Q_t + Hmj @ Sigmaj @ Hmj.T
                Qj_inv = np.linalg.inv(Qj)

                ### line 10
                Sigmaxj = np.linalg.inv(Hxj.T @ Qj_inv @ Hxj + R_inv)

                ### line 11
                muxj = Sigmaxj @ Hxj.T @ Qj_inv @ (z - zbarj) + xhatj

                ### line 12
                xt_feat[j] = np.random.multivariate_normal(mean=muxj, cov=Sigmaxj)[:, None]

                ### line 13
                zhat = particle.compute_expected_measurement(j)

                ### line 14
                w_features[j] = (2 * pi * np.linalg.det(Qj))**-.5 * np.exp(-.5 * (z - zhat).T @ Qj_inv @ (z - zhat))

                ### store caches
                Hmj_cache.append(Hmj)
                Hxj_cache.append(Hxj)
                Qj_cache.append(Qj)
                Qjinv_cache.append(Qj_inv)

            ### line 16
            w_features[particle.N] = p0

            ### line 17
            c = np.argmax(w_features)

            ### line 18, and hold the current N for further comparisons
            Ntp = particle.N
            Nt = max(Ntp, c+1)

            ### line 19
            dubious_feat = set()
            for j in range(Nt):
                ### line 20
                if j == c and j == Ntp:                                      # new feature
                    ### lines 21 - 26
                    particle.x = sample_model_velocity(x_tp=particle.x, u_t=u_t, alphas=alphas_motion, deltat=deltat)

                    mux, muy = particle.compute_expected_landmark_pos(z)
                    mu = np.array([[mux], [muy]])

                    Hmj = calc_map_jacobian(particle.pos(), mu.squeeze())
                    Hmj_inv = np.linalg.inv(Hmj)
                    Sigma = Hmj_inv.T @ Q_t @ Hmj_inv

                    particle.insert_new_feature(mu, Sigma)

                    w_particles[k] = p0

                elif j == c and j < Ntp:
                    mu_tp, Sigma_tp, _ = particle.access_feature(j)
                    zhat = particle.compute_expected_measurement(j)

                    Hmj = Hmj_cache[j]
                    Hxj = Hxj_cache[j]
                    Qj_inv = Qjinv_cache[j]

                    ### lines 28 - 34
                    particle.x = xt_feat[j]

                    K = Sigma_tp @ Hmj.T @ Qj_inv

                    mu = mu_tp + K @ (z - zhat)

                    Sigma = (np.eye(2) - K @ Hmj) @ Sigma_tp

                    particle.update_map(j, mu, Sigma)

                    L = Hxj @ R @ Hxj.T + Hmj @ Sigma_tp  @ Hmj.T + Q_t
                    L_inv = np.linalg.inv(L)

                    w_particles[k] *= (2 * pi * np.linalg.det(L))**-.5 * np.exp(-.5 * (z - zhat).T @ L_inv @ (z - zhat))

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

    survivors_idx = stochastic_universal_sampler(w_particles, len(Y))
    Yaux = []
    for i in survivors_idx:
        Yaux.append(deepcopy(Y[i]))

    w_particles = w_particles[survivors_idx]

    return Yaux, w_particles
