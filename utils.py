import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import shapely.geometry as shp
import seaborn as sns
import os
import shutil


class Map:
    def __init__(self, obstacles, landmarks, area_limit):
        """
        obstacles: deve ser uma lista de obstaculos.
                    o obstáculo é um polígono e deve ser convexos, definidos por pontos sequenciais.
        landmarks: é construído como um dicionário, com a key inteira e sequencial definindo como um identificador
                    o value da entrada do dicionário deve ser uma tupla (x_i, y_i)
        area_limit: deve ser um poligono convexo, conforme os obstacles
        """

        self.obstacles = obstacles      # lista de poligonos
        self.landmarks = landmarks      # dicionário de id_ldmk: (mx, my)
        self.area_limit = area_limit    # poligono

    def get_patches(self):
        patches = []
        background = Polygon(np.array(self.area_limit), color='white', zorder=-1)
        patches.append(background)

        for obs in self.obstacles:
            pat = Polygon(np.array(obs), closed=True, color='black')
            patches.append(pat)

        for ldmk_id in self.landmarks.keys():
            center = np.array(self.landmarks[ldmk_id])
            pat = Circle(xy=center, color='red', radius=0.3,
                         label=ldmk_id)
            patches.append(pat)

        return patches

    def point_is_occupied(self, point):
        occupied = False
        for obst in self.obstacles:
            occupied = occupied or shp.polygon.Polygon(obst).contains(shp.Point(point))
            if occupied:
                break

        return occupied

    def get_visual_landmark(self, X, camera_range, camera_fov, sigma_cam):
        x = X[0]
        y = X[1]
        theta = X[2]
        cit = []
        fit_r = []
        fit_phi = []
        fit_s = []
        for lmk_id, lmk_xy in self.landmarks.items():
            mx, my = lmk_xy
            dist = sqrt((mx - x) ** 2 + (my - y) ** 2).item()
            deltatheta = smaller_arc_between_angles(observer_angle=theta,
                                                    target_angle=arctan2(my - y, mx - x)).item()

            if dist < camera_range and abs(deltatheta) < camera_fov / 2:
                cit.append(lmk_id)
                fit_r.append(np.random.normal(dist, sigma_cam[0]))
                fit_phi.append(np.random.normal(deltatheta, sigma_cam[1]))
                fit_s.append(np.random.normal(lmk_id, sigma_cam[2]))

        fit = np.vstack((fit_r, fit_phi, fit_s))
        return cit, fit

    def get_map_limits(self):
        map_x_limits = np.array(self.area_limit)[:, 0]
        map_y_limits = np.array(self.area_limit)[:, 1]
        min_x, max_x = map_x_limits.min(), map_x_limits.max()
        min_y, max_y = map_y_limits.min(), map_y_limits.max()
        return min_x, max_x, min_y, max_y


class HGrid:
    def __init__(self, min_scales, max_scales, res):
        self.min_scales = min_scales
        self.max_scales = max_scales
        self.res = res

        self.x_min, self.y_min, self.theta_min = min_scales
        self.x_max, self.y_max, self.theta_max = max_scales
        self.x_res, self.y_res, self.theta_res = res

        self.n_x = int((self.x_max - self.x_min) / self.x_res)
        self.n_y = int((self.y_max - self.y_min) / self.y_res)
        self.n_theta = int((self.theta_max - self.theta_min) / self.theta_res)

        self.value = {(xi, yi, thetai): 0 for xi in range(self.n_x)
                      for yi in range(self.n_y)
                      for thetai in range(self.n_theta)}

    def cell_center(self, x_idx, y_idx, theta_idx):
        assert x_idx <= self.n_x - 1
        assert y_idx <= self.n_y - 1
        assert theta_idx <= self.n_theta - 1

        x = self.x_min + self.x_res * (x_idx + 0.5)
        y = self.y_min + self.y_res * (y_idx + 0.5)
        theta = self.theta_min + self.theta_res * (theta_idx + 0.5)
        return np.array([[x], [y], [theta]])

    def state_to_idx(self, x):
        x_idx = int((x[0] - self.x_min) / self.x_res)
        y_idx = int((x[1] - self.y_min) / self.y_res)

        x_idx = np.clip(x_idx, self.x_min, self.x_max)
        y_idx = np.clip(y_idx, self.y_min, self.y_max)
        theta_idx = int((normalize_angle(x[2]) - self.theta_min) / self.theta_res)
        return (x_idx, y_idx, theta_idx)

    def get_zeros_like(self):
        return self.__init__(self.min_scales, self.max_scales, self.res)

    def get_grid_map(self):
        gridmap = np.zeros((self.n_x, self.n_y))
        for xk in range(self.n_x):
            for yk in range(self.n_y):
                for thetak in range(self.n_theta):
                    gridmap[xk, yk] += self.value[xk, yk, thetak]

        return gridmap / gridmap.sum()


class PGrid(HGrid):
    def __init__(self, min_scales, max_scales, res):
        super().__init__(min_scales, max_scales, res)
        self.n_cells = self.n_x * self.n_y * self.n_theta
        self.value = {(xi, yi, thetai): 1 / self.n_cells
                      for xi in range(self.n_x)
                      for yi in range(self.n_y)
                      for thetai in range(self.n_theta)}

    def plot_p_grid(self, title=None):
        ax = sns.heatmap(self.get_grid_map().T, cmap='Reds', linewidths=0.1)
        ax.invert_yaxis()
        ax.set_title(title)
        plt.show()


class SLAMParticle:
    def __init__(self, x0):
        self.x = x0
        self.N = 0
        self.mu_features = []
        self.Sigmas_features = []
        self.counter_features = []

        # caches
        self.Q_cache = []
        self.zhat_cache = []

    def insert_new_feature(self, mu, Sigma):
        self.N += 1
        self.mu_features.append(mu)
        self.Sigmas_features.append(Sigma)
        self.counter_features.append(1)

    def update_pos(self, x):
        self.x = x

    def update_map(self, j, mu, Sigma):
        self.mu_features[j] = mu
        self.Sigmas_features[j] = Sigma

    def access_feature(self, j):
        mu = self.mu_features[j]
        Sigma = self.Sigmas_features[j]
        counter = self.counter_features[j]

        return mu, Sigma, counter

    def measurement_prediction(self, j):
        muk, _, _ = self.access_feature(j)
        deltax, deltay, q = calc_deltas(self.x[:2], muk)
        zhat = np.array([[q ** .5],
                         [smaller_arc_between_angles(self.x[2], arctan2(deltay, deltax)).item()]])

        return zhat

    def inverse_measure(self, z):
        dist = z[0]
        delta_theta = z[1]

        x_hat = self.x[0] + dist * cos(self.x[2] + delta_theta)
        y_hat = self.x[1] + dist * sin(self.x[2] + delta_theta)

        return x_hat.item(), y_hat.item()

    def delete_feature(self, j):
        self.mu_features.pop(j)
        self.Sigmas_features.pop(j)
        self.counter_features.pop(j)
        self.N -= 1
        pass


# fundamentals and problem geometry
def init_SLAM_population(X):
    population = []
    for x in X.T:
        population.append(SLAMParticle(x[:, None]))

    return population


def normalize_angle(angle):
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def smaller_arc_between_angles(observer_angle, target_angle):
    observer_angle = normalize_angle(observer_angle)
    target_angle = normalize_angle(target_angle)

    angle_difference = target_angle - observer_angle

    angle_difference = normalize_angle(angle_difference)

    return angle_difference


def stochastic_universal_sampler(w_array, n_samples):
    if w_array.sum() == 0:
        prob = np.ones_like(w_array) / len(w_array)
    else:
        prob = w_array / w_array.sum()

    cum_prob = np.cumsum(prob)
    r = np.random.uniform(0, 1 / n_samples)
    id_sel_list = []
    i = 0
    if n_samples > 0:
        while len(id_sel_list) < n_samples:
            if i < len(cum_prob):
                while r < cum_prob[i]:
                    id_sel_list.append(i)
                    r += 1 / n_samples
            else:
                id_sel_list.append(len(w_array) - 1)
            i += 1
    return id_sel_list


def prob(dist, sigma):
    if sigma < 1e-5:
        sigma += 1e-5
    p = np.exp(-.5 * dist ** 2 / sigma ** 2) / sqrt(2 * pi * sigma ** 2)
    return p


def get_random_samples(n_samples, area_map):
    X0 = []
    min_x, max_x, min_y, max_y = area_map.get_map_limits()
    while len(X0) < n_samples:
      x = np.random.uniform(low=min_x, high=max_x)
      y = np.random.uniform(low=min_y, high=max_y)
      theta = np.random.uniform(-pi/2, pi/2)
      if not area_map.point_is_occupied((x, y)):
          X0.append([x, y, theta])

    Xt = np.array(X0).T
    w0 = np.ones_like(Xt[0])
    Xt = np.vstack((Xt, w0))
    return Xt


def clear_directory(directory_path):
    try:
        # Delete all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Delete all subdirectories and their contents
        for subdir in os.listdir(directory_path):
            subdir_path = os.path.join(directory_path, subdir)
            if os.path.isdir(subdir_path):
                shutil.rmtree(subdir_path)

        print(f"Directory '{directory_path}' has been cleared.")

    except Exception as e:
        print(f"Error clearing directory '{directory_path}': {str(e)}")


def expand_sigma(sigma, sigma_ext=np.diag(3*[1e8])):
    dim = sigma.shape[0]

    sigma = np.block([
        [sigma,              np.zeros((dim, 3))],
        [np.zeros((3, dim)), sigma_ext]
    ])

    return sigma


def get_pop_poses(Y):
    X = np.zeros([[], [], []])
    for particle in Y:
        assert type(particle) == SLAMParticle
        X = np.hstack(X, Y.x)

    return X


def calc_deltas(x, mu):
    deltax = mu[0] - x[0]
    deltay = mu[1] - x[1]

    delta = mu - x

    q = (delta.T @ delta).item()

    return deltax, deltay, q


def calc_map_jacobian(x, mu):
    deltax, deltay, q = calc_deltas(x, mu)

    sq = sqrt(q)
    Haux_map = np.array([[sq * deltax, sq * deltay],
                         [-deltay, deltax]]) / q

    return Haux_map


def calc_pos_jacobian(x, mu):
    deltax, deltay, q = calc_deltas(x, mu)

    sq = sqrt(q)
    Haux_pos = np.array([[-sq * deltax, -sq * deltay],
                         [deltay, -deltax]]) / q

    return Haux_pos


def calc_aux_jacobian(x, mu):
    Haux_pos = calc_pos_jacobian(x, mu)
    Haux_map = calc_map_jacobian(x, mu)

    return Haux_pos, Haux_map
