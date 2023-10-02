from copy import deepcopy
import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2, sign
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge, ConnectionPatch, Ellipse
import shapely.geometry as shp
import seaborn as sns
from utils import Map
import matplotlib.animation as animation
import os
import shutil


# funções de plots
def plot_robot_v3(pose, radius=0.3, color='black', zorder=2):
    center = (pose[0], pose[1])
    ang = pose[2] * 180 / np.pi

    circ = Circle(center, radius, fill=False, color=color, zorder=zorder)
    wed = Wedge(center, radius, ang - 2, ang + 2, color=color, zorder=zorder)

    return [circ, wed]


def plot_MCL_scene(Xtp, Xt, xt_real, i, camera_range, camera_fov, area_map):
    xti_real = xt_real[:, i]
    # plot
    particles_original_pats0 = []
    for p in Xtp.T[:min(1000, Xtp.shape[1]), :]:
        particles_original_pats0 += plot_robot_v3(p, color='tan', zorder=1, radius=0.1)
    particles_original_pats = deepcopy(particles_original_pats0)
    camera_plot = [Wedge(center=xti_real[:2].T, r=camera_range,
                         theta1=((xti_real[2] - camera_fov / 2) * 180 / pi).item(),
                         theta2=((xti_real[2] + camera_fov / 2) * 180 / pi).item(),
                         color='lightgreen', zorder=0)]
    real_robot = plot_robot_v3(xti_real, color='red', zorder=3)
    plot_robots_in_map(xt_real[:, :i], area_map, real_robot + particles_original_pats0 + camera_plot,
                       title=f'move and read camera i={i}')

    # plot 2
    particles_resamp_pats = []
    for p in Xt.T[:min(1000, Xt.shape[1]), :]:
        particles_resamp_pats += plot_robot_v3(p, color='orange', zorder=2, radius=0.1)
    real_robot = plot_robot_v3(xti_real, color='red', zorder=3)
    plot_robots_in_map(xt_real[:, :i], area_map, real_robot + particles_resamp_pats + particles_original_pats,
                       title=f'localization and resampling, i={i}')


def get_mahalanobis_level_pat(mean, S, sigma_levels=None, colors=None, zorder=0):

    if sigma_levels is None:
        sigma_levels = [1]

    eigenvalues, eigenvectors = np.linalg.eig(S)

    principal_axis_index = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, principal_axis_index]

    if principal_axis_index == 0:
      sigma_x, sigma_y = eigenvalues
    else:
      sigma_y, sigma_x = eigenvalues

    if not np.isreal(principal_axis[1]) or not np.isreal(principal_axis[0]):
        return []

    angular_coeff = arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi

    curves = []
    if colors is None:
        line_patterns = ['r', 'g', 'b', 'r', 'g', 'b']
    else:
        line_patterns = colors

    for n_sigma_idx in range(len(sigma_levels)):
      lp = line_patterns[n_sigma_idx % len(line_patterns)]
      n_sigma = sigma_levels[n_sigma_idx]
      pat = Ellipse(xy=mean,
                    width=2*n_sigma*sigma_x, height=2*n_sigma*sigma_y,
                    angle=angular_coeff,
                    lw=1, edgecolor=lp, fc='None',
                    label=r'${} \sigma$'.format(n_sigma), zorder=zorder)

      # major_axis_endpoints = [
      #     [mean[0] - np.cos(np.radians(angular_coeff)) * n_sigma * sigma_x,
      #      mean[1] - np.sin(np.radians(angular_coeff)) * n_sigma * sigma_x],
      #     [mean[0] + np.cos(np.radians(angular_coeff)) * n_sigma * sigma_x,
      #      mean[1] + np.sin(np.radians(angular_coeff)) * n_sigma * sigma_x]
      # ]
      # minor_axis_endpoints = [
      #     [mean[0] + np.sin(np.radians(angular_coeff)) * n_sigma * sigma_y,
      #      mean[1] - np.cos(np.radians(angular_coeff)) * n_sigma * sigma_y],
      #     [mean[0] - np.sin(np.radians(angular_coeff)) * n_sigma * sigma_y,
      #      mean[1] + np.cos(np.radians(angular_coeff)) * n_sigma * sigma_y]
      # ]
      #
      # # Create lines for major and minor axes
      # major_axis_line = plt.Line2D(*zip(*major_axis_endpoints), lw=1, color=lp)
      # minor_axis_line = plt.Line2D(*zip(*minor_axis_endpoints), lw=1, color=lp)

      # curves.extend([pat, major_axis_line, minor_axis_line])

      curves.append(pat)

    return curves


def get_camera_fov_pat(xti, camera_range, camera_fov):
    camera_plot = [Wedge(center=xti[:2].T, r=camera_range,
                         theta1=((xti[2]-camera_fov/2)*180/pi).item(),
                         theta2=((xti[2]+camera_fov/2)*180/pi).item(),
                         color='lightgreen', zorder=0, alpha=0.7)]
    return camera_plot


def get_robot_pat(pose, radius=0.3, color='black', zorder=2):
  center = (pose[0], pose[1])
  ang = pose[2] * 180 / np.pi

  circ = Circle(center, radius, fill=False, color=color, zorder=zorder)
  wed = Wedge(center, radius, ang-2, ang+2, color=color, zorder=zorder)

  return [circ, wed]


def get_scans_pat(pose, intersections):
  center = (pose[0], pose[1])
  lines = []
  for i in range(len(intersections)):
    ray_final = intersections[i]
    # print(ray_final)
    lin = ConnectionPatch(center, ray_final, coordsA="data",
                          arrowstyle="->", color='green')
    lines.append(lin)

  return lines


def plot_robots_in_map(xt, reference_map, extra_patches=None, grid_res=None, title=None, save=False, figname=None):
    assert type(reference_map) == Map

    map_patches = reference_map.get_patches()
    robot_patches = []
    for i in range(len(xt.T)):
        robot_patches += get_robot_pat(pose=xt.T[i], radius=0.3)

    all_patches = map_patches + robot_patches
    if extra_patches is not None:
        all_patches += extra_patches

    fig, ax = plt.subplots()

    for p in all_patches:
        ax.add_artist(p)

    min_x, max_x, min_y, max_y = reference_map.get_map_limits()

    if grid_res is not None:
        n_x_grids = int((max_x - min_x) / grid_res) + 1
        n_y_grids = int((max_y - min_y) / grid_res) + 1
        x_grids = np.linspace(min_x, max_x, n_x_grids)
        y_grids = np.linspace(min_y, min_y, n_y_grids)
        ax.set_xticks(x_grids, minor=True)
        ax.set_yticks(y_grids, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.5)

    if title:
        ax.set_title(title)

    ax.axis('square')
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    plt.grid()
    if save:
        plt.savefig(figname)
    else:
        plt. show()


def plot_scene_with_confidence(xt, mu, Sigma, area_map, camera_on=False,
                               camera_fov=None, camera_range=None, title=None, save=False, figname=None):
    pats = []
    pos = True

    for idx in range(0, len(mu), 3):
        mean_x = mu[idx]
        mean_y = mu[idx+1]
        S = Sigma[idx:idx+2, idx:idx+2]
        if pos:
            pats += get_mahalanobis_level_pat((mean_x, mean_y), S, colors=['r'])
            pos = False
            continue
        # if round(mu[idx+2].item()) in seen_cits:
        pats += get_mahalanobis_level_pat((mean_x, mean_y), S, colors=['b'])

    if camera_on:
        pats += get_camera_fov_pat(xt, camera_range=camera_range, camera_fov=camera_fov)

    # real_robot
    pats += get_robot_pat(xt.T[0], color='red')

    plot_robots_in_map(xt=mu[:3], reference_map=area_map, title=title, extra_patches=pats, save=save, figname=figname)


def get_particle_pats(mu, Sigma):
    pats = []
    pos = True

    for idx in range(0, len(mu), 3):
        mean_x = mu[idx]
        mean_y = mu[idx + 1]
        S = Sigma[idx:idx + 2, idx:idx + 2]
        if pos:
            pats += get_mahalanobis_level_pat((mean_x, mean_y), S, colors=['r'])
            pos = False
            continue

        pats += get_mahalanobis_level_pat((mean_x, mean_y), S, colors=['b'])

    return pats


def plot_FastSLAM_scene(xt, Y, w, area_map, camera_fov, camera_range, title, save=False, figname=None):
    pats = []
    for k, particle in enumerate(Y):
        pats += get_robot_pat(particle.x.squeeze(), color='red', radius=0.2)
        # for i in range(particle.N):
        #     mu, Sigma, _ = particle.access_feature(i)
        #     pats += get_mahalanobis_level_pat(mean=(mu[0], mu[1]), S=Sigma, colors=['blue'], zorder=3)

    worst_particle = Y[np.argmin(w)]
    for i in range(worst_particle.N):
        mu, Sigma, _ = worst_particle.access_feature(i)
        pats += get_mahalanobis_level_pat(mean=(mu[0], mu[1]), S=Sigma[:2, :2], colors=['cyan'], zorder=3)

    best_particle = Y[np.argmax(w)]
    for i in range(best_particle.N):
        mu, Sigma, _ = best_particle.access_feature(i)
        pats += get_mahalanobis_level_pat(mean=(mu[0], mu[1]), S=Sigma[:2, :2], colors=['blue'], zorder=3)

    pats += get_camera_fov_pat(xt, camera_range=camera_range, camera_fov=camera_fov)

    plot_robots_in_map(xt=xt, reference_map=area_map, title=title, extra_patches=pats,
                       save=save, figname=figname)


def make_movie(dir, out_filename, fps=4):
    plot_files = sorted(os.listdir(dir))

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax = plt.gca()
    ax.axis('off')  # Turn off the axis

    # Initialize an empty list to hold plot images
    images = []

    # Loop through the plot files and add them to the images list
    for plot_file in plot_files:
        if plot_file.endswith('.png'):  # Adjust the file extension if necessary
            image = plt.imread(os.path.join(dir, plot_file))
            img = ax.imshow(image, animated=True)
            images.append([img])

    # Create an animation
    ani = animation.ArtistAnimation(fig, images, interval=200, blit=True)  # Adjust interval as needed

    # Display the animation (optional)
    # plt.show()
    #
    # # Save the animation to a file (optional)
    # ani.save(out_filename, writer='pillow', fps=4)
    ani.save(out_filename, writer='pillow', fps=fps)
    print('animation saved!')
