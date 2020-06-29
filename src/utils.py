from logging import getLogger
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors


logger = getLogger(__name__)


def get_colors(n):
    if n > 20:
        return matplotlib.colors.CSS4_COLORS.values()
    elif n > 10:
        name = 'tab20'
    else:
        name = 'tab10'
    return plt.cm.get_cmap(name).colors


def solve_F(pair_points):
    logger.debug('Solve F.')
    x = pair_points[..., 0].T
    y = pair_points[..., 1].T
    M = np.array([
        [
            u[0] * u[1], v[0] * u[1], u[1],
            u[0] * v[1], v[0] * v[1], v[1],
            u[0],        v[0],        1,
        ] for u, v in zip(x, y)
    ])
    logger.debug(f"\n{M=}, {M.shape}")
    # 固有値分解による解法（精度が悪い）
    # w, v = np.linalg.eig(M.T @ M)
    # F = v[w.argmin()].reshape([3, 3])
    # 特異値分解
    u, s, v = np.linalg.svd(M)
    F = v[s.argmin()].reshape((3, 3))
    return F


def save_poined_marker_image(image, points, i, prefix=''):
    logger.debug('Make & save pointed images.')
    colors = get_colors(len(points))
    plt.imshow(image)
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    center = np.array(image.shape[:2][::-1]) / 2
    for p, color in zip(points, colors):
        plt.scatter(*(p + center), c=[color])
    plt.savefig(f"{prefix}pointed_images_{i:02}.png", dpi=300)
    plt.close()
    return


def save_lined_epipolar_image(F, image, points, other_points, i, prefix=''):
    logger.debug('Line epipolar.')
    colors = get_colors(len(other_points))
    ABC = (F @ np.vstack([other_points.T, np.ones(len(other_points))])).T
    plt.imshow(image)
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    center = np.array(image.shape[:2][::-1]) / 2
    if points is not None:
        for p, color in zip(points, colors):
            plt.scatter(*(p + center), c=[color])
    x_points = np.array([-center[0], center[0]])
    # x_points = np.array([0, image.shape[1]])
    for abc, color in zip(ABC, colors):
        y_points = -(abc[0] * x_points + abc[2]) / abc[1]
        plt.plot(x_points + center[0], y_points + center[1], color=color)
    plt.savefig(f"{prefix}lined_epipolar_{i:02}.png", dpi=300)
    plt.close()
    return


# def save_random_epipole_image(F, images_list, i, other_i, prefix=''):
#     logger.debug('Grid line epipolar')
#     other_image = images_list[other_i]
#     points = np.random.uniform(-0.5, 0.5, [20, 2])
#     points[:, 0] *= other_image.shape[1]
#     points[:, 1] *= other_image.shape[0]
#     save_poined_marker_image(other_image, points, i, prefix + 'random_')
#     image = images_list[i]
#     save_lined_epipolar_image(F, image, None, points, i, prefix + 'random_')
#     return


# def save_grid_epipole_image(F, images_list, i, other_i, prefix=''):
#     logger.debug('Grid line epipolar')
#     other_image = images_list[other_i]
#     points = np.meshgrid(*[np.linspace(-0.5, 0.5, 2)] * 2)
#     points[0] *= other_image.shape[1]
#     points[1] *= other_image.shape[0]
#     points = np.array(points).reshape([2, -1]).T
#     save_poined_marker_image(other_image, points, i, prefix + 'grid_')
#     image = images_list[i]
#     save_lined_epipolar_image(F, image, None, points, i, prefix + 'grid_')
#     return


# def save_epipole_pair_image(F, images_list, i, other_i, prefix=''):
#     logger.debug('Grid line epipolar')
#     other_image = images_list[other_i]
#     points = np.meshgrid(*[np.linspace(-0.5, 0.5, 2)] * 2)
#     points[0] *= other_image.shape[1]
#     points[1] *= other_image.shape[0]
#     points = np.array(points).reshape([2, -1]).T
#     save_poined_marker_image(other_image, points, i, prefix + 'grid_')
#     image = images_list[i]
#     save_lined_epipolar_image(F, image, None, points, i, prefix + 'grid_')
#     return
