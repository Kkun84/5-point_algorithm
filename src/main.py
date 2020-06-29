from logging import getLogger
import hydra
import numpy as np
import cv2

import utils


logger = getLogger(__name__)


def solve_F(points_list, K):
    assert len(points_list) == 2
    E, _ = cv2.findEssentialMat(*points_list, K)
    logger.info(f"\n{E=}")

    if E.shape != (3, 3):
        return None

    Kinv = np.linalg.inv(K)

    F = Kinv.T @ E @ Kinv
    return F


def main(cfg):
    logger.info('\n' + str(cfg.pretty()))
    images_list = []
    points_list = []
    test_points_list = []
    logger.debug('Read pair images & points.')
    # Load
    for path, points, test_points in [[i.path, i.points, i.test_points] for i in cfg.images]:
        logger.info(f"{path=}")
        # images_list
        image = cv2.cvtColor(cv2.imread(hydra.utils.to_absolute_path(path)), cv2.COLOR_BGR2RGB)
        logger.info(f"{image.shape=}")
        images_list.append(image)
        center = np.array(image.shape[:2][::-1]) / 2
        # points_list
        points = np.array(points)
        points = points - center
        logger.info(f"\n{points=}")
        points_list.append(points)
        # test_points_list
        test_points = np.array(test_points)
        test_points = test_points - center
        logger.info(f"\n{test_points=}")
        test_points_list.append(test_points)
    # points_list
    assert len(images_list) == 2
    points_list = np.array(points_list)
    assert points_list.shape[0] == 2
    assert points_list.shape[2] == 2
    # test_points_list
    assert len(images_list) == 2
    test_points_list = np.array(test_points_list)
    assert test_points_list.shape[0] == 2
    assert test_points_list.shape[2] == 2
    # Make & save pointed images
    for i, (image, points) in enumerate(zip(images_list, points_list)):
        utils.save_poined_marker_image(image, points, i)
    for i, (image, points) in enumerate(zip(images_list, test_points_list)):
        utils.save_poined_marker_image(image, points, i, prefix='test_')
    # Solve F from K
    K = np.array(cfg.K)
    logger.info(f"\n{K=}")

    F = solve_F(points_list, K)
    if F is None:
        logger.info('Could not solve F.')
        return

    logger.info(f"\n{F=}")
    logger.info(f"{np.linalg.eig(F)[0]=}")
    # Line epipolar
    utils.save_lined_epipolar_image(F.T, images_list[0], points_list[0], points_list[1], 0)
    utils.save_lined_epipolar_image(F, images_list[1], points_list[1], points_list[0], 1)
    utils.save_lined_epipolar_image(F.T, images_list[0], test_points_list[0], test_points_list[1], 0, prefix='test_')
    utils.save_lined_epipolar_image(F, images_list[1], test_points_list[1], test_points_list[0], 1, prefix='test_')
    return


if __name__ == "__main__":
    hydra.main(config_path='../conf/config.yaml')(main)()
