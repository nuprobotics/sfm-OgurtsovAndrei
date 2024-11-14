import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: Intrinsic camera matrix, np.ndarray 3x3
    :param camera_position1: First camera position in world coordinates, np.ndarray 3x1
    :param camera_rotation1: First camera rotation matrix in world coordinates, np.ndarray 3x3
    :param camera_position2: Second camera position in world coordinates, np.ndarray 3x1
    :param camera_rotation2: Second camera rotation matrix in world coordinates, np.ndarray 3x3
    :param image_points1: Points in the first image, np.ndarray Nx2
    :param image_points2: Points in the second image, np.ndarray Nx2
    :return: Triangulated points, np.ndarray Nx3
    """
    rotation1_inv = camera_rotation1.T
    rotation2_inv = camera_rotation2.T

    translation1_inv = -rotation1_inv @ camera_position1
    translation2_inv = -rotation2_inv @ camera_position2

    proj_matrix1 = camera_matrix @ np.hstack((rotation1_inv, translation1_inv.reshape(-1, 1)))
    proj_matrix2 = camera_matrix @ np.hstack((rotation2_inv, translation2_inv.reshape(-1, 1)))

    N = image_points1.shape[0]
    ones = np.ones((N, 1))
    x1 = np.hstack((image_points1, ones))
    x2 = np.hstack((image_points2, ones))

    points_4d_hom = np.zeros((N, 4))

    for i in range(N):
        A = np.zeros((4, 4))

        A[0] = x1[i, 0] * proj_matrix1[2] - proj_matrix1[0]
        A[1] = x1[i, 1] * proj_matrix1[2] - proj_matrix1[1]
        A[2] = x2[i, 0] * proj_matrix2[2] - proj_matrix2[0]
        A[3] = x2[i, 1] * proj_matrix2[2] - proj_matrix2[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        points_4d_hom[i] = X

    points_3d = points_4d_hom[:, :3] / points_4d_hom[:, 3][:, np.newaxis]

    return points_3d
