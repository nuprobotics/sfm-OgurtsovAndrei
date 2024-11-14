import numpy as np
import cv2
import typing
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import yaml


# Task 2
def get_matches(image1, image2) -> typing.Tuple[
    typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.DMatch]]:
    sift = cv2.SIFT_create()
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    kp1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    kp2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()
    matches_1_to_2 = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches_2_to_1 = bf.knnMatch(descriptors2, descriptors1, k=2)

    ratio_threshold = 0.75
    good_matches_1_to_2 = []
    for m, n in matches_1_to_2:
        if m.distance < ratio_threshold * n.distance:
            good_matches_1_to_2.append(m)

    good_matches_2_to_1 = []
    for m, n in matches_2_to_1:
        if m.distance < ratio_threshold * n.distance:
            good_matches_2_to_1.append(m)

    matches_2_to_1_dict = {m.queryIdx: m.trainIdx for m in good_matches_2_to_1}

    cross_checked_matches = []
    for m in good_matches_1_to_2:
        queryIdx1 = m.queryIdx
        trainIdx1 = m.trainIdx
        if matches_2_to_1_dict.get(trainIdx1) == queryIdx1:
            cross_checked_matches.append(m)

    return kp1, kp2, cross_checked_matches


def get_second_camera_position(kp1, kp2, matches, camera_matrix):
    coordinates1 = np.array([kp1[match.queryIdx].pt for match in matches])
    coordinates2 = np.array([kp2[match.trainIdx].pt for match in matches])
    E, mask = cv2.findEssentialMat(coordinates1, coordinates2, camera_matrix)
    _, R, t, mask = cv2.recoverPose(E, coordinates1, coordinates2, camera_matrix)
    return R, t, E


# Task 3
def triangulation(
        camera_matrix: np.ndarray,
        camera1_translation_vector: np.ndarray,
        camera1_rotation_matrix: np.ndarray,
        camera2_translation_vector: np.ndarray,
        camera2_rotation_matrix: np.ndarray,
        kp1: typing.Sequence[cv2.KeyPoint],
        kp2: typing.Sequence[cv2.KeyPoint],
        matches: typing.Sequence[cv2.DMatch]
):
    """
    Triangulate 3D points from matched keypoints in two images.

    :param camera_matrix: Intrinsic camera matrix, np.ndarray of shape (3, 3)
    :param camera1_translation_vector: Translation vector of the first camera, np.ndarray of shape (3,)
    :param camera1_rotation_matrix: Rotation matrix of the first camera, np.ndarray of shape (3, 3)
    :param camera2_translation_vector: Translation vector of the second camera, np.ndarray of shape (3,)
    :param camera2_rotation_matrix: Rotation matrix of the second camera, np.ndarray of shape (3, 3)
    :param kp1: Keypoints from the first image, sequence of cv2.KeyPoint
    :param kp2: Keypoints from the second image, sequence of cv2.KeyPoint
    :param matches: Matches between keypoints, sequence of cv2.DMatch
    :return: Triangulated 3D points, np.ndarray of shape (N, 3)
    """

    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)  # Shape: (N, 2)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)  # Shape: (N, 2)

    t1 = camera1_translation_vector.reshape(3, 1)
    t2 = camera2_translation_vector.reshape(3, 1)

    proj_matrix1 = camera_matrix @ np.hstack((camera1_rotation_matrix, t1))
    proj_matrix2 = camera_matrix @ np.hstack((camera2_rotation_matrix, t2))

    points1_hom = points1.T  # Shape: (2, N)
    points2_hom = points2.T  # Shape: (2, N)

    points_4d_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1_hom, points2_hom)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # Shape: (N, 3)

    return points_3d



# Task 4
def resection(
        image1: np.ndarray,
        image2: np.ndarray,
        camera_matrix: np.ndarray,
        matches: typing.Sequence[cv2.DMatch],
        points_3d: np.ndarray,
):
    pass
    # YOUR CODE HERE


def convert_to_world_frame(translation_vector, rotation_matrix):
    pass
    # YOUR CODE HERE


def visualisation(
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        camera_position3: np.ndarray,
        camera_rotation3: np.ndarray,
):

    def plot_camera(ax, position, direction, label):
        color_scatter = 'blue' if label != 'Camera 3' else 'green'
        # print(position)
        ax.scatter(position[0][0], position[1][0], position[2][0], color=color_scatter, s=100)
        color_quiver = 'red' if label != 'Camera 3' else 'magenta'

        ax.quiver(position[0][0], position[1][0], position[2][0], direction[0], direction[1], direction[2],
                  length=1, color=color_quiver, arrow_length_ratio=0.2)
        ax.text(position[0][0], position[1][0], position[2][0], label, color='black')


    camera_positions = [camera_position1, camera_position2, camera_position3]
    camera_directions = [camera_rotation1[:, 2], camera_rotation2[:, 2], camera_rotation3[:, 2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_camera(ax, camera_positions[0], camera_directions[0], 'Camera 1')
    plot_camera(ax, camera_positions[1], camera_directions[1], 'Camera 2')
    plot_camera(ax, camera_positions[2], camera_directions[2], 'Camera 3')

    initial_elev = 0
    initial_azim = 270

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=initial_elev, azim=initial_azim)

    ax.set_xlim([-1.50, 2.0])
    ax.set_ylim([-.50, 3.0])
    ax.set_zlim([-.50, 3.0])

    ax_elev_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    elev_slider = Slider(ax_elev_slider, 'Elev', 0, 360, valinit=initial_elev)

    ax_azim_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    azim_slider = Slider(ax_azim_slider, 'Azim', 0, 360, valinit=initial_azim)


    def update(val):
        elev = elev_slider.val
        azim = azim_slider.val
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    elev_slider.on_changed(update)
    azim_slider.on_changed(update)

    plt.show()


def main():
    image1 = cv2.imread('./images/image0.jpg')
    image2 = cv2.imread('./images/image1.jpg')
    image3 = cv2.imread('./images/image2.jpg')
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    camera_matrix = np.array(config["camera_matrix"], dtype=np.float32, order='C')

    key_points1, key_points2, matches_1_to_2 = get_matches(image1, image2)
    R2, t2, E = get_second_camera_position(key_points1, key_points2, matches_1_to_2, camera_matrix)
    triangulated_points = triangulation(
        camera_matrix,
        np.array([0, 0, 0]).reshape((3,1)),
        np.eye(3),
        t2,
        R2,
        key_points1,
        key_points2,
        matches_1_to_2
    )

    R3, t3 = resection(image1, image3, camera_matrix, matches_1_to_2, triangulated_points)
    camera_position1, camera_rotation1 = convert_to_world_frame(np.array([0, 0, 0]).reshape((3,1)), np.eye(3))
    camera_position2, camera_rotation2 = convert_to_world_frame(t2, R2)
    camera_position3, camera_rotation3 = convert_to_world_frame(t3, R3)
    visualisation(
        camera_position1,
        camera_rotation1,
        camera_position2,
        camera_rotation2,
        camera_position3,
        camera_rotation3
    )

if __name__ == "__main__":
    main()
