import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def generate_image(image_index2, y_true, y_pred, prediction_path):
    """
    Generate pendulum visualizations for ground truth and prediction.

    Parameters
    ----------
    image_index2 : int
        Index of the image.
    y_true : np.ndarray
        Ground truth angles.
    y_pred : np.ndarray
        Predicted angles.
    prediction_path : str
        Path to save the generated image.

    Returns
    -------
    None
    """
    image1 = plot_pendulum_visualization_by_angles(y_true[0], y_true[1])
    image2 = plot_pendulum_visualization_by_angles(y_pred[0], y_pred[1])
    plot_images_side_by_side(image1, image2, show_not_save=False,
                             path=os.path.join(prediction_path, "image_{:05d}.png".format(image_index2)))


def plot_pendulum_visualization_by_angles(angle1: np.ndarray, angle2: np.ndarray):
    """
    Plot a pendulum visualization based on given angles.

    Parameters
    ----------
    angle1 : np.ndarray
        First angle.
    angle2 : np.ndarray
        Second angle.

    Returns
    -------
    np.ndarray
        RGB image of the pendulum visualization.
    """
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    center = np.ones(2, dtype=np.uint8) * 500
    arm_length = 200

    # draw reference axis
    cv2.line(image, center, center + (0, arm_length // 2), (0, 255, 0), lineType=cv2.LINE_AA, thickness=2)

    cos_angle_1 = math.cos(angle1)
    sin_angle_1 = math.sin(angle1)
    dest_position_arm_1 = (int(center[0] + arm_length * sin_angle_1), int(center[1] + arm_length * cos_angle_1))

    # draw angle
    cv2.ellipse(image, center, (30, 30), 0, 90, 90 - math.degrees(angle1), (0, 255, 0))  # angle ellipse

    # draw reference axis
    cv2.line(image, dest_position_arm_1, (int(dest_position_arm_1[0] + arm_length // 2 * sin_angle_1),
                                          int(dest_position_arm_1[1] + arm_length // 2 * cos_angle_1)),
             (0, 255, 0), lineType=cv2.LINE_AA,
             thickness=2)

    # draw center
    cv2.circle(image, dest_position_arm_1, 6, (255, 0, 0), thickness=-1)

    # draw vector between joints
    cos_angle_2 = math.cos(angle1 + angle2)
    sin_angle_2 = math.sin(angle1 + angle2)
    dest_position_arm_2 = (int(dest_position_arm_1[0] + arm_length * sin_angle_2),
                           int(dest_position_arm_1[1] + arm_length * cos_angle_2))

    # draw angle
    cv2.ellipse(image, dest_position_arm_1, (30, 30), 0, 90 - math.degrees(angle1),
                90 - math.degrees(angle1) - math.degrees(angle2),
                (0, 255, 0))  # angle ellipse

    # draw center
    cv2.circle(image, dest_position_arm_2, 6, (255, 0, 0), thickness=-1)

    cv2.line(image, center, dest_position_arm_1, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.line(image, dest_position_arm_1, dest_position_arm_2, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def plot_images_side_by_side(image1, image2, show_not_save=True, path=None):
    """
    Plot two images side by side.

    Parameters
    ----------
    image1 : np.ndarray
        First image.
    image2 : np.ndarray
        Second image.
    show_not_save : bool, optional
        Flag to indicate whether to show or save the plot. Default is True (show).
    path : str, optional
        Path to save the plot. Applicable if show_not_save is False.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    axes[0].imshow(image1)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    fig.suptitle('Pendulum Visualizations')

    if show_not_save:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
