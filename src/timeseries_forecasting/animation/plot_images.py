import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def generate_image(image_index2: int, y_true: np.ndarray, y_pred: np.ndarray, prediction_path: str, mse: float):
    """
    Generate pendulum visualizations for ground truth and prediction.

    Parameters
    ----------
    mse: float
            Mean Squared Error of the data
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
    image1 = plot_pendulum_visualization_by_angles(y_true[0], y_true[1], y_true[2:])
    image2 = plot_pendulum_visualization_by_angles(y_pred[0], y_pred[1], y_pred[2:])
    plot_images_side_by_side(image1=image1, image2=image2, mse=mse,
                             path=os.path.join(prediction_path, "image_{:05d}.png".format(image_index2)))


def run_image_generation_trajectory(ground_truth_output: np.array, prediction_output: np.array, prediction_path: str,
                                    mse: float):
    image1 = plot_trajectory_by_angles(ground_truth_output)
    image2 = plot_trajectory_by_angles(prediction_output)
    plot_images_side_by_side(image1=image1, image2=image2, mse=mse,
                             path=os.path.join(prediction_path, "trajectory.png"))


def draw_wide_rod(image, start, end, color, thickness):
    mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    cv2.line(image, start, end, color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.circle(image, mid_point, thickness // 2, color, thickness=cv2.FILLED)


def draw_past_dots(image, center, arm_length, past_values, dot_size):
    array_length = int(len(past_values) / 2)
    for i in range(array_length):
        cos_angle_1 = math.cos(past_values[i])
        sin_angle_1 = math.sin(past_values[i])

        cos_angle_2 = math.cos(past_values[i] + past_values[i + 3])
        sin_angle_2 = math.sin(past_values[i] + past_values[i + 3])

        dest_position_arm_1 = (int(center[0] + arm_length * sin_angle_1), int(center[1] + arm_length * cos_angle_1))

        dest_position_arm_2 = (int(dest_position_arm_1[0] + arm_length * sin_angle_2),
                               int(dest_position_arm_1[1] + arm_length * cos_angle_2))

        # draw red dot at the connection point
        cv2.circle(image, dest_position_arm_2, dot_size, (0, 255, 0), thickness=cv2.FILLED)


def plot_trajectory_by_angles(angles):
    # Set up the image
    image_size = 1000
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    center = np.array([image_size // 2, image_size // 2], dtype=np.int32)
    real_life_arm_length = 115  # in mm

    scaling_factor = 200 / real_life_arm_length  # 200 is the desired length in the animation
    arm_length = int(real_life_arm_length * scaling_factor)

    # Initialize an empty NumPy array to store trajectory points
    points_array = np.zeros((len(angles) - 1, 2), dtype=np.int32)

    # Loop through angles to calculate trajectory points
    for i in range(len(angles) - 1):
        angle_1 = angles[i][0]
        angle_2 = angles[i][0] + angles[i][1]

        cos_angle_1 = math.cos(angle_1)
        sin_angle_1 = math.sin(angle_1)
        cos_angle_2 = math.cos(angle_2)
        sin_angle_2 = math.sin(angle_2)

        dest_position_arm_1 = (int(center[0] + arm_length * sin_angle_1), int(center[1] + arm_length * cos_angle_1))
        dest_position_arm_2 = (int(dest_position_arm_1[0] + arm_length * sin_angle_2),
                               int(dest_position_arm_1[1] + arm_length * cos_angle_2))

        # Save the calculated point in the NumPy array
        points_array[i] = dest_position_arm_2

    # Draw spline on the image
    cv2.polylines(image, [points_array], isClosed=False, color=(255, 255, 255), thickness=2)

    return image


def plot_pendulum_visualization_by_angles(angle1: np.ndarray, angle2: np.ndarray, past_values: np.ndarray):
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    center = np.ones(2, dtype=np.uint8) * 500
    real_life_arm_length = 115  # in mm
    scaling_factor = 200 / real_life_arm_length  # 200 is the desired length in the animation

    arm_length = int(real_life_arm_length * scaling_factor)
    color_white = (255, 255, 255)
    rod_thickness = int(26 * scaling_factor)
    dot_size = 18

    draw_past_dots(image, center, arm_length, past_values, dot_size)

    cos_angle_1 = math.cos(angle1)
    sin_angle_1 = math.sin(angle1)
    dest_position_arm_1 = (int(center[0] + arm_length * sin_angle_1), int(center[1] + arm_length * cos_angle_1))

    # draw wide, white rod between joints
    draw_wide_rod(image, center, dest_position_arm_1, color_white, rod_thickness)

    cos_angle_2 = math.cos(angle1 + angle2)
    sin_angle_2 = math.sin(angle1 + angle2)
    dest_position_arm_2 = (int(dest_position_arm_1[0] + arm_length * sin_angle_2),
                           int(dest_position_arm_1[1] + arm_length * cos_angle_2))

    # draw wide, white rod between joints
    draw_wide_rod(image, dest_position_arm_1, dest_position_arm_2, color_white, rod_thickness)

    # draw red dot at the connection point
    cv2.circle(image, dest_position_arm_1, dot_size, (0, 0, 255), thickness=cv2.FILLED)

    # draw red dot at the connection point
    cv2.circle(image, dest_position_arm_2, dot_size, (0, 255, 0), thickness=cv2.FILLED)

    # Display the values of angle1 and angle2 with larger font size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1.5
    font_thickness = 2
    cv2.putText(image, f"Angle 1: {angle1:.2f} radians", (10, 40),
                font, font_size, color_white, font_thickness, cv2.LINE_AA)
    cv2.putText(image, f"Angle 2: {angle2:.2f} radians", (10, 90),
                font, font_size, color_white, font_thickness, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def plot_images_side_by_side(image1: np.ndarray, image2: np.ndarray, mse: float, path=None):
    """
    Plot two images side by side.

    Parameters
    ----------
    mse: float
            Mean Squared Error of the data
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
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(image1)
    axes[0].set_title("Ground Truth", fontsize=10, y=-0.2)  # Adjust the y parameter
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title("Prediction", fontsize=10, y=-0.2)  # Adjust the y parameter
    axes[1].axis('off')

    # Display MSE value under the images
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.5, 0.02, f'Overall Mean Squared Error: {mse:.4f}', ha='center', fontsize=10)

    fig.suptitle('Pendulum Visualizations', fontsize=12)

    plt.savefig(path)
    plt.close(fig)
