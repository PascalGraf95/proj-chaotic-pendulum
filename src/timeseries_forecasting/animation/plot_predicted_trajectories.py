import cv2
import math

import numpy
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
from PIL import Image
import glob
from multiprocessing import Pool
from functools import partial

dir = os.path.abspath(os.path.dirname(__file__))


def load_data_from_csv(path):
    return pd.read_csv(path, delimiter=",")


def filter_and_preprocess_data(data: pd.DataFrame, sequence_length: int, output_length: int):
    data['Sin_Angle1'] = np.sin(data['Angle1'])
    data['Cos_Angle1'] = np.cos(data['Angle1'])

    data['Sin_Angle2'] = np.sin(data['Angle2'])
    data['Cos_Angle2'] = np.cos(data['Angle2'])

    sin_angle1 = data["Sin_Angle1"].to_numpy()
    cos_angle1 = data["Cos_Angle1"].to_numpy()

    sin_angle2 = data["Sin_Angle2"].to_numpy()
    cos_angle2 = data["Cos_Angle2"].to_numpy()

    numpy_data = np.column_stack((sin_angle1, cos_angle1, sin_angle2, cos_angle2))
    print("DATA SHAPE: {}".format(numpy_data.shape))

    x_sample = numpy_data[:sequence_length]
    y_sample = numpy_data[sequence_length:sequence_length + output_length].flatten()

    x = np.array(x_sample)
    y = np.array(y_sample)

    return x, y


def reconstruct_radian(array: numpy.ndarray) -> numpy.ndarray:
    reconstructed1 = np.arctan2(array[:, 0], array[:, 1])
    reconstructed2 = np.arctan2(array[:, 2], array[:, 3])
    return np.column_stack((reconstructed1, reconstructed2))


def overlay_images(background, overlay):
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:, :, 3] / 255.0
    alpha_overlay = overlay[:, :, 3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[:, :, color] = alpha_overlay * overlay[:, :, color] + \
                                  alpha_background * background[:, :, color] * (1 - alpha_overlay)

    # set adjusted alpha and denormalize back to 0-255
    background[:, :, 3] = (1 - (1 - alpha_overlay) * (1 - alpha_background)) * 255
    return background


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def translate_image(image, x, y):
    rows, cols = image.shape[:2]
    mat = np.float32([[1, 0, y], [0, 1, x]])
    result = cv2.warpAffine(image, mat, (cols, rows))
    return result


def plot_pendulum_by_angles(angle1: numpy.ndarray, angle2: numpy.ndarray):
    arm1_image = cv2.imread("../images/Arm1.png", cv2.IMREAD_UNCHANGED)
    arm2_image = cv2.imread("../images/Arm2.png", cv2.IMREAD_UNCHANGED)
    print(arm1_image.shape, arm2_image.shape)

    background = np.zeros(arm1_image.shape)
    print(background.shape)

    arm1_image = rotate_image(arm1_image, angle1)
    arm2_image = rotate_image(arm2_image, angle2)
    print(math.cos(math.radians(angle1)) * (arm2_image.shape[0] // 2))
    print(math.sin(math.radians(angle1)) * (arm2_image.shape[0] // 2))
    arm2_image = translate_image(arm2_image, (1 - int(math.cos(math.radians(angle1))) * 400),
                                 int(math.sin(math.radians(angle1)) * 150))
    image = overlay_images(background, arm1_image)
    image = overlay_images(image, arm2_image)

    # display the image
    cv2.imshow("Composited image", image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_pendulum_visualization_by_angles(angle1: numpy.ndarray, angle2: numpy.ndarray):
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


def build_animation(folder_path):
    # TODO: Refactor to class to make directory handling easier
    # TODO: Check Typehinting input and return for every function
    # Use glob to find all PNG files in the specified directory
    image_paths = glob.glob(os.path.join(os.path.join(folder_path, "prediction"), '*.png'))

    # Create a list to store image objects
    images = []

    # Open each image and append it to the list
    for image_path in tqdm(image_paths):
        img = Image.open(image_path)
        images.append(img)

    # Specify the output GIF file path
    output_gif_path = os.path.join(folder_path, "output.gif")

    # Save the GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=5,  # Duration between frames in milliseconds
        loop=1,  # 0 means infinite loop
    )


def generate_image(idx2, y_true, y_pred, output_folder):
    image1 = plot_pendulum_visualization_by_angles(y_true[0], y_true[1])
    image2 = plot_pendulum_visualization_by_angles(y_pred[0], y_pred[1])
    plot_images_side_by_side(image1, image2, show_not_save=False,
                             path=os.path.join(output_folder, "image_{:05d}.png".format(idx2)))


def generate_animation_parallel(data_path,
                                model_path=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\training\model_checkpoints\2023-12-09_19-52-31_rnn_model_rnn_model.h5",
                                output_length=300,
                                sequence_length=50):
    data = load_data_from_csv(data_path)
    prediction_input, ground_truth_output = filter_and_preprocess_data(data, output_length=output_length,
                                                                       sequence_length=sequence_length)
    model = keras.models.load_model(model_path)

    folder_path = os.path.join(dir, "prediction_output")

    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(os.path.join(folder_path, "prediction"))
    prediction_output = model.predict(np.expand_dims(prediction_input, axis=0))

    prediction_output_reconstructed = reconstruct_radian(prediction_output.reshape(-1, 4))
    ground_truth_output_reconstructed = reconstruct_radian(ground_truth_output.reshape(-1, 4))

    # Add zeros for the non-prediction time
    prediction_output_final = np.concatenate((np.zeros((sequence_length, 2)), prediction_output_reconstructed),
                                             axis=0)

    ground_truth_output_final = np.concatenate(
        (reconstruct_radian(prediction_input), ground_truth_output_reconstructed), axis=0)

    # Use multiprocessing to generate images in parallel
    output_folder = os.path.join(dir, "prediction_output/prediction")
    with Pool() as pool:
        args_list = [(idx2, y_true, y_pred, output_folder) for idx2, (y_true, y_pred) in
                     enumerate(zip(ground_truth_output_final, prediction_output_final))]
        pool.starmap(generate_image, args_list)

    build_animation(folder_path)


if __name__ == '__main__':
    generate_animation_parallel(
        data_path=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\data\processed\12.csv",
        model_path=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\training\model_checkpoints\2023-12-09_19-52-31_rnn_model_rnn_model.h5")
