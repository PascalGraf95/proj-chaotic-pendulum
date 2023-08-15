import cv2
import numpy as np
import math
from time_series_forecasting import *
from tensorflow import keras
import matplotlib.pyplot as plt
import os


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


def plot_pendulum_by_angles(angle1, angle2):
    arm1_image = cv2.imread("../images/Arm1.png", cv2.IMREAD_UNCHANGED)
    arm2_image = cv2.imread("../images/Arm2.png", cv2.IMREAD_UNCHANGED)
    print(arm1_image.shape, arm2_image.shape)

    background = np.zeros(arm1_image.shape)
    print(background.shape)

    arm1_image = rotate_image(arm1_image, angle1)
    arm2_image = rotate_image(arm2_image, angle2)
    print(math.cos(math.radians(angle1))*(arm2_image.shape[0]//2))
    print(math.sin(math.radians(angle1))*(arm2_image.shape[0]//2))
    arm2_image = translate_image(arm2_image, (1-int(math.cos(math.radians(angle1)))*400),
                                 int(math.sin(math.radians(angle1))*150))
    image = overlay_images(background, arm1_image)
    image = overlay_images(image, arm2_image)






    # display the image
    cv2.imshow("Composited image", image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_pendulum_visualization_by_angles(angle1, angle2):
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    center = np.ones(2, dtype=np.uint8)*500
    arm_length = 200

    # draw reference axis
    cv2.line(image, center, center + (0, arm_length//2), (0, 255, 0), lineType=cv2.LINE_AA, thickness=2)

    cos_angle_1 = math.cos(angle1)
    sin_angle_1 = math.sin(angle1)
    dest_position_arm_1 = (int(center[0] + arm_length*sin_angle_1), int(center[1] + arm_length*cos_angle_1))

    # draw angle
    cv2.ellipse(image, center, (30, 30), 0, 90, 90-math.degrees(angle1), (0, 255, 0))  # angle ellipse

    # draw reference axis
    cv2.line(image, dest_position_arm_1, (int(dest_position_arm_1[0] + arm_length//2*sin_angle_1),
                                          int(dest_position_arm_1[1] + arm_length//2*cos_angle_1)),
             (0, 255, 0), lineType=cv2.LINE_AA,
             thickness=2)

    # draw center
    cv2.circle(image, dest_position_arm_1, 6, (255, 0, 0), thickness=-1)


    # draw vector between joints
    cos_angle_2 = math.cos(angle1 + angle2)
    sin_angle_2 = math.sin(angle1 + angle2)
    dest_position_arm_2 = (int(dest_position_arm_1[0] + arm_length*sin_angle_2),
                           int(dest_position_arm_1[1] + arm_length*cos_angle_2))

    # draw angle
    cv2.ellipse(image, dest_position_arm_1, (30, 30), 0, 90-math.degrees(angle1),
                90-math.degrees(angle1) - math.degrees(angle2),
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




if __name__ == '__main__':
    data = load_data_from_csv(r"A:\Arbeit\Github\proj-chaotic-pendulum\DataRecords\2023-08-11_14-29-45_DemoFolder\2023-08-11_14-29-45_log.csv")
    x_train, x_val, x_test, y_train, y_val, y_test = filter_and_preprocess_data(data)
    model = keras.models.load_model(r"A:\Arbeit\Github\proj-chaotic-pendulum\src\timeseries_forecasting\model_checkpoint.h5")

    for idx1, (x_batch, y_batch) in enumerate(zip(x_train, y_train)):
        if not os.path.isdir(os.path.join("prediction_output", "prediction_{:05d}".format(idx1))):
            os.makedirs(os.path.join("prediction_output", "prediction_{:05d}".format(idx1)))
        y_predictions = model.predict(np.expand_dims(x_batch, axis=0))
        # for x in x_batch:
        #     image1 = plot_pendulum_visualization_by_angles(x[0], x[1])

        for idx2, (y_true, y_pred) in enumerate(zip(y_batch.reshape(-1, 2), y_predictions.reshape(-1, 2))):
            image1 = plot_pendulum_visualization_by_angles(y_true[0], y_true[1])
            image2 = plot_pendulum_visualization_by_angles(y_pred[0], y_pred[1])
            plot_images_side_by_side(image1, image2, show_not_save=False, path=os.path.join("prediction_output", "prediction_{:05d}/image_{:05d}.png".format(idx1, idx2)))