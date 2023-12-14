import pandas as pd
import matplotlib.pyplot as plt
import argparse
from camera_controller import ImageStreamController
import math
import datetime
import os
import cv2
from matplotlib.collections import LineCollection


def main(path):
    # create timestamp for file names
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = f"../Plots/{timestamp}_Plot"
    idx = 0

    # create path for saving files
    if not os.path.exists(directory):
        os.makedirs(directory)

    camera = ImageStreamController(path)
    x = []
    y1 = []
    y1_line = []
    y2 = []
    y2_line = []
    while True:
        # Create a figure with two horizontal subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

        # Plot the image in the first subplot
        # Replace 'image_path.png' with the path to your image file
        ax1.imshow(cv2.cvtColor(camera.capture_image(), cv2.COLOR_BGR2RGB))
        ax1.set_title('Image')
        x.append(camera.timestamp)
        y1.append(camera.log_file.loc[camera.current_idx]['Angle1'])
        y2.append(camera.log_file.loc[camera.current_idx]['Angle2'])
        if len(y1) > 2:
            if (y1[-2] < -2.5 and y1[-1] > 2.5) or (y1[-2] > 2.5 and y1[-1] < -2.5):
                y1_line.append(None)
            else:
                y1_line.append(camera.log_file.loc[camera.current_idx]['Angle1'])

            if (y2[-2] < -2.5 and y2[-1] > 2.5) or (y2[-2] > 2.5 and y2[-1] < -2.5):
                y2_line.append(None)
            else:
                y2_line.append(camera.log_file.loc[camera.current_idx]['Angle2'])
        else:
            y1_line.append(camera.log_file.loc[camera.current_idx]['Angle1'])
            y2_line.append(camera.log_file.loc[camera.current_idx]['Angle2'])

        # Plot the line plot in the second subplot
        ax2.scatter(x, y1, color='blue', marker='.')
        ax2.plot(x, y1_line, color='blue')
        ax2.set_title('Pendulum Arm 1 Angle')
        ax2.set_xlabel('Time in s')
        ax2.set_ylabel('Angle in rad')
        ax2.grid('on')
        ax2.set_xlim([camera.timestamp - 3, camera.timestamp + 3])  # camera.log_file['Time'].max()])
        ax2.set_ylim([-math.pi - 0.2, math.pi + 0.2])

        ax3.scatter(x, y2, color='red', marker='.')
        ax3.plot(x, y2_line, color='red')
        ax3.set_title('Pendulum Arm 2 Angle')
        ax3.set_xlabel('Time in s')
        ax3.set_ylabel('Angle in rad')
        ax3.grid('on')
        ax3.set_xlim([camera.timestamp - 3, camera.timestamp + 3])  # camera.log_file['Time'].max()])
        ax3.set_ylim([-math.pi - 0.2, math.pi + 0.2])

        # Adjust spacing between subplots
        # plt.tight_layout()
        plt.savefig(os.path.join(directory, "frame_{:05d}.jpg".format(idx)))
        idx += 1

        plt.close(fig)
        # Show the plot
        # plt.show()

        if camera.video_has_ended:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('-dp', '--data_path', type=str, default=None, required=False)
    args = parser.parse_args()

    main(args.data_path)
