import cv2
import os
from tqdm import tqdm


def create_video(image_folder, video_name, frame_rate=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video = cv2.VideoWriter(video_name,
                            fourcc=fourcc,
                            fps=frame_rate, frameSize=(width, height))

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    create_video(r"A:\Arbeit\Github\proj-chaotic-pendulum\Plots\2023-08-11_14-42-25_Plot", "230811_chaotic_pendulum.mp4", frame_rate=40)