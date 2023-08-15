from pyueye import ueye
import numpy as np
import cv2
import time
import os
import csv
import pandas as pd


class IDSCameraController:
    def __init__(self, param_file=r"../CameraParameters/cp_AngleDetection.ini"):
        # Variables
        self.h_cam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
        self.pcImageMemory = ueye.c_mem_p()
        self.mem_id = ueye.int()
        rect_aoi = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.start_time = time.time()
        self.timestamp = float("NaN")
        self.frame_rate = None
        self.video_has_ended = False
        m_ncolormode = ueye.INT()  # Y8/RGB16/RGB24/REG32
        # ---------------------------------------------------------------------------------------------------------------------

        # Starts the driver and establishes the connection to the camera
        ueye.is_InitCamera(self.h_cam, None)

        p_param = ueye.wchar_p()
        p_param.value = param_file
        ueye.is_ParameterSet(self.h_cam, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, p_param, 0)

        # Set display mode to DIB
        ueye.is_SetDisplayMode(self.h_cam, ueye.IS_SET_DM_DIB)

        # Set the right color mode
        ueye.is_GetColorDepth(self.h_cam, self.nBitsPerPixel, m_ncolormode)

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

        self.width = rect_aoi.s32Width
        self.height = rect_aoi.s32Height

        # Allocates an image memory for an image having its dimensions defined by
        # self.width and self.height and its color depth defined by self.nBitsPerPixel
        ueye.is_AllocImageMem(self.h_cam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.mem_id)
        ueye.is_SetImageMem(self.h_cam, self.pcImageMemory, self.mem_id)
        ueye.is_SetColorMode(self.h_cam, m_ncolormode)

        # Activates the camera's live video mode (free run mode)
        ueye.is_CaptureVideo(self.h_cam, ueye.IS_DONT_WAIT)
        # Enables the queue mode for existing image memory sequences
        ueye.is_InquireImageMem(self.h_cam, self.pcImageMemory, self.mem_id, self.width, self.height,
                                self.nBitsPerPixel, self.pitch)
        
    def capture_image(self):
        # Read image from camera
        array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
        bytes_per_pixel = int(self.nBitsPerPixel / 8)
    
        # Reshape data into numpy array
        frame = np.reshape(array, (self.height.value, self.width.value, bytes_per_pixel))

        new_timestamp = time.time() - self.start_time
        self.frame_rate = 1 / (new_timestamp - self.timestamp)
        self.timestamp = new_timestamp
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def close_camera_connection(self):
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.mem_id)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.h_cam)


class WebcamCameraController:
    def __init__(self, cam_idx=0):
        self.vid = cv2.VideoCapture(cam_idx)

    def capture_image(self):
        ret, frame = self.vid.read()
        return frame

    def close_camera_connection(self):
        self.vid.release()


class ImageStreamController:
    def __init__(self, video_path):
        self.video_path = video_path
        self.images_in_directory = [f for f in os.listdir(self.video_path) if ".jpg" in f]

        self.current_idx = 0
        self.timestamp = float("NaN")
        self.frame_rate = None
        self.start_time = time.time()
        self.video_has_ended = False

        log_file_path = [f for f in os.listdir(self.video_path) if ".csv" in f]
        self.log_file = None
        if len(log_file_path):
            csv_file_path = os.path.join(self.video_path, log_file_path[0])
            self.log_file = pd.read_csv(csv_file_path,
                                        delimiter=";")

    def capture_image(self):
        # Get image by idx and preprocess
        frame = cv2.imread(os.path.join(self.video_path, self.images_in_directory[self.current_idx]))

        if self.log_file is not None and any(self.log_file):
            if not np.isnan(self.log_file.loc[self.current_idx]["Time"]):
                new_timestamp = self.log_file.loc[self.current_idx]["Time"]
                self.frame_rate = 1 / (new_timestamp - self.timestamp)
                self.timestamp = new_timestamp
        else:
            new_timestamp = time.time() - self.start_time
            self.frame_rate = 1 / (new_timestamp - self.timestamp)
            self.timestamp = new_timestamp

        # Increase idx by 1
        self.current_idx += 1
        if self.current_idx >= len(self.images_in_directory):
            self.video_has_ended = True
            self.current_idx = 0
            self.timestamp = float("NaN")
            self.frame_rate = float("NaN")
            self.start_time = time.time()
        return frame

    def close_camera_connection(self):
        pass


def main():
    cam = IDSCameraController()
    for i in range(10000):
        frame = cam.capture_image()
        cv2.namedWindow("TestOutput", cv2.WINDOW_NORMAL)
        cv2.imshow("TestOutput", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.close_camera_connection()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
