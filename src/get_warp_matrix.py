import cv2 as cv
import os.path
import numpy as np
import calibration_params as cal_params
from camera_controller import IDSCameraController


class WarpMatrixCalculator:
    """
    Class for calculating the warp matrix for a camera.

    Attributes
    ----------
    camera: IDSCameraController
        The camera object used for capturing the video stream.

    Methods
    -------
    get_matrix()
        Calculates the warp matrix based on the detected ArUco markers in the captured video stream.
    """
    def __init__(self):
        # initialize camera with marker detection parameter file
        self.camera = IDSCameraController(param_file=r"../CameraParameters/cp_DetectMarker.ini")

        #create file path for saving the calibration data
        if not os.path.exists(r"../CalibrationData"):
            os.makedirs(r"../CalibrationData")

    def get_matrix(self):
        """
        Calculates the warp matrix using ArUco markers.

        Returns
        -------
        warp_matrix : numpy.ndarray
            The calculated warp matrix.
        """
        print("Calculate warp matrix...")
        frame_count = 0
        # search for aruco markers in frames until all 4 markers are found or max values for checked frames is reached
        while True:
            frame_count += 1
            frame = self.camera.capture_image()
            corners, ids, _ = cv.aruco.detectMarkers(frame, cal_params.aruco_dict, parameters=cal_params.aruco_params)
            if contains_zero_to_three(ids):
                break
            elif frame_count >= 1000:
                if ids is None:
                    ids_count = 0
                else:
                    ids_count = len(ids)
                self.camera.close_camera_connection()
                raise RuntimeError(f"Found only {ids_count} of 4 markers after {frame_count} frames. Check conditions.")

        warp_matrix = calc_matrix(corners, ids)

        # save warp matrix to numpy file
        np.save('../CalibrationData/WarpMatrix.npy', warp_matrix)
        self.camera.close_camera_connection()
        print("Calculation of warp matrix finished successfully.")


# Check if all Markers (IDs 0-3) were found
def contains_zero_to_three(ids):
    """
    Checks if the given ArUco marker IDs contain values between 0 and 3 (inclusive).

    Parameters
    ----------
    ids : numpy.ndarray
        Array of ArUco marker IDs.

    Returns
    -------
    bool
        True if all of the values between 0 and 3 are found in the array, False otherwise.
    """
    if ids is None:
        return False
    return np.isin([0, 1, 2, 3], ids).all()


def calc_matrix(corners, ids):
    """
    Calculates a perspective warp matrix based on the corners and IDs of four ArUco markers.

    Parameters
    ----------
    corners : list of numpy.ndarray
        List of 2D arrays of shape (4,2) containing the coordinates of the corners of detected ArUco markers.
    ids : numpy.ndarray
        Array of integer IDs of the detected ArUco markers.

    Returns
    -------
    warp_matrix : numpy.ndarray
        3x3 perspective warp matrix used for warping the input image to a top-down view.
    """
    # Find index of specific markers in ID-List
    marker_idxs = [np.where(ids == i)[0][0] for i in range(4)]

    # Extract corner points-coordinates corresponding to the four markers with the desired IDs
    up_left = corners[marker_idxs[0]][0][2]
    up_right = corners[marker_idxs[1]][0][3]
    down_right = corners[marker_idxs[2]][0][3]
    down_left = corners[marker_idxs[3]][0][2]

    # Define source and destination points for perspective transform
    src_pts = np.float32([up_left, up_right, down_right, down_left])
    dst_pts = np.float32([[0, 0], [cal_params.warped_frame_side, 0],
                          [cal_params.warped_frame_side, cal_params.warped_frame_side],
                          [0, cal_params.warped_frame_side]])

    # Compute perspective transform matrix
    warp_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    return warp_matrix


def main():
    matrix_calc = WarpMatrixCalculator()
    matrix_calc.get_matrix()


if __name__ == '__main__':
    main()
