import cv2 as cv
import numpy as np
import colorsys
import calibration_params as cal_params
import os.path
from camera_controller import IDSCameraController


class HSVCollector:
    """
    A class for collecting HSV values for image segmentation.

    Attributes
    ----------
    frame_warped : numpy.ndarray
        The warped image used for HSV value collection.
    hsv_values : numpy.ndarray
        A matrix storing the collected HSV values.
    hsv_values_idx : int
        The current index in the `hsv_values` matrix.
    hsv_min_max : numpy.ndarray
        A matrix storing the minimum and maximum values for each HSV component.

    Methods
    -------
    collect_hsv_values(event, x, y, flags, params)
        Callback function for mouse events used to collect HSV values.
    get_hsv_range()
        Calculates the minimum and maximum HSV values from the collected HSV values.
    """
    def __init__(self):
        self.frame_warped = None
        self.hsv_values = np.empty((10, 3), dtype=np.uint8)
        self.hsv_values_idx = 0
        self.hsv_min_max = np.zeros((2, 3), dtype=int)
        if not os.path.exists(r"../CalibrationData"):
            os.makedirs(r"../CalibrationData")

    def collect_hsv_values(self, event, x, y, flags, params):
        """
        Callback function for mouse events used to collect HSV values.

        Parameters
        ----------
        event : int
            The mouse event.
        x : int
            The x-coordinate of the mouse position.
        y : int
            The y-coordinate of the mouse position.
        flags : int
            Additional/Optional flags for the mouse event.
        params : object
            Additional/Optional parameters for the callback function.
        """
        if event == cv.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
            if self.hsv_values_idx < 10:
                rgb = np.zeros(3, dtype=int)
                rgb[0] = self.frame_warped[y, x, 2]
                rgb[1] = self.frame_warped[y, x, 1]
                rgb[2] = self.frame_warped[y, x, 0]
                hsv = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
                self.hsv_values[self.hsv_values_idx, 0] = np.round(hsv[0] * 179).astype(np.uint8)
                self.hsv_values[self.hsv_values_idx, 1] = np.round(hsv[1] * 255).astype(np.uint8)
                self.hsv_values[self.hsv_values_idx, 2] = np.round(hsv[2] * 255).astype(np.uint8)
                self.hsv_values_idx += 1
            else:
                self.hsv_values_idx += 1

    def get_hsv_range(self):
        """
        Calculates the minimum and maximum HSV values from the collected HSV values.
        """
        # Calculate the minimum and maximum values for each component
        hsv_values = np.array(self.hsv_values[:self.hsv_values_idx])
        hue_min, hue_max = np.min(hsv_values[:, 0]), np.max(hsv_values[:, 0])
        if hue_max - hue_min > 120:
            print("WARNING: Collected values might span across the boundary of the HSV colour space."
                  "This may lead to performance issues. Check debug scripts.")
        saturation_min, saturation_max = np.min(hsv_values[:, 1]), np.max(hsv_values[:, 1])
        value_min, value_max = np.min(hsv_values[:, 2]), np.max(hsv_values[:, 2])

        # Ensure that the minimum and maximum values are within the valid HSV range
        self.hsv_min_max[0, 0] = max(0, hue_min - cal_params.hue_tolerance)
        self.hsv_min_max[1, 0] = min(179, hue_max + cal_params.hue_tolerance)
        self.hsv_min_max[0, 1] = max(0, saturation_min - cal_params.sat_tolerance)
        self.hsv_min_max[1, 1] = min(255, saturation_max + cal_params.sat_tolerance)
        self.hsv_min_max[0, 2] = max(0, value_min - cal_params.val_tolerance)
        self.hsv_min_max[1, 2] = min(255, value_max + cal_params.val_tolerance)


def main():
    # create instance of camera
    camera = IDSCameraController(param_file=r"../CameraParameters/cp_AngleDetection.ini")

    # create instances of HSV collector for red and green
    hsv_values_red = HSVCollector()
    hsv_values_green = HSVCollector()
    cancel = False

    # load existing warp matrix or use default one
    if os.path.exists("../CalibrationData/WarpMatrix.npy"):
        warp_matrix = np.load('../CalibrationData/WarpMatrix.npy')
    else:
        warp_matrix = np.array([[8.58769289e-01, -1.08283228e-02, -3.82004069e+02],
                                [2.48709061e-03, 8.58046261e-01, -8.09075447e+01],
                                [-7.62754274e-06, -1.20314697e-05, 1.00000000e+00]])
        print("WARNING: No 'WarpMatrix.npy'-file found. Use 'get_warp_matrix.py'-script to compute warp matrix. "
              "Continue with default matrix.")
    cv.namedWindow('ColourCalibration')

    # set mouse callback for collecting colour values for red
    cv.setMouseCallback('ColourCalibration', hsv_values_red.collect_hsv_values)

    # first while loop for collection colour values for red
    while True:
        # capture frame from camera input
        frame = camera.capture_image()

        # warp frame
        hsv_values_red.frame_warped = cv.warpPerspective(frame, warp_matrix,
                                                         (cal_params.warped_frame_side, cal_params.warped_frame_side))

        # print user instructions in window
        cv.rectangle(hsv_values_red.frame_warped, (0, 0), (cal_params.warped_frame_side, 40), (255, 255, 255), -1)
        label1 = "Move RED circle to different positions and click on it to collect colour values (max.10). " \
                 "Press 'n' to continue. Press 'q' to close window."
        label2 = "Collected Values: " + str(hsv_values_red.hsv_values_idx)
        cv.putText(hsv_values_red.frame_warped, label1, (2, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(hsv_values_red.frame_warped, label2, (2, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

        # show cv window
        cv.imshow('ColourCalibration', hsv_values_red.frame_warped)

        # check if user pressed q (quit) or n (next step)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cancel = True
            break
        elif key == ord('n') or hsv_values_red.hsv_values_idx >= 10:    # collect maximum 10 values
            break

    # save hsv values range to npy file
    if hsv_values_red.hsv_values_idx > 0:
        hsv_values_red.get_hsv_range()
        np.save('../CalibrationData/HsvMinMaxRed.npy', hsv_values_red.hsv_min_max)
        print(f'Calculated hsv-min/max values from {hsv_values_red.hsv_values_idx} collected values. '
              f'Saved to file "CalibrationData/HsvMinMaxRed.npy".')
    else:
        print("WARNING: No values saved for red.")

    # set mouse callback for collecting colour values for green
    cv.setMouseCallback('ColourCalibration', hsv_values_green.collect_hsv_values)

    # second while loop for collection colour values for green, only enter if user didn't quit before
    while True & cancel is False:
        # capture frame from camera input
        frame = camera.capture_image()

        # warp frame
        hsv_values_green.frame_warped = cv.warpPerspective(frame, warp_matrix,
                                                           (cal_params.warped_frame_side, cal_params.warped_frame_side))

        # print user instructions in window
        cv.rectangle(hsv_values_green.frame_warped, (0, 0), (cal_params.warped_frame_side, 40), (255, 255, 255), -1)
        label1 = "Move GREEN circle to different positions and click on it to collect colour values (max.10). " \
                 "Press 'n' to continue. Press 'q' to close window."
        label2 = "Collected Values: " + str(hsv_values_green.hsv_values_idx)
        cv.putText(hsv_values_green.frame_warped, label1, (2, 15), cv.FONT_HERSHEY_SIMPLEX,
                   0.4, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(hsv_values_green.frame_warped, label2, (2, 30), cv.FONT_HERSHEY_SIMPLEX,
                   0.4, (0, 0, 0), 1, cv.LINE_AA)

        # show cv window
        cv.imshow('ColourCalibration', hsv_values_green.frame_warped)

        # check if user pressed q (quit) or n (next step)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cancel = True
            break
        elif key == ord('n') or hsv_values_green.hsv_values_idx >= 10:  # collect maximum 10 values
            break

    # save hsv values range to npy file
    if hsv_values_green.hsv_values_idx > 0:
        hsv_values_green.get_hsv_range()
        np.save('../CalibrationData/HsvMinMaxGreen.npy', hsv_values_green.hsv_min_max)
        print(f'Calculated hsv-min/max values from {hsv_values_green.hsv_values_idx} collected values. '
              f'Saved to file "CalibrationData/HsvMinMaxGreen.npy".')
    else:
        print("WARNING: No values saved for green.")

    # third while loop as end screen
    while True & cancel is False:
        # capture frame from camera input
        frame = camera.capture_image()

        # warp frame
        frame_warped = cv.warpPerspective(frame, warp_matrix,
                                          (cal_params.warped_frame_side, cal_params.warped_frame_side))

        # print user instructions in window
        cv.rectangle(frame_warped, (0, 0), (cal_params.warped_frame_side, 40), (255, 255, 255), -1)
        label1 = "Colour calibration finished. Values saved successfully. Press 'q' to close window."
        cv.putText(frame_warped, label1, (2, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

        # show cv window
        cv.imshow('ColourCalibration', frame_warped)

        # check if user pressed q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all resources
    camera.close_camera_connection()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
