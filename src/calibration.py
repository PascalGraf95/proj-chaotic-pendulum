import get_warp_matrix
import get_hsv_masks
import get_reference_axis
import calibration_params as cal_params
from camera_controller import IDSCameraController
from angle_detection import AngleDetector
import cv2 as cv
import numpy as np


def main():
    print("Start system calibration.")
    # Calculate Warp Matrix
    get_warp_matrix.main()

    # Start script for collecting hsv values of coloured dots
    get_hsv_masks.main()

    # User instruction to ensure that the pendulum no longer moves during alignment calibration
    blank = np.zeros((50, 550, 3), np.uint8)
    label = "HSV mask calibration finished. Stop pendulum and press 'q' to continue calibration."
    cv.putText(blank, label, (2, 25), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

    while True:
        cv.imshow("User instruction", blank)
        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cv.destroyAllWindows()

    # perform alignment calibration
    get_reference_axis.main()
    print("System calibration finished successfully.")

    # Start visualization of detection so that the user can check the calibration result
    # Initialize camera
    camera = IDSCameraController(param_file=r"../CameraParameters/cp_AngleDetection.ini")
    # Initialize AngleDetector object,
    measurement = AngleDetector(camera, definition=0)

    while True:
        measurement.get_angle()
        measurement.get_angular_vel()

        # additional note for the user
        cv.rectangle(measurement.visu, (0, 0), (cal_params.warped_frame_side, 80), (255, 255, 255), -1)
        label = "Check calibration result. Use the debug scripts to track possible deviations." \
                " Press 'q' to close window."
        cv.putText(measurement.visu, label, (cal_params.warped_frame_side-670, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)

        measurement.visualize()

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all resources
    # Close visualization
    cv.destroyAllWindows()

    # End camera connection
    camera.close_camera_connection()


if __name__ == '__main__':
    main()
