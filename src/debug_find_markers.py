import cv2 as cv
import calibration_params
from camera_controller import IDSCameraController

# --------------------------------------------------IMPORTANT NOTE-----------------------------------------------------
# This script is intended only for debugging purposes. It can be used to visualize the aruco markers that can be found
# in the frame with the current setup. This makes it easier to track possible deviations in the calibration process.
# ---------------------------------------------------------------------------------------------------------------------


def main():
    # initialize the camera with the detect marker parameter file (that is also used in the calibration process).
    camera = IDSCameraController(param_file=r"../CameraParameters/cp_DetectMarker.ini")

    while True:
        # capture a frame
        frame = camera.capture_image()

        # detect the aruco markers
        corners, ids, _ = cv.aruco.detectMarkers(frame, calibration_params.aruco_dict,
                                                 parameters=calibration_params.aruco_params)

        # draw marker visu in frame
        cv.aruco.drawDetectedMarkers(frame, corners)

        # show frame
        cv.namedWindow('Show Marker', cv.WINDOW_NORMAL)
        cv.imshow('Show Marker', frame)

        # check if user wants to quit by hitting 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    camera.close_camera_connection()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
