import cv2 as cv
import detection_params as params
from camera_controller import IDSCameraController

# --------------------------------------------------IMPORTANT NOTE-----------------------------------------------------
# This script is intended only for debugging purposes. It can be used to visualize the hsv colour masks
# that are used for the contour detection. This makes it easier to track possible deviations in the calibration process.
# ---------------------------------------------------------------------------------------------------------------------


def main():
    # initialize camera with parameter file used for angle detection
    camera = IDSCameraController(param_file=r"../CameraParameters/cp_AngleDetection.ini")

    while True:
        # capture frame
        frame = camera.capture_image()

        # warp frame
        frame_warped = cv.warpPerspective(frame, params.warp_matrix,
                                          (params.warped_frame_side, params.warped_frame_side))

        # Convert frame to hsv-colour-space for colour filtering
        frame_hsv = cv.cvtColor(frame_warped, cv.COLOR_BGR2HSV)

        # Preparing mask to overlay
        mask_green = cv.inRange(frame_hsv, params.green_min, params.green_max)
        mask_red = cv.inRange(frame_hsv, params.red_min, params.red_max)

        # Find contours for red and green shapes in frame
        contours_green, _ = cv.findContours(mask_green, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours_red, _ = cv.findContours(mask_red, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        # filter contours by size and draw them on the frame
        for i, c in enumerate(contours_green):
            area = cv.contourArea(c)
            if area < params.area_min or params.area_max < area:
                continue
            cv.drawContours(frame_warped, contours_green, i, (0, 0, 255), 2)

        for i, c in enumerate(contours_red):
            area = cv.contourArea(c)
            if area < params.area_min or params.area_max < area:
                continue
            cv.drawContours(frame_warped, contours_red, i, (0, 0, 255), 2)

        # open windows for visualization
        cv.imshow('Detected Contours', frame_warped)
        cv.imshow('Mask green', mask_green)
        cv.imshow('Mask red', mask_red)

        # check if user wants to quit by hitting 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    camera.close_camera_connection()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
