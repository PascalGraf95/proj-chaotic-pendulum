import cv2
import math
import numpy as np

def create_visualization(frame, contour_red_id, contour_green_id, contours_red, contours_green,
                         angle_1, angle_2, angular_vel_1, angular_vel_2, pos_a, pos_b):
    """
    Visualizes the live results of angle detection.

    Parameters
    ----------
    """
    # visualization of the contours used for angle computation
    # draw contours only when function get_center_position returned a valid value
    if contour_red_id != -1:
        cv2.drawContours(frame, contours_red, contour_red_id, (0, 0, 255), 1)
    if contour_green_id != -1:
        cv2.drawContours(frame, contours_green, contour_green_id, (0, 0, 255), 1)

    # if angle_1 could be detected set text labels and draw vectors for angle visualization
    if not math.isnan(angle_1):
        label11 = str(round((np.rad2deg(angle_1)), 1)) + " deg "
        # exclude that angle was detected but angular velocity could not be calculated due to missing values
        if not math.isnan(angular_vel_1):
            label12 = str(round(np.rad2deg(angular_vel_1), 1)) + " deg/s"
        else:
            label12 = "NaN"

        # draw vector between joints
        cv2.line(frame, pos_a, pos_b, (255, 0, 0), thickness=1)
        # draw reference axis
        cv2.line(frame, pos_a, pos_a + get_axis_visu(vec_ref_1), (0, 255, 0),
                lineType=cv2.LINE_8, thickness=1)
        # draw angle visu
        if check_side(pos_a, pos_a + (0, 100), pos_a + vec_ref_1) == (-1 | 0):
            angular_offset = np.rad2deg(calc_angle(np.array((0, 1)), detection_params.vec_ref_1))
        else:
            angular_offset = -np.rad2deg(calc_angle(np.array((0, 1)), detection_params.vec_ref_1))
        start_angle = 90 - angular_offset
        cv2.ellipse(frame, pos_A, (45, 45), 0, start_angle, start_angle - np.rad2deg(angle_1),
                   (0, 255, 0))  # angle ellipse
    else:
        label11 = "NaN"
        label12 = "NaN"

    # if angle_2 could be detected set text labels and draw vectors for angle visualization
    if not math.isnan(angle_2):
        label21 = str(round(np.rad2deg(angle_2), 1)) + " deg "
        # exclude that angle was detected but angular velocity could not be calculated due to missing values
        if not math.isnan(angular_vel_2):
            label22 = str(round(np.rad2deg(angular_vel_2), 1)) + " deg/s"
        else:
            label22 = "NaN"
        if vis_vectors:
            # draw vector between joints
            cv2.line(frame, pos_B, pos_C, (255, 0, 0), thickness=1)
            # distinguish by chosen definition
            if definition == 0:  # relative definition
                # draw reference axis
                cv2.line(frame, pos_B, pos_B + get_axis_visu(vec_ref_2), (0, 255, 0),
                        lineType=cv2.LINE_8, thickness=1)
                # draw angle visu
                cv2.ellipse(frame, pos_B, (45, 45), 0, 90 - np.rad2deg(angle_1),
                           90 - np.rad2deg(angle_1) - np.rad2deg(angle_2), (0, 255, 0))
            elif definition == 1:  # absolute definition
                # draw reference axis
                cv2.line(frame, pos_B, pos_B + get_axis_visu(vec_ref_2),
                        (0, 255, 0), lineType=cv2.LINE_8,
                        thickness=1)
                # draw angle visu
                if check_side(pos_B, pos_B + (0, 100), pos_B + vec_ref_2) == (-1 | 0):
                    angular_offset = np.rad2deg(calc_angle(np.array((0, 1)), vec_ref_2))
                else:
                    angular_offset = -np.rad2deg(calc_angle(np.array((0, 1)), vec_ref_2))
                start_angle = 90 - angular_offset
                cv2.ellipse(frame, pos_B, (45, 45), 0, start_angle,
                           start_angle - np.rad2deg(angle_2), (0, 255, 0))
    else:
        label21 = "NaN"
        label22 = "NaN"

    # print measured values in opencv2 window using defined labels
    # white background
    cv2.rectangle(frame, (0, 0), (235, 80), (255, 255, 255), -1)

    # splitting in multiple function calls to achieve fixed positions (tabular)
    cv2.putText(frame, "angle_1", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "=", (79, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, label11, (102, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "velocity_1", (2, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "=", (79, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, label12, (102, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "angle_2", (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "=", (79, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, label21, (102, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "velocity_2", (2, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "=", (79, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, label22, (102, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # draw found center position of contours
    cv2.circle(frame, pos_A, 2, (255, 0, 0), thickness=3)
    if not math.isnan(pos_B[0]):
        cv2.circle(frame, pos_B, 2, (255, 0, 0), thickness=3)
    if not math.isnan(pos_C[0]):
        cv2.circle(frame, pos_C, 2, (255, 0, 0), thickness=3)

    cv2.rectangle(frame, (detection_params.warped_frame_side-130, 0),
                 (detection_params.warped_frame_side, 20), (255, 255, 255), -1)
    cv2.putText(frame, f"t = {timestamp}s", (detection_params.warped_frame_side-128, 16),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # show frame in pop-up window
    cv2.namedWindow('Angle Detection', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Angle Detection', frame)