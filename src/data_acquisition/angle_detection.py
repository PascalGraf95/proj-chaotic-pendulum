import importlib
import cv2
import numpy as np
import math
from src.calibration import detection_params
import time

"""
# Create a black background image
size = 50
circle_image = np.zeros((size, size), dtype=np.uint8)

# Define the circle parameters
center = (size//2, size//2)
radius = size//2
color = 255 # White color
thickness = -1  # Filled circle

# Draw the white circle on the black background
cv2.circle(circle_image, center, radius, color, thickness)

cv2.imshow("NAME", circle_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


def norm_vector(vector):
    """
    Returns the normalized version of the input vector.

    Parameters
    ----------
    vector : numpy.ndarray
        An array representing a vector.

    Returns
    -------
    numpy.ndarray
        A normalized vector with the same direction as the input vector.
    """
    return vector / np.linalg.norm(vector)


def calc_angle(v1, v2):
    """
    Calculates the angle in radians between two vectors.

    Parameters
    ----------
    v1 :  numpy.ndarray
        A numpy array representing the first vector.
    v2 :  numpy.ndarray
        A numpy array representing the second vector.

    Returns
    -------
    float
        The angle in radians between the two vectors.
    """
    # Normalize vectors
    v1_u = norm_vector(v1)
    v2_u = norm_vector(v2)

    # Dot-product of normalized vectors, limited to values between -1 and 1, calculate angle with arc-cos
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def check_side(point_a, point_b, point_c):
    """
    Determines on which side of the line between points A and B the point C is located.

    Parameters
    ----------
    point_a : numpy.ndarray
        Coordinates of point A in the form [x, y].
    point_b : numpy.ndarray
        Coordinates of point B in the form [x, y].
    point_c : numpy.ndarray
        Coordinates of point C in the form [x, y].

    Returns
    -------
    int
        -1 if C is on the right side of the line, +1 if C is on the left side, 0 if C is on the line.
    """
    # Calculate determinant of 2x2-matrix built with the 3 points
    position = np.sign((point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (point_b[1] - point_a[1])
                       * (point_c[0] - point_a[0]))
    return position


def get_contour_center(pts):
    """
    Calculates the center point of a contour.

    Parameters
    ----------
    pts : numpy.ndarray
        A list of points that make up the contour.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the center point of the contour.
    """
    moments = cv2.moments(pts)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    center = np.array([cx, cy])
    return center


def get_axis_visu(axis):
    """
    Computes the visual representation of an axis vector for display.

    Parameters
    ----------
    axis : numpy.ndarray
        A vector representing an axis.

    Returns
    -------
    numpy.ndarray
        A vector representing the visual representation of the input axis, with selected length and integer coordinates.
    """
    axis = norm_vector(axis) * detection_params.visu_axis_length
    return axis.astype(int)


def get_center_position(contours):
    """
    Computes the center of the contour that is searched for.

    Parameters
    ----------
    contours : numpy.ndarray
        A list of contours.

    Returns
    -------
    position : numpy.ndarray
        The center point of the searched for contour.  If the contour was not found, the position is set to NaN.
    found : bool
        A boolean value, indicating if the contour was found.
    contour_id : int
        The index of the contour used for center calculation in the contours array.
    """
    contours_cnt = 0
    position = np.zeros(2)
    contour_id = -1
    for i, c in enumerate(contours):
        # Filter contours by size of the area they are covering to make sure only the coloured circles are detected
        area = cv2.contourArea(c)
        if area < detection_params.area_min or detection_params.area_max < area:
            continue
        # safe center of contour
        position = get_contour_center(c)
        contour_id = i
        contours_cnt += 1

    # Make sure that only one closed contour is found by checking counter
    if contours_cnt != 1:
        position = np.full(2, np.nan)
        contour_id = -1
        found = False
    else:
        found = True
    return position, found, contour_id


"""
def get_center_position_by_template(frame):
    Computes the center of the contour that is searched for.

    Parameters
    ----------
    # Convert frame to hsv-colour-space for colour filtering
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Preparing mask to overlay
    mask_red = cv2.inRange(frame_hsv, detection_params.red_min, detection_params.red_max)
    mask_green = cv2.inRange(frame_hsv, detection_params.green_min, detection_params.green_max)

    # Perform template matching
    red_result = cv2.matchTemplate(mask_red, circle_image, cv2.TM_CCOEFF_NORMED)
    green_result = cv2.matchTemplate(mask_green, circle_image, cv2.TM_CCOEFF_NORMED)

    # Get the position of the best match
    min_val, max_val_red, min_loc, max_loc = cv2.minMaxLoc(red_result)
    h, w = circle_image.shape
    red_center = np.array((int(max_loc[0] + w/2), int(max_loc[1] + h/2)))

    min_val, max_val_green, min_loc, max_loc = cv2.minMaxLoc(green_result)
    h, w = circle_image.shape
    green_center = np.array((int(max_loc[0] + w/2), int(max_loc[1] + h/2)))
    if max_val_red > 0.5 and max_val_green > 0.5:
        return red_center, True, green_center, True
    else:
        return red_center, False, green_center, False
"""


class AngleDetector:
    """
    Class for detecting the angles and angular velocities of a double pendulum system.

    Parameters
    ----------
    definition : int, optional
        An integer value that determines the reference vector for the second pendulum arm.
        0 for absolute, 1 for relative measurement.

    Attributes
    ----------
    definition : int
        An integer value that determines the reference vector for the second pendulum arm.
    contours_red : numpy.ndarray
        The contours of the red objects in the most recent frame.
    contours_green : numpy.ndarray
        The contours of the green objects in the most recent frame.
    contour_red_id : int
        The index of the red contour used for center calculation in the contours_red array.
    contour_green_id : int
        The index of the green contour used for center calculation in the contours_green array.
    pos_A : numpy.ndarray
        The position of the fixed pivot point of the double pendulum.
    pos_B : numpy.ndarray
        The position of the second pivot point.
    pos_C : numpy.ndarray
        The position of the end of the second pendulum arm.
    vec_ref_1 : numpy.ndarray
        The reference vector used for the first pendulum arm.
    vec_ref_2 : numpy.ndarray
        The reference vector used for the second pendulum arm.
    angle_1 : float
        The angle of the first pendulum arm. [rad]
    angle_2 : float
        The angle of the second pendulum arm. [rad]
    angular_vel_1 : float
        The angular velocity of the first pendulum arm. [rad/s]
    angular_vel_2 : float
        The angular velocity of the second pendulum arm. [rad/s]
    angle_buffer_1 : AngleBuffer
        The buffer used for storing the previous angle values of the first pendulum arm.
    angle_buffer_2 : AngleBuffer
        The buffer used for storing the previous angle values of the second pendulum arm.
    start_time : float
        The timestamp of when the AngleDetector object was created.
    timestamp : float
        The timestamp of the most recent angle calculation.

    Methods
    -------
    get_contours()
        Filters the captured frame for red and green colour and extracts the contours separately.
    get_angle()
        Calculates the angles of the double pendulum using the extracted contours.
    get_angular_vel()
        Calculates the angular velocity with the values in the two angle buffers.
    visualize(vis_text=True, vis_contours=True, vis_vectors=True)
        Visualizes the live results of angle detection.
    """

    def __init__(self, definition=0):
        importlib.reload(detection_params)  # reload parameter file in case of changes due to calibration

        self.definition = definition
        self.contours_red = None
        self.contours_green = None
        self.contour_red_id = None
        self.contour_green_id = None
        self.pos_A = detection_params.pos_A
        self.pos_B = np.full(2, np.nan)
        self.pos_C = np.full(2, np.nan)
        self.vec_ref_1 = detection_params.vec_ref_1
        self.vec_ref_2 = np.zeros(2, dtype=int)
        self.angle_1 = float("NaN")
        self.angle_2 = float("NaN")
        self.angular_vel_1 = float("NaN")
        self.angular_vel_2 = float("NaN")
        self.angle_buffer_1 = AngleBuffer()
        self.angle_buffer_2 = AngleBuffer()
        self.timestamp = float("NaN")
        self.visualization = None
        self.frame_rate = None

    def set_timestamp(self, timestamp, frame_rate):
        # Set timestamp to value given by camera controller
        self.timestamp = timestamp
        self.frame_rate = frame_rate

    def warp_image(self, frame, already_warped=False):
        if already_warped:
            self.visualization = frame
            return frame

        # Warp frame with warp matrix to frame with defined side length
        frame_warped = cv2.warpPerspective(frame, detection_params.warp_matrix,
                                           (detection_params.warped_frame_side, detection_params.warped_frame_side))
        self.visualization = frame_warped
        return frame_warped

    def get_contours(self, frame, visualize=False):
        """
        Filters the captured frame for red and green colour and extracts the contours separately

        Returns
        -------
        contours_red : numpy.ndarray
            The red contours found in the frame.
        contours_green: numpy.ndarray
            The green contours found in the frame.
        """
        # Convert frame to hsv-colour-space for colour filtering
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Media filtering for removing noise
        ksize = 5  # Kernel size (odd number)
        frame_hsv = cv2.medianBlur(frame_hsv, ksize)

        # Preparing mask to overlay
        mask_red = cv2.inRange(frame_hsv, detection_params.red_min, detection_params.red_max)
        mask_green = cv2.inRange(frame_hsv, detection_params.green_min, detection_params.green_max)

        # cv2.imshow("mask_red", mask_red)
        # cv2.waitKey(0)
        #
        # cv2.imshow("mask_green", mask_green)
        # cv2.waitKey(0)

        # Find contours for red and green shapes in frame
        self.contours_red, _ = cv2.findContours(mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.contours_green, _ = cv2.findContours(mask_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    def get_angle(self, frame):
        start_time = time.time()
        """
        Calculates the angles of the double pendulum using the extracted contours.

        Returns
        -------
        list
            A list of two float elements representing the first and second angle respectively.
        """
        # Try to detect red and green circle in frame and calculate center
        self.get_contours(frame)
        self.pos_B, found_B, self.contour_red_id = get_center_position(self.contours_red)
        self.pos_C, found_C, self.contour_green_id = get_center_position(self.contours_green)


        # if len(self.contours_red) > 10 or len(self.contours_green) > 10:
        #     print("GREEN CONTOURS: {}, RED CONTOURS: {}".format(len(self.contours_green), len(self.contours_red)))
        #     cv2.imshow("IMAGE1", frame)
        #     cv2.waitKey(0)
        # if not found_B or not found_C:
        #     print("DIDN'T FIND CIRCLE")
        #     cv2.imshow("IMAGE2", frame)
        #     cv2.waitKey(0)
        # self.pos_B, found_B, self.pos_C, found_C = get_center_position_by_template(frame)

        # Calculate angle of first arm
        if found_B:
            vec_ab = self.pos_B - self.pos_A
            # Check in which rotational direction angle should be measured
            if check_side(self.pos_A, self.pos_A + self.vec_ref_1, self.pos_B) == (-1 | 0):
                self.angle_1 = calc_angle(self.vec_ref_1, vec_ab)
            else:
                self.angle_1 = -calc_angle(self.vec_ref_1, vec_ab)
        else:
            # Set value to NaN when no matching red contour could be found
            self.angle_1 = float('NaN')

        # Calculate angle of second arm
        if found_B & found_C:
            vec_ab = self.pos_B - self.pos_A
            vec_bc = self.pos_C - self.pos_B
            # Check for chosen angle definition
            if self.definition == 0:
                self.vec_ref_2 = vec_ab
                # Check in which rotational direction angle should be measured
                if check_side(self.pos_B, self.pos_B + self.vec_ref_2, self.pos_C) == (-1 | 0):
                    self.angle_2 = calc_angle(self.vec_ref_2, vec_bc)
                else:
                    self.angle_2 = -calc_angle(self.vec_ref_2, vec_bc)
            elif self.definition == 1:
                self.vec_ref_2 = detection_params.vec_ref_1
                # Check in which rotational direction angle should be measured
                if check_side(self.pos_B, self.pos_B + self.vec_ref_2, self.pos_C) == (-1 | 0):
                    self.angle_2 = calc_angle(self.vec_ref_2, vec_bc)
                else:
                    self.angle_2 = -calc_angle(self.vec_ref_2, vec_bc)
        else:
            # Set Value to Nan when one of the contours could not be found (calculating second angle would be
            # impossible without position of point B
            self.angle_2 = float('NaN')

        # Fill angle buffer for calculation of angular velocities
        self.angle_buffer_1.shift_buffer(self.angle_1, self.timestamp)
        self.angle_buffer_2.shift_buffer(self.angle_2, self.timestamp)

        # print("Angle detection needed: {:.1f} ms".format((time.time() - start_time)*1000))
        return [self.angle_1, self.angle_2]

    def get_angular_vel(self):
        """
        Calculates the angular velocity with the values in the two angle buffers.

        Returns
        -------
        list
            A list of two float elements representing the angular velocity of the
            first and second pendulum arm respectively.
        """
        # Calculate velocities, make sure that angle-values are available
        if not np.all(self.angle_buffer_1.timestamps == 0):
            self.angular_vel_1 = self.angle_buffer_1.calc_velocity()
            self.angular_vel_2 = self.angle_buffer_2.calc_velocity()
            return [self.angular_vel_1, self.angular_vel_2]
        else:
            raise RuntimeError("No values for velocity calculation available. "
                               "Use 'get_angular_vel()' only in combination with 'get_angle()'-function.")

    def visualize(self, vis_text=True, vis_contours=True, vis_vectors=True, vis_timestamp=False):
        """
        Visualizes the live results of angle detection.

        Parameters
        ----------
        vis_text : bool
            Boolean value to decide if text should be visualized.
        vis_contours : bool
            Boolean value to decide if contours should be visualized.
        vis_vectors : bool
            Boolean value to decide if vectors should be visualized.
        vis_timestamp : bool
            Boolean value to decide if timestamp should be visualized.
        """
        # Ensure that frame for visualization are available
        if self.visualization is not None:
            # visualization of the contours used for angle computation
            if vis_contours:
                # draw contours only when function get_center_position returned a valid value
                if self.contour_red_id != -1:
                    cv2.drawContours(self.visualization, self.contours_red, self.contour_red_id, (0, 0, 255), 1)
                if self.contour_green_id != -1:
                    cv2.drawContours(self.visualization, self.contours_green, self.contour_green_id, (0, 0, 255), 1)

            # if angle_1 could be detected set text labels and draw vectors for angle visualization
            if not math.isnan(self.angle_1):
                label11 = str(round((np.rad2deg(self.angle_1)), 1)) + " deg "
                # exclude that angle was detected but angular velocity could not be calculated due to missing values
                if not math.isnan(self.angular_vel_1):
                    label12 = str(round(np.rad2deg(self.angular_vel_1), 1)) + " deg/s"
                else:
                    label12 = "NaN"
                if vis_vectors:
                    # draw vector between joints
                    cv2.line(self.visualization, self.pos_A, self.pos_B, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    # draw reference axis
                    cv2.line(self.visualization, self.pos_A, self.pos_A + get_axis_visu(self.vec_ref_1), (0, 255, 0),
                             lineType=cv2.LINE_AA, thickness=1)
                    # draw angle visu
                    if check_side(self.pos_A, self.pos_A + (0, 100), self.pos_A + self.vec_ref_1) == (-1 | 0):
                        angular_offset = np.rad2deg(calc_angle(np.array((0, 1)), detection_params.vec_ref_1))
                    else:
                        angular_offset = -np.rad2deg(calc_angle(np.array((0, 1)), detection_params.vec_ref_1))
                    start_angle = 90 - angular_offset
                    cv2.ellipse(self.visualization, self.pos_A, (45, 45), 0, start_angle,
                                start_angle - np.rad2deg(self.angle_1),
                                (0, 255, 0))  # angle ellipse
            else:
                label11 = "NaN"
                label12 = "NaN"

            # if angle_2 could be detected set text labels and draw vectors for angle visualization
            if not math.isnan(self.angle_2):
                label21 = str(round(np.rad2deg(self.angle_2), 1)) + " deg "
                # exclude that angle was detected but angular velocity could not be calculated due to missing values
                if not math.isnan(self.angular_vel_2):
                    label22 = str(round(np.rad2deg(self.angular_vel_2), 1)) + " deg/s"
                else:
                    label22 = "NaN"
                if vis_vectors:
                    # draw vector between joints
                    cv2.line(self.visualization, self.pos_B, self.pos_C, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    # distinguish by chosen definition
                    if self.definition == 0:  # relative definition
                        # draw reference axis
                        cv2.line(self.visualization, self.pos_B, self.pos_B + get_axis_visu(self.vec_ref_2),
                                 (0, 255, 0),
                                 lineType=cv2.LINE_AA, thickness=1)
                        # draw angle visu
                        cv2.ellipse(self.visualization, self.pos_B, (45, 45), 0, 90 - np.rad2deg(self.angle_1),
                                    90 - np.rad2deg(self.angle_1) - np.rad2deg(self.angle_2), (0, 255, 0))
                    elif self.definition == 1:  # absolute definition
                        # draw reference axis
                        cv2.line(self.visualization, self.pos_B, self.pos_B + get_axis_visu(self.vec_ref_2),
                                 (0, 255, 0), lineType=cv2.LINE_AA,
                                 thickness=1)
                        # draw angle visu
                        if check_side(self.pos_B, self.pos_B + (0, 100), self.pos_B + self.vec_ref_2) == (-1 | 0):
                            angular_offset = np.rad2deg(calc_angle(np.array((0, 1)), self.vec_ref_2))
                        else:
                            angular_offset = -np.rad2deg(calc_angle(np.array((0, 1)), self.vec_ref_2))
                        start_angle = 90 - angular_offset
                        cv2.ellipse(self.visualization, self.pos_B, (45, 45), 0, start_angle,
                                    start_angle - np.rad2deg(self.angle_2), (0, 255, 0))
            else:
                label21 = "NaN"
                label22 = "NaN"

            # print measured values in openCV window using defined labels
            if vis_text:
                # white background
                cv2.rectangle(self.visualization, (0, 0), (235, 80), (255, 255, 255), -1)

                # splitting in multiple function calls to achieve fixed positions (tabular)
                cv2.putText(self.visualization, "angle_1", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(self.visualization, "=", (79, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.visualization, label11, (102, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)

                cv2.putText(self.visualization, "velocity_1", (2, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(self.visualization, "=", (79, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.visualization, label12, (102, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)

                cv2.putText(self.visualization, "angle_2", (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(self.visualization, "=", (79, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.visualization, label21, (102, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)

                cv2.putText(self.visualization, "velocity_2", (2, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(self.visualization, "=", (79, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.visualization, label22, (102, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)

            # draw found center position of contours
            if vis_vectors:
                cv2.circle(self.visualization, self.pos_A, 2, (255, 0, 0), thickness=3)
                if not math.isnan(self.pos_B[0]):
                    cv2.circle(self.visualization, self.pos_B, 2, (255, 0, 0), thickness=3)
                if not math.isnan(self.pos_C[0]):
                    cv2.circle(self.visualization, self.pos_C, 2, (255, 0, 0), thickness=3)

            if vis_timestamp:
                cv2.rectangle(self.visualization, (detection_params.warped_frame_side - 130, 0),
                              (detection_params.warped_frame_side, 40), (255, 255, 255), -1)
                cv2.putText(self.visualization, "t = {:.3f}s".format(self.timestamp),
                            (detection_params.warped_frame_side - 128, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.visualization, "fps = {:.0f}".format(self.frame_rate),
                            (detection_params.warped_frame_side - 128, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            raise RuntimeError("Nothing to visualize. "
                               "Use 'visualize()'-function only in combination with 'get_angle()'-function.")
        return self.visualization


class AngleBuffer:
    """
        Class for collecting angles with corresponding timestamps to calculate angular velocities of a
        double pendulum system.

        Attributes
        ----------
        timestamps: numpy.ndarray
            The timestamps used to calculate the angular velocity.
        angles: numpy.ndarray
            The angles used to calculate the angular velocity.
        current_vel: float
            The current angular velocity calculated with the buffer class.

        Methods
        -------
        shift_buffer(angle, timestamp)
            Shifts the angle buffer by replacing the oldest angle and timestamp with the newest ones.
        calc_velocity()
            Calculates the angular velocity based two angle values and corresponding timestamps.
        """

    def __init__(self):
        self.timestamps = np.zeros(2)
        self.angles = np.zeros(2)
        self.current_vel = 0

    def shift_buffer(self, angle, timestamp):
        """
        Shifts the angle buffer by replacing the oldest angle and timestamp with the newest ones.

        Parameters
        ----------
        angle : float
            The newest angle value to add to the buffer.
        timestamp : float
            The corresponding timestamp of the newest angle value.
        """
        self.timestamps[0] = self.timestamps[1]
        self.angles[0] = self.angles[1]
        self.timestamps[1] = timestamp
        self.angles[1] = angle

    def calc_velocity(self):
        """
        Calculates the angular velocity based on the difference between two angle values
        and corresponding timestamps in the angle buffer.

        Returns
        -------
        float
            The calculated velocity as a float value in radians per seconds.
        """

        # check if OT was passed between the two measured values
        passed_ot_from_right = self.angles[0] > 0 > self.angles[1] and self.current_vel > 0
        passed_ot_from_left = self.angles[0] < 0 < self.angles[1] and self.current_vel < 0

        if passed_ot_from_right:
            self.current_vel = (self.angles[1] - self.angles[0] + 2 * math.pi) / \
                               (self.timestamps[1] - self.timestamps[0])
            return self.current_vel

        if passed_ot_from_left:
            self.current_vel = (self.angles[1] - self.angles[0] - 2 * math.pi) / \
                               (self.timestamps[1] - self.timestamps[0])
            return self.current_vel
        self.current_vel = (self.angles[1] - self.angles[0]) / (self.timestamps[1] - self.timestamps[0])
        return self.current_vel
