import importlib
import cv2 as cv
import numpy as np
import math
import detection_params
import time


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
    moments = cv.moments(pts)
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
        area = cv.contourArea(c)

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


class AngleDetector:
    """
    Class for detecting the angles and angular velocities of a double pendulum system.

    Parameters
    ----------
    camera : Camera
        The camera object used for capturing the video stream.
    definition : int, optional
        An integer value that determines the reference vector for the second pendulum arm.
        0 for absolute, 1 for relative measurement.

    Attributes
    ----------
    camera: Camera
        The camera object used for capturing the video stream.
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
    visu : numpy.ndarray
        The most recent warped frame. Also used for visualization.
    visu_used : Bool
        A boolean indicator to show if visu function is used.

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
    def __init__(self, camera, definition=0):
        importlib.reload(detection_params)  # reload parameter file in case of changes due to calibration

        self.camera = camera
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
        self.start_time = time.time()
        self.timestamp = float("NaN")
        self.visu = None
        self.visu_used = False

    def get_contours(self):
        """
        Filters the captured frame for red and green colour and extracts the contours separately

        Returns
        -------
        contours_red : numpy.ndarray
            The red contours found in the frame.
        contours_green: numpy.ndarray
            The green contours found in the frame.
        """
        frame = self.camera.capture_image()

        # Warp frame with warp matrix to frame with defined side length
        frame_warped = cv.warpPerspective(frame, detection_params.warp_matrix,
                                          (detection_params.warped_frame_side, detection_params.warped_frame_side))
        self.visu = frame_warped

        # Convert frame to hsv-colour-space for colour filtering
        frame_hsv = cv.cvtColor(frame_warped, cv.COLOR_BGR2HSV)

        # Preparing mask to overlay
        mask_red = cv.inRange(frame_hsv, detection_params.red_min, detection_params.red_max)
        mask_green = cv.inRange(frame_hsv, detection_params.green_min, detection_params.green_max)

        # Find contours for red and green shapes in frame
        contours_red, _ = cv.findContours(mask_red, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours_green, _ = cv.findContours(mask_green, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        return contours_red, contours_green

    def get_angle(self):
        """
        Calculates the angles of the double pendulum using the extracted contours.

        Returns
        -------
        list
            A list of two float elements representing the first and second angle respectively.
        """
        # Try to detect red and green circle in frame and calculate center
        self.contours_red, self.contours_green = self.get_contours()
        self.pos_B, found_B, self.contour_red_id = get_center_position(self.contours_red)
        self.pos_C, found_C, self.contour_green_id = get_center_position(self.contours_green)

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
        self.timestamp = float(round(time.time()-self.start_time, 4))
        self.angle_buffer_1.shift_buffer(self.angle_1, self.timestamp)
        self.angle_buffer_2.shift_buffer(self.angle_2, self.timestamp)

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
        if self.visu is not None:
            # set marker for other error-handling functionalities
            self.visu_used = True
            # visualization of the contours used for angle computation
            if vis_contours:
                # draw contours only when function get_center_position returned a valid value
                if self.contour_red_id != -1:
                    cv.drawContours(self.visu, self.contours_red, self.contour_red_id, (0, 0, 255), 1)
                if self.contour_green_id != -1:
                    cv.drawContours(self.visu, self.contours_green, self.contour_green_id, (0, 0, 255), 1)

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
                    cv.line(self.visu, self.pos_A, self.pos_B, (255, 0, 0), thickness=1)
                    # draw reference axis
                    cv.line(self.visu, self.pos_A, self.pos_A + get_axis_visu(self.vec_ref_1), (0, 255, 0),
                            lineType=cv.LINE_8, thickness=1)
                    # draw angle visu
                    if check_side(self.pos_A, self.pos_A + (0, 100), self.pos_A + self.vec_ref_1) == (-1 | 0):
                        angular_offset = np.rad2deg(calc_angle(np.array((0, 1)), detection_params.vec_ref_1))
                    else:
                        angular_offset = -np.rad2deg(calc_angle(np.array((0, 1)), detection_params.vec_ref_1))
                    start_angle = 90 - angular_offset
                    cv.ellipse(self.visu, self.pos_A, (45, 45), 0, start_angle, start_angle - np.rad2deg(self.angle_1),
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
                    cv.line(self.visu, self.pos_B, self.pos_C, (255, 0, 0), thickness=1)
                    # distinguish by chosen definition
                    if self.definition == 0:  # relative definition
                        # draw reference axis
                        cv.line(self.visu, self.pos_B, self.pos_B + get_axis_visu(self.vec_ref_2), (0, 255, 0),
                                lineType=cv.LINE_8, thickness=1)
                        # draw angle visu
                        cv.ellipse(self.visu, self.pos_B, (45, 45), 0, 90 - np.rad2deg(self.angle_1),
                                   90 - np.rad2deg(self.angle_1) - np.rad2deg(self.angle_2), (0, 255, 0))
                    elif self.definition == 1:  # absolute definition
                        # draw reference axis
                        cv.line(self.visu, self.pos_B, self.pos_B + get_axis_visu(self.vec_ref_2),
                                (0, 255, 0), lineType=cv.LINE_8,
                                thickness=1)
                        # draw angle visu
                        if check_side(self.pos_B, self.pos_B + (0, 100), self.pos_B + self.vec_ref_2) == (-1 | 0):
                            angular_offset = np.rad2deg(calc_angle(np.array((0, 1)), self.vec_ref_2))
                        else:
                            angular_offset = -np.rad2deg(calc_angle(np.array((0, 1)), self.vec_ref_2))
                        start_angle = 90 - angular_offset
                        cv.ellipse(self.visu, self.pos_B, (45, 45), 0, start_angle,
                                   start_angle - np.rad2deg(self.angle_2), (0, 255, 0))
            else:
                label21 = "NaN"
                label22 = "NaN"

            # print measured values in openCV window using defined labels
            if vis_text:
                # white background
                cv.rectangle(self.visu, (0, 0), (235, 80), (255, 255, 255), -1)

                # splitting in multiple function calls to achieve fixed positions (tabular)
                cv.putText(self.visu, "angle_1", (2, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, "=", (79, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, label11, (102, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

                cv.putText(self.visu, "velocity_1", (2, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, "=", (79, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, label12, (102, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

                cv.putText(self.visu, "angle_2", (2, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, "=", (79, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, label21, (102, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

                cv.putText(self.visu, "velocity_2", (2, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, "=", (79, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, label22, (102, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

            # draw found center position of contours
            if vis_vectors:
                cv.circle(self.visu, self.pos_A, 2, (255, 0, 0), thickness=3)
                if not math.isnan(self.pos_B[0]):
                    cv.circle(self.visu, self.pos_B, 2, (255, 0, 0), thickness=3)
                if not math.isnan(self.pos_C[0]):
                    cv.circle(self.visu, self.pos_C, 2, (255, 0, 0), thickness=3)

            if vis_timestamp:
                cv.rectangle(self.visu, (detection_params.warped_frame_side-130, 0),
                             (detection_params.warped_frame_side, 20), (255, 255, 255), -1)
                cv.putText(self.visu, f"t = {self.timestamp}s", (detection_params.warped_frame_side-128, 16),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

            # show frame in pop-up window
            cv.namedWindow('Angle Detection', cv.WINDOW_AUTOSIZE)
            cv.imshow('Angle Detection', self.visu)
        else:
            raise RuntimeError("Nothing to visualize. "
                               "Use 'visualize()'-function only in combination with 'get_angle()'-function.")


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
