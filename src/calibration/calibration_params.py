import cv2 as cv

# Parameter file used for storing/loading the parameters for the calibration.

# Side length of the frame after warping. [pixel]
warped_frame_side = 1000

# Define used aruco markers
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
aruco_params = cv.aruco.DetectorParameters()

# Tolerances for colour masks [int]
hue_tolerance = 20
sat_tolerance = 20
val_tolerance = 15
