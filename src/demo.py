from camera_controller import IDSCameraController
from angle_detection import AngleDetector
from recording import VisuRecorder
from recording import DataRecorder
from recording import FrameExtractor
import cv2 as cv
import time

# --------------------------------------------------IMPORTANT NOTE-----------------------------------------------------
# To ensure optimal performance, it is important to calibrate the system properly before using the following functions.
# To perform a complete calibration of the system, you can use the "calibration.py" script.
# To perform individual calibration steps, use the scripts "get_warp_matrix.py", "get_hsv_masks.py" and
# "get_reference_axis.py". If no calibration data is available, an error may occur. It is also recommended, to adjust
# the camera-parameter file to the current illumination using "uEye Cockpit".
# In case of functional issues, you can try using the debug scripts.
# ---------------------------------------------------------------------------------------------------------------------

# Initialize camera, param_file = set camera parameters with .ini file
camera = IDSCameraController(param_file=r"../CameraParameters/cp_AngleDetection.ini")

# Initialize AngleDetector object,
# definition = set definition of angle (0 = second angle relative to the first, 1=second angle absolute)
measurement = AngleDetector(camera, definition=0)

# Initialize evaluation objects
# VisuRecorder to record video stream, rec_filename = set video filename
visu_rec = VisuRecorder(rec_filename='Demo')

# DataRecorder to record measurement data, log_filename = set log filename
data_rec = DataRecorder(log_filename='Demo')

# FrameExtractor to extract single frames from video stream,
# frame_filename = set filename for frames, folder = set folder name,
# rate = rate to capture frames, count =  number of frames to capture
frame_extr = FrameExtractor(frame_filename='DemoFrame', folder='DemoFolder', rate=20, count=100)

while True:
    # If needed measure execution time
    # start_time = time.time()

    # Detect angles of pendulum
    [angle1, angle2] = measurement.get_angle()

    # Compute angular velocities
    [angular_vel1, angular_vel2] = measurement.get_angular_vel()

    # Visualize measurement live, vis_text = enable text field with current data,
    # vis_contours = enable visualization of used contours, vis_vectors = enable visualization of angle definition,
    # vis_timestamp = enable text field with timestamp
    # Note: Use only in combination with get_angle() and if velocities needed with get_angular_vel()
    measurement.visualize(vis_text=True, vis_contours=True, vis_vectors=True, vis_timestamp=True)

    # Record visualization of measurement, passing parameter: AngleDetector-Object
    # Note: Use only in combination with visualize() function
    visu_rec.record_visu(measurement)

    # Record data of active measurement, passing parameter: AngleDetector-Object
    # Note: Use only in combination with get_angle() and if needed get_angular_vel() function
    data_rec.write_datarow(measurement)

    # Extract frames from visualization, passing parameter: AngleDetector-Object
    # Note: Use only in combination with visualize() function
    frame_extr.extract_frames(measurement)

    # Print execution time per cycle if needed
    # print(time.time() - start_time)

    # Possibility to quit measurement by hitting 'q'. Only usable if visualize() function is used.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release all resources
# Close visualization
cv.destroyAllWindows()

# End camera connection
camera.close_camera_connection()

# Save recording of visualization
visu_rec.stop_recording_visu()

# Save data-files as .csv or .pkl
data_rec.save_csv()
data_rec.save_pickle()
