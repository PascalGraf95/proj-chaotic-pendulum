from camera_controller import IDSCameraController
from angle_detection import AngleDetector
from recording import VideoRecorder
from recording import DataRecorder
from recording import FrameExtractor
import cv2
import time
import argparse
import numpy as np

# --------------------------------------------------IMPORTANT NOTE-----------------------------------------------------
# To ensure optimal performance, it is important to calibrate the system properly before using the following functions.
# To perform a complete calibration of the system, you can use the "calibration.py" script.
# To perform individual calibration steps, use the scripts "get_warp_matrix.py", "get_hsv_masks.py" and
# "get_reference_axis.py". If no calibration data is available, an error may occur. It is also recommended, to adjust
# the camera-parameter file to the current illumination using "uEye Cockpit".
# In case of functional issues, you can try using the debug scripts.
# ---------------------------------------------------------------------------------------------------------------------

def main(mode, video_path):
    if mode == "live_detection" or mode == "recording" or mode == "record_with_detection" or mode == "live_feed":
        # Initialize camera, param_file = set camera parameters with .ini file
        camera = IDSCameraController(param_file=r"../CameraParameters/cp_230809_AngleDetection.ini")
    else:
        # ToDo: Setup video controller
        camera = None
        pass

    # Initialize AngleDetector object,
    # definition = set definition of angle (0 = second angle relative to the first, 1=second angle absolute)
    measurement = AngleDetector(definition=0)

    # Initialize evaluation objects
    # VisuRecorder to record video stream, rec_filename = set video filename
    video_recorder = VideoRecorder(rec_filename='Demo')

    # DataRecorder to record measurement data, log_filename = set log filename
    data_rec = DataRecorder(log_filename='Demo')

    # FrameExtractor to extract single frames from video stream,
    # frame_filename = set filename for frames, folder = set folder name,
    # rate = rate to capture frames, count =  number of frames to capture
    frame_extr = FrameExtractor(frame_filename='DemoFrame', folder='DemoFolder')

    while True:
        # start_time = time.time()
        # Get the latest camera image
        frame = camera.capture_image()
        # takes about 1 ms

        # Warp image according to warp matrix
        warped_frame = measurement.warp_image(frame)
        # takes about 3 ms

        if mode == "live_detection" or mode == "record_with_detection":
            # Detect angles of pendulum
            [angle1, angle2] = measurement.get_angle(warped_frame)
            # takes about 5 ms

            # Compute angular velocities
            [angular_vel1, angular_vel2] = measurement.get_angular_vel()
            # takes about 1 ms


        # Visualize measurement live, vis_text = enable text field with current data,
        # vis_contours = enable visualization of used contours, vis_vectors = enable visualization of angle definition,
        # vis_timestamp = enable text field with timestamp
        # Note: Use only in combination with get_angle() and if velocities needed with get_angular_vel()
        if mode == "live_detection" or mode == "record_with_detection":
            visualization = measurement.visualize(vis_text=True, vis_contours=True,
                                                  vis_vectors=True, vis_timestamp=True)
        else:
            visualization = measurement.visualize(vis_text=False, vis_contours=False,
                                                  vis_vectors=False, vis_timestamp=True)
        # takes about 1ms


        # Record visualization of measurement, passing parameter: AngleDetector-Object
        # Note: Use only in combination with visualize() function
        # video_recorder.record_video(measurement)
        if mode == "record_with_detection":
            # Record data of active measurement, passing parameter: AngleDetector-Object
            # Note: Use only in combination with get_angle() and if needed get_angular_vel() function
            data_rec.write_datarow(measurement)
            # takes about 5-6 ms of time

        if mode == "recording" or mode == "record_with_detection":
            # Extract frames from visualization, passing parameter: AngleDetector-Object
            # Note: Use only in combination with visualize() function
            frame_extr.save_latest_frame(visualization)

        if mode == "live_detection" or mode == "live_feed":
            cv2.imshow("Live Image", visualization)
            # takes about 3 ms

        # Possibility to quit measurement by hitting 'q'. Only usable if visualize() function is used.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print("Total Time: {:.1f}".format((time.time() - start_time)*1000))


    # Release all resources
    # Close visualization
    cv2.destroyAllWindows()

    # End camera connection
    camera.close_camera_connection()

    # Save recording of visualization
    video_recorder.stop_recording_video()

    # Save data-files as .csv or .pkl
    data_rec.save_csv()
    data_rec.save_pickle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('-m', '--mode', type=str, default="live_detection",
                        help="", required=False)
    parser.add_argument('-vp', '--video_path', type=str, default=None, required=False)

    args = parser.parse_args()
    main(args.mode, args.video_path)
