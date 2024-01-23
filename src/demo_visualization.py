from data_acquisition.camera_controller import IDSCameraController, ImageStreamController
from data_acquisition.angle_detection import AngleDetector
from data_acquisition.recording import VideoRecorder
from data_acquisition.recording import DataRecorder
from data_acquisition.recording import FrameRecorder
import cv2
import argparse


# --------------------------------------------------IMPORTANT NOTE-----------------------------------------------------
# To ensure optimal performance, it is important to calibrate the system properly before using the following functions.
# To perform a complete calibration of the system, you can use the "calibration.py" script.
# To perform individual calibration steps, use the scripts "get_warp_matrix.py", "get_hsv_masks.py" and
# "get_reference_axis.py". If no calibration data is available, an error may occur. It is also recommended, to adjust
# the camera-parameter file to the current illumination using "uEye Cockpit".
# In case of functional issues, you can try using the debug scripts.
# ---------------------------------------------------------------------------------------------------------------------


def main(video_path, record=False, live_feed=True, detect=False):
    if not video_path:
        # Initialize camera, param_file = set camera parameters with .ini file
        camera = IDSCameraController(param_file=r"../CameraParameters/cp_230809_AngleDetection.ini")
    else:
        camera = ImageStreamController(video_path=video_path)

    # Initialize AngleDetector object,
    # definition = set definition of angle (0 = second angle relative to the first, 1=second angle absolute)
    measurement = AngleDetector()

    # Initialize evaluation objects
    # VisuRecorder to record video stream, rec_filename = set video filename
    # video_recorder = VideoRecorder(rec_filename='Demo')
    if record:
        # FrameExtractor to extract single frames from video stream,
        # frame_filename = set filename for frames, folder = set folder name,
        # rate = rate to capture frames, count =  number of frames to capture
        frame_extr = FrameRecorder(folder='DemoFolder')

        # DataRecorder to record measurement data, log_filename = set log filename
        data_rec = DataRecorder(folder='DemoFolder', timestamp=frame_extr.timestamp)
    else:
        frame_extr, data_rec = None, None

    # Set default values for angles and their velocities
    angles = [None, None]
    angular_velocities = [None, None]
    try:
        while True:
            # start_time = time.time()
            # Get the latest camera image
            frame = camera.capture_image()
            # takes about 1 ms

            # Transfer frame rate and fps from camera
            measurement.set_timestamp(camera.timestamp, camera.frame_rate)

            # Warp image according to warp matrix
            warped_frame = measurement.warp_image(frame, already_warped=video_path is not None)
            # takes about 3 ms

            if detect:
                # Detect angles of pendulum
                angles = measurement.get_angle(warped_frame)
                # takes about 5 ms

                # Compute angular velocities
                angular_velocities = measurement.get_angular_vel()
                # takes about 1 ms

            # Visualize measurement live, vis_text = enable text field with current data,
            # vis_contours = enable visualization of used contours, vis_vectors = enable visualization of angle definition,
            # vis_timestamp = enable text field with timestamp
            # Note: Use only in combination with get_angle() and if velocities needed with get_angular_vel()
            if detect:
                visualization = measurement.visualize(vis_text=True, vis_contours=True,
                                                      vis_vectors=True, vis_timestamp=True)
            else:
                visualization = measurement.visualize(vis_text=False, vis_contours=False,
                                                      vis_vectors=False, vis_timestamp=True)
            # takes about 1ms

            # Record visualization of measurement, passing parameter: AngleDetector-Object
            # Note: Use only in combination with visualize() function
            # video_recorder.record_video(measurement)
            if record:
                # Extract frames from visualization, passing parameter: AngleDetector-Object
                # Note: Use only in combination with visualize() function
                frame_extr.save_latest_frame(visualization)
                # takes about 5-6 ms of time
                # Record data of active measurement, passing parameter: AngleDetector-Object
                # Note: Use only in combination with get_angle() and if needed get_angular_vel() function
                data_rec.write_datarow(angles, angular_velocities, measurement.timestamp)

            if live_feed:
                cv2.imshow("Live Image", visualization)
                # takes about 3 ms

                # Possibility to quit measurement by hitting 'q'. Only usable if visualize() function is used.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if camera.video_has_ended:
                break
            # print("Total Time: {:.1f}".format((time.time() - start_time)*1000))
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")

    # Release all resources
    # Close visualization
    cv2.destroyAllWindows()

    # End camera connection
    camera.close_camera_connection()

    # Save recording of visualization
    # video_recorder.stop_recording_video()

    if record:
        # Save data-files as .csv or .pkl
        data_rec.save_csv()
        data_rec.save_pickle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('-rec', '--record', action=argparse.BooleanOptionalAction, default=False,
                        help="", required=False)
    parser.add_argument('-det', '--detect', action=argparse.BooleanOptionalAction, default=False,
                        help="", required=False)
    parser.add_argument('-liv', '--live_feed', action=argparse.BooleanOptionalAction, default=False,
                        help="", required=False)
    parser.add_argument('-vp', '--video_path', type=str, default=None, required=False)

    args = parser.parse_args()
    main(video_path=args.video_path, record=args.record, detect=args.detect, live_feed=args.live_feed)
