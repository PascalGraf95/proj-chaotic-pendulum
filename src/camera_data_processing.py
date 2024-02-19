from data_acquisition.camera_controller import IDSCameraController, ImageStreamController
from data_acquisition.angle_detection import AngleDetector
from data_acquisition.recording import DataRecorder
from data_acquisition.recording import FrameRecorder


def run_data_acquisition(pendulum_data: str, data_length: int, video_path: str):
    """
    Run the data acquisition process to capture and record pendulum data.

    Parameters
    ----------
    pendulum_data : str
        Path to save the recorded pendulum data in CSV format.
    data_length : int
        Number of data points to be collected.
    video_path : str
        Path to the video file if using a pre-recorded video, otherwise None.

    Notes
    -----
    If `video_path` is None, the function will capture data from a connected camera using the IDSCameraController.
    If `video_path` is provided, the function will use ImageStreamController to process a pre-recorded video.
    """
    if not video_path:
        # Initialize camera, param_file = set camera parameters with .ini file
        camera = IDSCameraController(param_file=r"../CameraParameters/cp_230809_AngleDetection.ini")
    else:
        camera = ImageStreamController(video_path=video_path)
    print("Start!")
    measurement = AngleDetector()

    frame_extr = FrameRecorder()

    # DataRecorder to record measurement data, log_filename = set log filename
    data_rec = DataRecorder(timestamp=frame_extr.timestamp)

    for _ in range(data_length + 20):
        frame = camera.capture_image()

        measurement.set_timestamp(camera.timestamp, camera.frame_rate)

        warped_frame = measurement.warp_image(frame, already_warped=video_path is not None)

        angles = measurement.get_angle(warped_frame)

        angular_velocities = measurement.get_angular_vel()

        data_rec.write_datarow(angles, angular_velocities, measurement.timestamp)

        if camera.video_has_ended:
            break

    camera.close_camera_connection()

    data_rec.save_csv(csv_path=pendulum_data)
    print("Done!")


if __name__ == '__main__':
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_rnn_model")

    run_data_acquisition(video_path=None, data_length=600,
                         pendulum_data=rf"C:\Users\Marco\dev\git\proj-chaotic-pendulum\DataRecords\{timestamp}.csv")
