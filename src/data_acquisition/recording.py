import os.path
import datetime
from src.calibration import detection_params
import cv2
import pandas as pd
from PIL import Image


class VideoRecorder:
    """
    Class for recording visualizations produced by an angle detector object.

    Parameters
    ----------
    rec_filename : str, optional
        The filename for the recorded video. Default is "rec".

    Attributes
    ----------
    recorder : cv2.VideoWriter
        The video writer object used for recording the visualization.
    rec_filename : str
        The filename for the recorded video.

    Methods
    -------
    record_visu(angle_detector)
        Records the visualization produced by the angle detector object.
    stop_recording_visu()
        Stops the recording of the video and releases the video writer object.
    """
    def __init__(self, rec_filename="rec"):
        # create path for saving records
        if not os.path.exists("../VideoRecords"):
            os.makedirs("../VideoRecords")

        # create timestamp for file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # initialize recorder
        self.recorder = cv2.VideoWriter(f"../VideoRecords/{timestamp}_{rec_filename}.avi",
                                        cv2.VideoWriter_fourcc(*'MJPG'), detection_params.recorder_frame_rate,
                                        (detection_params.warped_frame_side, detection_params.warped_frame_side))

    def record_video(self, angle_detector):
        # write frame to file with recorder method
        # make sure that visualization is created
        if angle_detector.visu is not None and angle_detector.visu_used is True:
            self.recorder.write(angle_detector.visu)
        elif angle_detector.visu_used is False:
            raise RuntimeError("No visualization found to be recorded. "
                               "Use 'record_visu()'-function only in combination with 'visualize()-function.")
        else:
            raise RuntimeError("No visualization found to be recorded. "
                               "Use 'record_visu()'-function only in combination with 'get_angle()'-function")

    def stop_recording_video(self):
        # release resources
        self.recorder.release()


class DataRecorder:
    """
    Class for saving data produced by an angle detector object.

    Parameters
    ----------
    log_filename : str, optional
        The name of the log file (default: "log").

    Attributes
    ----------
    timestamp : str
        The timestamp of when the DataRecorder object was created.
    filename : str
        The name of the log file.
    df : pandas.DataFrame
        A DataFrame containing the recorded data.

    Methods
    -------
    write_datarow(angle_detector)
        Writes a row of data to the DataFrame.
    save_pickle()
        Saves the DataFrame to a pickle file.
    save_csv()
        Saves the DataFrame to a CSV file.
    """
    def __init__(self, filename="log", folder="folder", timestamp=None):
        if timestamp:
            self.timestamp = timestamp
        else:
            # create timestamp for file names
            self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # load filename from parameter
        self.filename = filename
        self.folder = folder

        # create path for saving files
        # if not os.path.exists(f"../DataRecords/{self.timestamp}_{self.folder}"):
            # os.makedirs(f"../DataRecords/{self.timestamp}_{self.folder}")

        # create dataframe
        self.df = pd.DataFrame({"Time": [], "Angle1": [], "Angle2": [], "AngularVel1": [], "AngularVel2": []})

    def write_datarow(self, angles, angular_velocity, timestamp):
        new_row = pd.Series(
            {"Time": timestamp, "Angle1": angles[0], "Angle2": angles[1],
             "AngularVel1": angular_velocity[0], "AngularVel2": angular_velocity[1]})
        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)

    def save_pickle(self):
        if len(self.df.index) != 0:
            self.df.to_pickle(f"../DataRecords/{self.timestamp}_{self.folder}/{self.timestamp}_{self.filename}.pkl")
        else:
            print("WARNING: No values found to save to .pkl-file. Use 'write_datarow'-function to collect data.")

    def save_csv(self, csv_path=None):
        if csv_path is None:
            csv_path = f"../DataRecords/{self.timestamp}_{self.folder}/{self.timestamp}_{self.filename}.csv"

        if len(self.df.Time.value_counts()) > 0:
            self.df.to_csv(csv_path, sep=';', index=False, decimal='.')
        else:
            print("WARNING: No values found to save to .csv-file. Use 'write_datarow'-function to collect data.")


class FrameRecorder:
    """
    A class for extracting frames from an angle detector's visualization.

    Parameters
    ----------
    frame_filename : str, optional
        The prefix of the extracted frame file names (default: "frame").
    folder : str, optional
        The name of the folder where the extracted frames will be saved (default: "folder").

    Attributes
    ----------
    filename : str
        The prefix of the extracted frame file names.
    folder : str
        The name of the folder where the extracted frames will be saved.
    frame_count : int
        A counter for the number of frames extracted.
    timestamp : str
        The timestamp of when the FrameExtractor object was created.

    Methods
    -------
    save_latest_frame(frame)
        Extracts frames from the angle detector's visualization.
    """
    def __init__(self, frame_filename='frame', folder="folder"):
        # load parameters
        self.filename = frame_filename
        self.folder = folder

        self.frame_count = 0

        # create timestamp
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create file path for saving frames
        # if not os.path.exists(f"../DataRecords/{self.timestamp}_{self.folder}"):
            # os.makedirs(f"../DataRecords/{self.timestamp}_{self.folder}")

    def save_latest_frame(self, frame):
        # make sure that angle detection is active and frames are captured
        if frame is not None:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image.save("../DataRecords/{}_"
                           "{}/{}_{:06d}.jpg".format(self.timestamp, self.folder, self.filename,
                                                     self.frame_count))
            self.frame_count += 1

    def save_latest_frame_specific_path(self, frame):
        # make sure that angle detection is active and frames are captured
        if frame is not None:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image.save("{}/{}_{:06d}.jpg".format(self.folder, self.filename, self.frame_count))
            self.frame_count += 1

