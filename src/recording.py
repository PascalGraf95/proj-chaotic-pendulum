import os.path
import datetime
import detection_params
import cv2 as cv
import pandas as pd


class VisuRecorder:
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
        if not os.path.exists("../VisuRecords"):
            os.makedirs("../VisuRecords")

        # create timestamp for file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # initialize recorder
        self.recorder = cv.VideoWriter(f"../VisuRecords/{timestamp}_{rec_filename}.avi",
                                       cv.VideoWriter_fourcc(*'MJPG'), detection_params.recorder_frame_rate,
                                       (detection_params.warped_frame_side, detection_params.warped_frame_side))

    def record_visu(self, angle_detector):
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

    def stop_recording_visu(self):
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
    def __init__(self, log_filename="log"):
        # create timestamp for file names
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create path for saving files
        if not os.path.exists("../DataRecords"):
            os.makedirs("../DataRecords")

        # load filename from parameter
        self.filename = log_filename

        # create dataframe
        self.df = pd.DataFrame({"Time": [], "Angle1": [], "Angle2": [], "AngularVel1": [], "AngularVel2": []})

    def write_datarow(self, angle_detector):
        # make sure that angle detection is active by checking visu attribute
        if angle_detector.visu is not None:
            new_row = pd.Series(
                {"Time": angle_detector.timestamp, "Angle1": angle_detector.angle_1, "Angle2": angle_detector.angle_2,
                 "AngularVel1": angle_detector.angular_vel_1, "AngularVel2": angle_detector.angular_vel_2})
            self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
        else:
            raise RuntimeError("No data found to be saved. "
                               "Use 'write_datarow()'-function only in combination with 'get_angle()'-function")

    def save_pickle(self):
        if len(self.df.index) != 0:
            self.df.to_pickle(f"../DataRecords/{self.timestamp}_{self.filename}.pkl")
        else:
            print("WARNING: No values found to save to .pkl-file. Use 'write_datarow'-function to collect data.")

    def save_csv(self):
        if len(self.df.Time.value_counts()) > 0:
            self.df.to_csv(f"../DataRecords/{self.timestamp}_{self.filename}.csv", sep=';', index=False, decimal='.')
        else:
            print("WARNING: No values found to save to .csv-file. Use 'write_datarow'-function to collect data.")


class FrameExtractor:
    """
    A class for extracting frames from an angle detector's visualization.

    Parameters
    ----------
    frame_filename : str, optional
        The prefix of the extracted frame file names (default: "frame").
    folder : str, optional
        The name of the folder where the extracted frames will be saved (default: "folder").
    rate : int, optional
        The frame extraction rate in frames per second (default: 10).
    count : int, optional
        The maximum number of frames to extract (default: 10).

    Attributes
    ----------
    filename : str
        The prefix of the extracted frame file names.
    folder : str
        The name of the folder where the extracted frames will be saved.
    rate : int
        The frame extraction rate in frames per second.
    frames : int
        The maximum number of frames to extract.
    rate_count : int
        A counter for the frame extraction rate.
    frame_count : int
        A counter for the number of frames extracted.
    timestamp : str
        The timestamp of when the FrameExtractor object was created.

    Methods
    -------
    extract_frames(angle_detector)
        Extracts frames from the angle detector's visualization.
    """
    def __init__(self, frame_filename='frame', folder="folder", rate=10, count=10):
        # load parameters
        self.filename = frame_filename
        self.folder = folder
        self.rate = rate
        self.frames = count

        self.rate_count = 0
        self.frame_count = 1

        # create timestamp
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create file path for saving frames
        if not os.path.exists(f"../ExtractedFrames/{self.timestamp}_{self.folder}"):
            os.makedirs(f"../ExtractedFrames/{self.timestamp}_{self.folder}")

    def extract_frames(self, angle_detector):
        # make sure that angle detection is active and frames are captured
        if angle_detector.visu is not None:
            # safe specified number of frames with specified rate
            self.rate_count += 1
            if (self.rate_count >= self.rate) & (self.frame_count <= self.frames):
                cv.imwrite(f"../ExtractedFrames/{self.timestamp}_"
                           f"{self.folder}/{self.filename}_{angle_detector.timestamp}.jpg", angle_detector.visu)
                self.frame_count += 1
                self.rate_count = 0
        elif angle_detector.visu_used is False:
            raise RuntimeError("No visualization found to extract frames from. "
                               "Use 'extract_frames()'-function only in combination with 'visualize()-function.")
        else:
            raise RuntimeError("No visualization found to extract frames from. "
                               "Use 'extract_frames()'-function only in combination with 'get_angle()'-function")
