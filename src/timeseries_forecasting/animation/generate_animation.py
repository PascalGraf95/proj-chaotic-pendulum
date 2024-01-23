from tensorflow import keras
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import glob
from multiprocessing import Pool
from typing import Tuple
from src.timeseries_forecasting.animation.utils.generation_utils import reconstruct_radian
from src.timeseries_forecasting.animation.utils.validation_utils import validate_data_correctness, h5_file_exists
from src.timeseries_forecasting.animation.plot_images import generate_image

input_sequence = 50
output_sequence = 200
sequence = input_sequence + output_sequence


def plot_returns(prediction_output, ground_truth_output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Create a range of values for x-axis
    x_values = np.arange(sequence)

    # Arm 1
    axes[0].plot(x_values, prediction_output[:, 0], label="Prediction", linestyle='-', marker='o')
    axes[0].plot(x_values, ground_truth_output[:, 0], label="Ground Truth", linestyle='-', marker='o')

    # Arm 2
    axes[1].plot(x_values, prediction_output[:, 1], label="Prediction", linestyle='-', marker='o')
    axes[1].plot(x_values, ground_truth_output[:, 1], label="Ground Truth", linestyle='-', marker='o')

    # Add titles and legends
    for i in range(2):
        axes[i].set_title(f"Arm {i + 1}")
        axes[i].set_ylim([-3.2, 3.2])
        axes[i].legend()
        axes[i].grid(visible=True)

    plt.show()


class GenerateAnimation:
    def __init__(self, pendulum_path: str,
                 prediction_path: str,
                 gif_path: str,
                 model_path: str = r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\training\model_checkpoints\2024-01-21_22-06-57_rnn_model_rnn_model.h5",
                 output_length: int = output_sequence,
                 sequence_length: int = input_sequence):
        """
        Initialize the GenerateAnimation object.

        Parameters
        ----------
        pendulum_path : str
            Path to the CSV file containing time series data.
        prediction_path : str
            Path to store intermediate image prediction results.
        gif_path : str
            Path to save the final GIF animation.
        model_path : str, optional
            Path to the pre-trained Keras model file, by default set to a specific file path.
        output_length : int, optional
            Length of the predicted output sequence, by default 300.
        sequence_length : int, optional
            Length of the input sequence, by default 50.
        """
        self.pendulum_path = pendulum_path
        self.prediction_path = prediction_path
        self.gif_path = gif_path
        self.model = keras.models.load_model(h5_file_exists(model_path))
        self.output_length = output_length
        self.sequence_length = sequence_length
        self.overall_length = sequence_length + output_length

    def load_data_from_csv(self) -> pd.DataFrame:
        """
        Extracts DataFrame from CSV file.

        Returns
        -------
        pd.DataFrame
            Data which includes time series data.
        """
        return pd.read_csv(self.pendulum_path, delimiter=",")

    def filter_and_preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters data and converts it into sin, cos values.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Processed input and output data arrays.
        """
        # Remove rows with NaN values
        data.dropna(inplace=True)

        # Cut to the needed length
        data = data.iloc[:self.overall_length, :]

        data['Sin_Angle1'] = np.sin(data['Angle1'])
        data['Cos_Angle1'] = np.cos(data['Angle1'])

        data['Sin_Angle2'] = np.sin(data['Angle2'])
        data['Cos_Angle2'] = np.cos(data['Angle2'])

        sin_angle1 = data["Sin_Angle1"].to_numpy()
        cos_angle1 = data["Cos_Angle1"].to_numpy()

        sin_angle2 = data["Sin_Angle2"].to_numpy()
        cos_angle2 = data["Cos_Angle2"].to_numpy()

        numpy_data = np.column_stack((sin_angle1, cos_angle1, sin_angle2, cos_angle2))

        x_sample = numpy_data[:self.sequence_length]
        y_sample = numpy_data[self.sequence_length:self.sequence_length + self.output_length].flatten()

        x = np.array(x_sample)
        y = np.array(y_sample)

        return x, y

    def predict_and_reconstruct_data(self, prediction_input: np.array,
                                     ground_truth_output: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts and reconstructs data using the loaded model.

        Parameters
        ----------
        prediction_input : np.array
            Input data for prediction.
        ground_truth_output : np.array
            Ground truth output data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Processed prediction and ground truth output arrays.
        """
        prediction_output = self.model.predict(np.expand_dims(prediction_input, axis=0))

        prediction_output_reconstructed = reconstruct_radian(prediction_output.reshape(-1, 4))
        ground_truth_output_reconstructed = reconstruct_radian(ground_truth_output.reshape(-1, 4))

        # Add zeros for the non-prediction time
        prediction_output_final = np.concatenate((np.zeros((self.sequence_length, 2)), prediction_output_reconstructed),
                                                 axis=0)

        ground_truth_output_final = np.concatenate(
            (reconstruct_radian(prediction_input), ground_truth_output_reconstructed), axis=0)

        return prediction_output_final, ground_truth_output_final

    def run_image_generation(self, ground_truth_output: np.array, prediction_output: np.array):
        """
        Generates images based on ground truth and predicted outputs.

        Parameters
        ----------
        ground_truth_output : np.array
            Ground truth output data.
        prediction_output : np.array
            Predicted output data.
        """
        # Use multiprocessing to generate images in parallel
        with Pool() as pool:
            args_list = [(image_index2, y_true, y_pred, self.prediction_path) for image_index2, (y_true, y_pred) in
                         enumerate(zip(ground_truth_output, prediction_output))]
            pool.starmap(generate_image, args_list)

    def build_gif_animation(self):
        """
        Builds a GIF animation from generated images.
        """
        # Use glob to find all PNG files in the specified directory
        image_paths = glob.glob(os.path.join(self.prediction_path, '*.png'))

        # Create a list to store image objects
        images = []

        # Open each image and append it to the list
        for image_path in tqdm(image_paths):
            img = Image.open(image_path)
            images.append(img)

        # Save the GIF
        images[0].save(
            self.gif_path,
            save_all=True,
            append_images=images[1:],
            duration=5,  # Duration between frames in milliseconds
            loop=1,  # 0 means infinite loop
        )

    def main(self):
        """
        Main method to execute the animation generation process.
        """
        data = self.load_data_from_csv()

        validate_data_correctness(data, overall_length=self.overall_length)

        prediction_input, ground_truth_output = self.filter_and_preprocess_data(data=data)

        prediction_output, ground_truth_output = self.predict_and_reconstruct_data(prediction_input=prediction_input,
                                                                                   ground_truth_output=ground_truth_output)

        # plot_returns(prediction_output, ground_truth_output)

        self.run_image_generation(ground_truth_output=ground_truth_output, prediction_output=prediction_output)

        self.build_gif_animation()


if __name__ == "__main__":
    generate_animation = GenerateAnimation(
        pendulum_path=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\data\processed\3.csv",
        prediction_path=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\work_dir\prediction_data",
        gif_path=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\work_dir\output.gif")

    generate_animation.main()
