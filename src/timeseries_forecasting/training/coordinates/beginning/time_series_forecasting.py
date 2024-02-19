import numpy as np
import tensorflow as tf
import os
from typing import List, Dict
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import tensorboard
import time
import datetime


# Custom MSE loss function with respect to modulo 1
def custom_mse_modulo(y_true, y_pred):
    y_pred_mod = tf.where(tf.logical_and(tf.greater_equal(y_pred, -0.05), tf.less_equal(y_pred, 1.05)),
                          y_pred - (y_pred // 1),
                          y_pred)
    squared_diff = tf.square(y_true - y_pred_mod)
    mse = tf.reduce_mean(squared_diff, axis=-1)
    return mse


def load_data_from_csv(path: str):
    return pd.read_csv(path, delimiter=",")


def get_file_names(folder_path: str) -> List:
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Error: Folder '{folder_path}' does not exist.")

    # Get a list of all files in the folder
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path)]


def filter_and_preprocess_data_multiple_series(folder_path: str, sequence_length: int, output_length: int):
    file_names = get_file_names(folder_path)

    center = (0, 0)
    arm_length = 115

    x_list = []
    y_list = []
    x = None
    y = None
    for file in file_names:
        data = load_data_from_csv(file)
        if len(data) >= sequence_length + output_length:
            data['x1'] = center[0] + arm_length * np.sin(data['Angle1'])
            data['y1'] = center[1] - arm_length * np.cos(data['Angle1'])

            data['x2'] = data['x1'] + arm_length * np.sin(data['Angle1'] + data['Angle2'])
            data['y2'] = data['y1'] - arm_length * np.cos(data['Angle1'] + data['Angle2'])

            x1 = data["x1"].to_numpy()
            y1 = data["y1"].to_numpy()

            x2 = data["x2"].to_numpy()
            y2 = data["y2"].to_numpy()

            numpy_data = np.column_stack((x1, y1, x2, y2))
            # print("DATA SHAPE: {}".format(numpy_data.shape))

            x_sample = numpy_data[:sequence_length]
            y_sample = numpy_data[sequence_length:sequence_length + output_length].flatten()

            x_list.append(x_sample)
            y_list.append(y_sample)

    x = np.array(x_list)
    y = np.array(y_list)

    print("DATA SHAPE AFTER: {}, {}".format(x.shape, y.shape))

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    x_norm = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    y_norm = scaler.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

    x_train, x_val, y_train, y_val = train_test_split(x_norm, y_norm, test_size=0.2, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25,
                                                        random_state=1)  # 0.25 x 0.8 = 0.2

    return x_train, x_val, x_test, y_train, y_val, y_test


class RNNModel:
    def __init__(self, inputs, outputs, layers=1, units=None, model_path=None, input_shape=4):
        if model_path:
            self.model = keras.models.load_model(model_path)
            return
        self.input_shape = input_shape
        self.construct_network(inputs, outputs, layers=layers, units=units)

    def construct_network(self, inputs, outputs, layers=1, units=None):
        first_lstm = True
        # Build the RNN model
        if units is None:
            units = [32]

        model_input = keras.layers.Input((inputs, self.input_shape))
        x = model_input

        for l in range(layers):
            x = Bidirectional(LSTM(units[l], activation='relu', return_sequences=(first_lstm and layers > 1)))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            first_lstm = False

        x = Dense(2 * units[0], activation='relu')(x)
        model_output = Dense(self.input_shape * outputs)(x)

        self.model = keras.Model(inputs=model_input, outputs=model_output)
        self.model.summary()

        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def train_model(self, x_train, x_val, y_train, y_val, epochs=100, batch_size=32):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_rnn_model")
        path_checkpoint = f"./model_checkpoints/{timestamp}_rnn_model.h5"
        path_tensorboard = f"./logs/{timestamp}_rnn_model"
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

        checkpoint = keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=False,
            save_best_only=True,
        )

        # Create a TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir=path_tensorboard)

        # Train the model
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                                 callbacks=[checkpoint, tensorboard_callback, early_stopping])

        return history

    def evaluate_model(self, x_test, y_test, sequence_length, output_length):
        # Evaluate the model
        loss = self.model.evaluate(x_test, y_test)
        print("Test Loss:", loss)

        for x_sample, y_sample in zip(x_test, y_test):
            fig, axes = plt.subplots(1, 4, figsize=(12, 6))
            time_array = np.array(range(sequence_length))
            y_pred = self.model.predict(np.expand_dims(x_sample, axis=0)).reshape(-1, 4)
            y_sample = y_sample.reshape(-1, 4)
            axes[0].plot(time_array, x_sample[:, 0], label="Input Sequence", linestyle='-', marker='o')
            axes[0].plot(list(range(sequence_length, sequence_length + output_length)), y_pred[:, 0],
                         label="Prediction", linestyle='-', marker='o')
            axes[0].plot(list(range(sequence_length, sequence_length + output_length)), y_sample[:, 0],
                         label="Ground Truth", linestyle='-', marker='o')
            axes[0].set_title("x1")
            axes[0].set_ylim([-3.2, 3.2])
            axes[0].legend()
            axes[0].grid(visible=True)

            axes[1].plot(time_array, x_sample[:, 1], label="Input Sequence", linestyle='-', marker='o')
            axes[1].plot(list(range(sequence_length, sequence_length + output_length)), y_pred[:, 1],
                         label="Prediction", linestyle='-', marker='o')
            axes[1].plot(list(range(sequence_length, sequence_length + output_length)), y_sample[:, 1],
                         label="Ground Truth", linestyle='-', marker='o')
            axes[1].set_title("y1")
            axes[1].set_ylim([-3.2, 3.2])
            axes[1].legend()
            axes[1].grid(visible=True)

            axes[2].plot(time_array, x_sample[:, 2], label="Input Sequence", linestyle='-', marker='o')
            axes[2].plot(list(range(sequence_length, sequence_length + output_length)), y_pred[:, 2],
                         label="Prediction", linestyle='-', marker='o')
            axes[2].plot(list(range(sequence_length, sequence_length + output_length)), y_sample[:, 2],
                         label="Ground Truth", linestyle='-', marker='o')
            axes[2].set_title("x2")
            axes[2].set_ylim([-3.2, 3.2])
            axes[2].legend()
            axes[2].grid(visible=True)

            axes[3].plot(time_array, x_sample[:, 3], label="Input Sequence", linestyle='-', marker='o')
            axes[3].plot(list(range(sequence_length, sequence_length + output_length)), y_pred[:, 3],
                         label="Prediction", linestyle='-', marker='o')
            axes[3].plot(list(range(sequence_length, sequence_length + output_length)), y_sample[:, 3],
                         label="Ground Truth", linestyle='-', marker='o')
            axes[3].set_title("y2")
            axes[3].set_ylim([-3.2, 3.2])
            axes[3].legend()
            axes[3].grid(visible=True)

            plt.show()

    def eval_single_losses(self, y_pred, y_ground):
        from sklearn.metrics import mean_squared_error

        mse0 = mean_squared_error(y_ground[0], y_pred[0])
        mse1 = mean_squared_error(y_ground[1], y_pred[1])
        mse2 = mean_squared_error(y_ground[2], y_pred[2])
        mse3 = mean_squared_error(y_ground[3], y_pred[3])

        mse = (mse0 + mse1 + mse2 + mse3) / 4

        print(f"x1: {mse0} y1: {mse1} x2: {mse2} y2: {mse3} overall: {mse}")

    def evaluate_model_trajectory(self, x_test, y_test):
        # Evaluate the model
        loss = self.model.evaluate(x_test, y_test)
        print("Test Loss:", loss)

        for x_sample, y_sample in zip(x_test, y_test):
            y_pred = self.model.predict(np.expand_dims(x_sample, axis=0)).reshape(-1, 4)
            y_sample = y_sample.reshape(-1, 4)

            self.eval_single_losses(y_pred, y_sample)

            plt.figure()
            plt.plot(y_sample[:, 0], y_sample[:, 1], label='Arm 1 groundtruth', color='blue')
            plt.plot(y_sample[:, 2], y_sample[:, 3], label='Arm 2 groundtruth', color='green')
            plt.plot(y_pred[:, 0], y_pred[:, 1], label='Arm 1 prediction', color='red')
            plt.plot(y_pred[:, 2], y_pred[:, 3], label='Arm 2 prediction', color='yellow')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Chaotic Pendulum Positions')
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    # Modes
    parser.add_argument('-t', '--train', action=argparse.BooleanOptionalAction, default=False,
                        help="", required=False)
    parser.add_argument('-e', '--evaluate', action=argparse.BooleanOptionalAction, default=False,
                        help="", required=False)

    # Training Parameters
    parser.add_argument('-seq_len', '--sequence_length', default=100, type=int, help="", required=False)
    parser.add_argument('-out_len', '--output_length', default=50, type=int, help="", required=False)
    parser.add_argument('-ov', '--overlap', default=1, type=int, help="", required=False)
    parser.add_argument('-bs', '--batch_size', default=256, type=int, help="", required=False)
    parser.add_argument('-ep', '--epochs', default=400, type=int, help="", required=False)

    # Network
    parser.add_argument('-l', '--layers', default=2, type=int, help="", required=False)
    parser.add_argument('-u', '--units', default=[16, 8], type=list, help="", required=False)

    # Paths
    parser.add_argument('-mp', '--model_path', type=str, default=None,
                        required=False)
    parser.add_argument('-dp', '--folder_path', type=str,
                        default=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\data\processed",
                        required=False)

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    x_train, x_val, x_test, y_train, y_val, y_test = filter_and_preprocess_data_multiple_series(
        os.path.join(args.folder_path),
        args.sequence_length,
        args.output_length)

    model = RNNModel(args.sequence_length, args.output_length, layers=args.layers, units=args.units,
                     model_path=args.model_path)

    if args.train:
        model.train_model(x_train, x_val, y_train, y_val, args.epochs, args.batch_size)

    if args.evaluate:
        model.evaluate_model_trajectory(x_test, y_test)
        # model.evaluate_model(x_test, y_test, args.sequence_length, args.output_length)
