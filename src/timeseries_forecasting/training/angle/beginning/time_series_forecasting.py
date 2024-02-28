import random

import numpy as np
import tensorflow as tf
import os
from typing import List, Dict
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout, Bidirectional
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import tensorboard
import time
import datetime

'''
Script to train model only from beginning of time series and a specific length, with no overlap. 
Data is in angle format
'''


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


def filter_and_preprocess_data_multiple_series(folder_path: str, sequence_length: int, output_length: int,
                                               overlap: int = -1):
    if overlap < 0:
        overlap = sequence_length + output_length

    file_names = get_file_names(folder_path)

    np.random.seed(42)
    training_file_names = np.random.choice(file_names, int(0.6*len(file_names)))
    remaining_file_names = [f for f in file_names if f not in training_file_names]

    validation_file_names = np.random.choice(remaining_file_names, int(0.5*len(remaining_file_names)))
    test_file_names = [f for f in file_names if f not in validation_file_names]

    x_train_list = []
    y_train_list = []

    x_val_list = []
    y_val_list = []

    x_test_list = []
    y_test_list = []

    for file in file_names:
        data = load_data_from_csv(file)
        if len(data) >= sequence_length + output_length:
            # Convert angles to sine and cosine values for better trainings results
            data['Sin_Angle1'] = np.sin(data['Angle1'])
            data['Cos_Angle1'] = np.cos(data['Angle1'])

            data['Sin_Angle2'] = np.sin(data['Angle2'])
            data['Cos_Angle2'] = np.cos(data['Angle2'])

            sin_angle1 = data["Sin_Angle1"].to_numpy()
            cos_angle1 = data["Cos_Angle1"].to_numpy()

            sin_angle2 = data["Sin_Angle2"].to_numpy()
            cos_angle2 = data["Cos_Angle2"].to_numpy()

            numpy_data = np.column_stack((sin_angle1, cos_angle1, sin_angle2, cos_angle2))
            # print("DATA SHAPE: {}".format(numpy_data.shape))

            for i in range(0, numpy_data.shape[0] - sequence_length - output_length, overlap):
                x_sample = numpy_data[i:i+sequence_length]
                y_sample = numpy_data[i+sequence_length:i+sequence_length + output_length].flatten()

                if file in training_file_names:
                    x_train_list.append(x_sample)
                    y_train_list.append(y_sample)
                elif file in validation_file_names:
                    x_val_list.append(x_sample)
                    y_val_list.append(y_sample)
                else:
                    x_test_list.append(x_sample)
                    y_test_list.append(y_sample)

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)

    x_val = np.array(x_val_list)
    y_val = np.array(y_val_list)

    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list)

    print("DATA SHAPE AFTER: {}, {}".format(x_train.shape, y_train.shape))

    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25,
                                                            # random_state=1)  # 0.25 x 0.8 = 0.2
    return x_train, x_val, x_test, y_train, y_val, y_test


class RNNModel:
    def __init__(self, inputs, outputs, layers=1, units=None, model_path=None, input_shape=4):
        if model_path:
            self.model = keras.models.load_model(model_path)
            return
        self.input_shape = input_shape
        self.construct_network(inputs, outputs, layers=layers, units=units)

    def construct_network(self, inputs, outputs, layers, units=None):
        # Set default units if not provided
        if units is None:
            units = [32]

        # Build the RNN model
        model_input = keras.layers.Input((inputs, self.input_shape))
        x = model_input

        for i in range(layers):
            print(i)
            # First LSTM layer
            x = LSTM(units[i], activation='relu', return_sequences=(i < layers-1))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)

        # Dense layers
        # x = Dense(2 * units[-1], activation='linear')(x)
        model_output = Dense(self.input_shape * outputs)(x)

        self.model = keras.Model(inputs=model_input, outputs=model_output)
        self.model.summary()

        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, x_train, x_val, y_train, y_val, epochs=100, batch_size=32):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_rnn_model")
        path_checkpoint = f"./model_checkpoints/{timestamp}_rnn_model.keras"
        path_tensorboard = f"./logs/{timestamp}_rnn_model"
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

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

    def reconstruct_radian(self, sin_value, cos_value):
        return np.arctan2(sin_value, cos_value)

    def evaluate_model(self, x_test, y_test, sequence_length, output_length):
        # Evaluate the model
        loss = self.model.evaluate(x_test, y_test)
        print("Test Loss:", loss)

        for x_sample, y_sample in zip(x_test, y_test):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            time_array = np.array(range(sequence_length))
            y_pred = self.model.predict(np.expand_dims(x_sample, axis=0)).reshape(-1, 4)
            y_sample = y_sample.reshape(-1, 4)
            axes[0].plot(time_array, self.reconstruct_radian(x_sample[:, 0], x_sample[:, 1]), label="Input Sequence",
                         linestyle='-', marker='o')
            axes[0].plot(list(range(sequence_length, sequence_length + output_length)),
                         self.reconstruct_radian(y_pred[:, 0], y_pred[:, 1]),
                         label="Prediction", linestyle='-', marker='o')
            axes[0].plot(list(range(sequence_length, sequence_length + output_length)),
                         self.reconstruct_radian(y_sample[:, 0], y_sample[:, 1]),
                         label="Ground Truth", linestyle='-', marker='o')
            axes[0].set_title("Arm 1")
            axes[0].set_ylim([-3.2, 3.2])
            axes[0].legend()
            axes[0].grid(visible=True)

            axes[1].plot(time_array, self.reconstruct_radian(x_sample[:, 2], x_sample[:, 3]), label="Input Sequence",
                         linestyle='-', marker='o')
            axes[1].plot(list(range(sequence_length, sequence_length + output_length)),
                         self.reconstruct_radian(y_pred[:, 2], y_pred[:, 3]),
                         label="Prediction", linestyle='-', marker='o')
            axes[1].plot(list(range(sequence_length, sequence_length + output_length)),
                         self.reconstruct_radian(y_sample[:, 2], y_sample[:, 3]),
                         label="Ground Truth", linestyle='-', marker='o')
            axes[1].set_title("Arm 2")
            axes[1].set_ylim([-3.2, 3.2])
            axes[1].legend()
            axes[1].grid(visible=True)

            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    # Modes
    parser.add_argument('-t', '--train', action=argparse.BooleanOptionalAction, default=True,
                        help="", required=False)
    parser.add_argument('-e', '--evaluate', action=argparse.BooleanOptionalAction, default=True,
                        help="", required=False)

    # Training Parameters
    parser.add_argument('-seq_len', '--sequence_length', default=200, type=int, help="", required=False)
    parser.add_argument('-out_len', '--output_length', default=100, type=int, help="", required=False)
    parser.add_argument('-ov', '--overlap', default=-1, type=int, help="", required=False)
    parser.add_argument('-bs', '--batch_size', default=512, type=int, help="", required=False)
    parser.add_argument('-ep', '--epochs', default=400, type=int, help="", required=False)

    # Network
    parser.add_argument('-l', '--layers', default=2, type=int, help="", required=False)
    parser.add_argument('-u', '--units', default=[64, 128], type=list, help="", required=False)

    # Paths
    parser.add_argument('-mp', '--model_path', type=str, default=None,
                        required=False)
    parser.add_argument('-dp', '--folder_path', type=str,
                        default=r"..\..\..\data\processed",
                        required=False)

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    x_train, x_val, x_test, y_train, y_val, y_test = filter_and_preprocess_data_multiple_series(
        os.path.join(args.folder_path),
        args.sequence_length,
        args.output_length, overlap=20)

    model = RNNModel(args.sequence_length, args.output_length, layers=args.layers, units=args.units,
                     model_path=args.model_path)

    if args.train:
        model.train_model(x_train, x_val, y_train, y_val, args.epochs, args.batch_size)

    if args.evaluate:
        model.evaluate_model(x_test, y_test, args.sequence_length, args.output_length)
