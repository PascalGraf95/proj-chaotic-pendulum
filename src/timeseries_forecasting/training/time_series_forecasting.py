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

    x_list = []
    y_list = []
    x = None
    y = None
    for file in file_names:
        data = load_data_from_csv(file)
        if len(data) >= sequence_length + output_length:
            # Convert angles to sine and cosine values
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

            x_sample = numpy_data[:sequence_length]
            y_sample = numpy_data[sequence_length:sequence_length + output_length].flatten()

            x_list.append(x_sample)
            y_list.append(y_sample)

            if isinstance(x, np.ndarray):
                x = np.concatenate((x, np.array(x_list)))
                y = np.concatenate((y, np.array(y_list)))
            else:
                x = np.array(x_list)
                y = np.array(y_list)

    print("DATA SHAPE AFTER: {}, {}".format(x.shape, y.shape))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
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

        def r2_metric(y_true, y_pred):
            SS_res = K.sum(K.square(y_true - y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            r2 = 1 - SS_res / (SS_tot + K.epsilon())
            return r2

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

        """
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        """
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
    parser.add_argument('-t', '--train', action=argparse.BooleanOptionalAction, default=False,
                        help="", required=False)
    parser.add_argument('-e', '--evaluate', action=argparse.BooleanOptionalAction, default=False,
                        help="", required=False)

    # Training Parameters
    parser.add_argument('-seq_len', '--sequence_length', default=50, type=int, help="", required=False)
    parser.add_argument('-out_len', '--output_length', default=100, type=int, help="", required=False)
    parser.add_argument('-ov', '--overlap', default=1, type=int, help="", required=False)
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help="", required=False)
    parser.add_argument('-ep', '--epochs', default=300, type=int, help="", required=False)

    # Network
    parser.add_argument('-l', '--layers', default=2, type=int, help="", required=False)
    parser.add_argument('-u', '--units', default=[32, 16], type=list, help="", required=False)

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
        model.evaluate_model(x_test, y_test, args.sequence_length, args.output_length)
