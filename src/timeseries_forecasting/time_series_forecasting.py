import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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


def load_data_from_csv(path):
    return pd.read_csv(path, delimiter=";")


def filter_and_preprocess_data(data, sequence_length, output_length, overlap):
    timestamps = data["Time"].to_numpy()
    angle_1 = data["Angle1"].to_numpy()
    angle_2 = data["Angle2"].to_numpy()
    angular_vel_1 = data["AngularVel1"].to_numpy()
    angular_vel_2 = data["AngularVel2"].to_numpy()

    # angle_1 = np.array(range(2000))
    # angle_2 = np.array(range(2000, 4000))

    numpy_data = np.column_stack((angle_1, angle_2))
    print("DATA SHAPE: {}".format(numpy_data.shape))

    x_list = []
    y_list = []
    for i in range(0, numpy_data.shape[0]-sequence_length-output_length, overlap):
        x_sample = numpy_data[i:i+sequence_length]
        y_sample = numpy_data[i+sequence_length:i+sequence_length+output_length].flatten()

        if np.any(np.isnan(x_sample)) or np.any(np.isnan(y_sample)):
            continue

        x_list.append(x_sample)
        y_list.append(y_sample)

    x = np.array(x_list)
    y = np.array(y_list)
    # y = np.expand_dims(np.array(y_list), axis=1)
    print("DATA SHAPE AFTER: {}, {}".format(x.shape, y.shape))

    # x_scaled = scaler_x.fit_transform(x.reshape((-1, 2))).reshape(-1, sequence_length, 2)
    # y_scaled = scaler_y.fit_transform(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    return x_train, x_val, x_test, y_train, y_val, y_test


class RNNModel:
    def __init__(self, inputs, outputs, layers=1, units=None, model_path=None):
        if model_path:
            self.model = keras.models.load_model(model_path)
            return

        self.construct_network(inputs, outputs, layers=layers, units=units)

    def construct_network(self, inputs, outputs, layers=1, units=None):
        first_lstm = True
        # Build the RNN model
        if units is None:
            units = [32]

        model_input = keras.layers.Input((inputs, 2))
        x = model_input

        for l in range(layers):
            x = LSTM(units[l], activation='relu', return_sequences=(first_lstm and layers > 1))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            first_lstm = False

        x = Dense(2*units[0], activation='relu')(x)
        model_output = Dense(2*outputs)(x)

        self.model = keras.Model(inputs=model_input, outputs=model_output)
        self.model.summary()
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, x_train, x_val, y_train, y_val, epochs=100, batch_size=32):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_rnn_model")
        path_checkpoint = f"./model_checkpoints/{timestamp}_rnn_model.h5"
        path_tensorboard = f"./logs/{timestamp}_rnn_model"
        # early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss")

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
                                 callbacks=[checkpoint, tensorboard_callback])

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

    def evaluate_model(self, x_test, y_test, sequence_length, output_length):
        # Evaluate the model
        loss = self.model.evaluate(x_test, y_test)
        print("Test Loss:", loss)

        for x_sample, y_sample in zip(x_test, y_test):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            time_array = np.array(range(sequence_length))
            y_pred = self.model.predict(np.expand_dims(x_sample, axis=0)).reshape(-1, 2)
            y_sample = y_sample.reshape(-1, 2)
            axes[0].scatter(time_array, x_sample[:, 0], label="Input Sequence")
            axes[0].scatter(list(range(sequence_length, sequence_length+output_length)), y_pred[:, 0],
                            label="Prediction")
            axes[0].scatter(list(range(sequence_length, sequence_length+output_length)), y_sample[:, 0],
                            label="Ground Truth")
            axes[0].set_title("Arm 1")
            axes[0].set_ylim([-3.2, 3.2])
            axes[0].legend()
            axes[0].grid(visible=True)

            axes[1].scatter(time_array, x_sample[:, 1], label="Input Sequence")
            axes[1].scatter(list(range(sequence_length, sequence_length+output_length)), y_pred[:, 1],
                            label="Prediction")
            axes[1].scatter(list(range(sequence_length, sequence_length+output_length)), y_sample[:, 1],
                            label="Ground Truth")
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
    parser.add_argument('-out_len', '--output_length', default=20, type=int, help="", required=False)
    parser.add_argument('-ov', '--overlap', default=1, type=int, help="", required=False)
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help="", required=False)
    parser.add_argument('-ep', '--epochs', default=100, type=int, help="", required=False)

    # Network
    parser.add_argument('-l', '--layers', default=2, type=int, help="", required=False)
    parser.add_argument('-u', '--units', default=[32, 16], type=list, help="", required=False)

    # Paths
    parser.add_argument('-mp', '--model_path', type=str, default=None, required=False)
    parser.add_argument('-dp', '--data_path', type=str, default=None, required=False)

    args = parser.parse_args()

    # Data loading
    data = load_data_from_csv(args.data_path)
    x_train, x_val, x_test, y_train, y_val, y_test = filter_and_preprocess_data(data,
                                                                                args.sequence_length,
                                                                                args.output_length,
                                                                                args.overlap)

    if args.model_path:
        model = keras.models.load_model(args.model_path)
    else:
        model = RNNModel(args.sequence_length, args.output_length, layers=args.layers, units=args.units,
                         model_path=args.model_path)

    if args.train:
        model.train_model(x_train, x_val, y_train, y_val, args.epochs, args.batch_size)

    if args.evaluate:
        model.evaluate_model(x_test, y_test, args.sequence_length, args.output_length)