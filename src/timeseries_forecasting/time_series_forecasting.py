import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import os


sequence_length = 50
output_length = 20
overlap = 1
batch_size = 32
epochs = 500


# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()


def load_data_from_csv(path):
    return pd.read_csv(path, delimiter=";")


def filter_and_preprocess_data(data):
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



"""
X_train, X_test, y_train, y_test
= train_test_split(X, y, test_size=0.2, random_state=1)


X_train, X_val, y_train, y_val
= train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
"""


"""
# Generate sample data (replace with your own data)
num_samples = 1000
num_features = 4

X = np.random.rand(num_samples, num_features)
y = np.random.rand(num_samples, 4)  # Four output values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape the data for RNN input
time_steps = 10
X_train_reshaped = X_train_scaled.reshape(-1, time_steps, num_features)
X_test_reshaped = X_test_scaled.reshape(-1, time_steps, num_features)
"""


def construct_network():
    # Build the RNN model
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(sequence_length, 2), return_sequences=False))
    model.add(Dense(2*output_length))  # Output layer with 2 neurons for forecasting

    model.summary()
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, x_train, x_val, y_train, y_val):
    path_checkpoint = "model_checkpoint.h5"
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=25)

    checkpoint = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
    )

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        callbacks=[early_stopping, checkpoint])

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    return history


# Custom MSE loss function with respect to modulo 1
def custom_mse_modulo(y_true, y_pred):
    y_pred_mod = tf.where(tf.logical_and(tf.greater_equal(y_pred, -0.05), tf.less_equal(y_pred, 1.05)),
                          y_pred - (y_pred // 1),
                          y_pred)
    squared_diff = tf.square(y_true - y_pred_mod)
    mse = tf.reduce_mean(squared_diff, axis=-1)
    return mse


def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    loss = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)

    for x_sample, y_sample in zip(x_test, y_test):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        time_array = np.array(range(sequence_length))
        y_pred = model.predict(np.expand_dims(x_sample, axis=0)).reshape(-1, 2)
        y_sample = y_sample.reshape(-1, 2)
        axes[0].scatter(time_array, x_sample[:, 0], label="Input Sequence")
        axes[0].scatter(list(range(sequence_length, sequence_length+output_length)), y_pred[:, 0], label="Prediction")
        axes[0].scatter(list(range(sequence_length, sequence_length+output_length)), y_sample[:, 0], label="Ground Truth")
        axes[0].set_title("Arm 1")
        axes[0].set_ylim([-3.2, 3.2])
        axes[0].legend()
        axes[0].grid(visible=True)

        axes[1].scatter(time_array, x_sample[:, 1], label="Input Sequence")
        axes[1].scatter(list(range(sequence_length, sequence_length+output_length)), y_pred[:, 1], label="Prediction")
        axes[1].scatter(list(range(sequence_length, sequence_length+output_length)), y_sample[:, 1], label="Ground Truth")
        axes[1].set_title("Arm 2")
        axes[1].set_ylim([-3.2, 3.2])
        axes[1].legend()
        axes[1].grid(visible=True)



        plt.show()




if __name__ == '__main__':
    data = load_data_from_csv(r"C:\PGraf\Arbeit\RL\ZML_GitLab\proj-chaotic-pendulum\DataRecords\2023-08-16_13-12-22_DemoFolder\2023-08-16_13-12-22_log.csv")
    x_train, x_val, x_test, y_train, y_val, y_test = filter_and_preprocess_data(data)
    model = keras.models.load_model('model_checkpoint.h5')
    # model = construct_network()
    # train_model(model, x_train, x_val, y_train, y_val)
    evaluate_model(model, x_test, y_test)