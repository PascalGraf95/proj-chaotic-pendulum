import tensorflow as tf
import numpy as np


def custom_mse_modulo(y_true, y_pred):
    y_pred_original = y_pred.copy()
    mask = (y_pred >= -0.05) & (y_pred <= 1.05)
    y_pred[mask] = y_pred[mask] % 1
    y_pred2 = y_pred_original - (y_pred_original // 1)
    absolute_diff = (y_true - y_pred)
    squared_diff = tf.square(absolute_diff)
    mse = tf.reduce_mean(squared_diff, axis=-1)
    return mse


if __name__ == '__main__':
    custom_mse_modulo(np.array([0.5, 0.95, 0.99, 0.01, 0.2, 1, 0.04, 1.0]),
                      np.array([0.6, -0.01, 1.01, 1.01, 1.1, 3, 1.04, -0.03]))