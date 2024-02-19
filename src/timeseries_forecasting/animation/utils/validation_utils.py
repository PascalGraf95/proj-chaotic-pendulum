import os
import numpy as np
import pandas as pd


class GenerateAnimationException(Exception):
    """Base exception class for GenerateAnimation."""


class GenerateAnimationExceptionShapeNotCorrect(GenerateAnimationException):
    """Exception raised when the shape of the DataFrame is not correct."""


class GenerateAnimationExceptionQualityNotCorrect(GenerateAnimationException):
    """Exception raised when the quality of the DataFrame is not correct."""


class GenerateAnimationExceptionLengthNotCorrect(GenerateAnimationException):
    """Exception raised when the length of the DataFrame is not correct."""


class GenerateAnimationExceptionTypeNotCorrect(GenerateAnimationException):
    """Exception raised when the input data type is not correct."""


class GenerateAnimationExceptionDataNearZero(GenerateAnimationException):
    """Exception raised when all values in the DataFrame are nearly zero."""


class GenerateAnimationExceptionModelFileNotFound(GenerateAnimationException):
    """Exception raised when the model file is not found."""


def h5_file_exists(file_path: str) -> str:
    """
    Check if an H5 file exists and is valid.

    Parameters
    ----------
    file_path : str
        Path to the H5 file.

    Returns
    -------
    str
        The input file path if it's valid.

    Raises
    ------
    GenerateAnimationExceptionModelFileNotFound
        If the file is either nonexistent or has the wrong file type.
    """
    if os.path.exists(file_path) and os.path.isfile(file_path) and file_path.lower().endswith('.h5'):
        return file_path
    else:
        raise GenerateAnimationExceptionModelFileNotFound('Model file is either nonexistent or has the wrong file type')


def validate_data_type(data: pd.DataFrame):
    """
    Validate the type of input data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.

    Raises
    ------
    GenerateAnimationExceptionTypeNotCorrect
        If the input data is not a DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise GenerateAnimationExceptionTypeNotCorrect("Input data is not a DataFrame.")


def validate_data_shape(data: pd.DataFrame):
    """
    Validate the shape of the input data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.

    Raises
    ------
    GenerateAnimationExceptionShapeNotCorrect
        If the shape of the DataFrame is not correct (at least 4 columns).
    """
    if data.shape[1] < 4:
        raise GenerateAnimationExceptionShapeNotCorrect("DataFrame shape is not correct. It has to be (:,4)")


def validate_data_quality(data: pd.DataFrame):
    """
    Validate the quality of the input data. If too many NaN values are in the data

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.

    Raises
    ------
    GenerateAnimationExceptionQualityNotCorrect
        If the shape of the DataFrame is not correct.
    """
    threshold = 30

    # Count the NaN values
    nan_values_angle1 = data['Angle1'].isna().sum()
    nan_values_angle2 = data['Angle2'].isna().sum()

    if nan_values_angle1 > threshold or nan_values_angle2 > threshold:
        raise GenerateAnimationExceptionQualityNotCorrect(
            "DataFrame quality is not good enough data contains out of to many NaN values. Try again")


def validate_data_length(data: pd.DataFrame, overall_length: int):
    """
    Validate the length of the input data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    overall_length : int
        Desired overall length.

    Raises
    ------
    GenerateAnimationExceptionLengthNotCorrect
        If the length of the DataFrame is not correct (at least the specified overall length).
    """
    if len(data) < overall_length:
        raise GenerateAnimationExceptionLengthNotCorrect("DataFrame length is not correct.")


def check_pendulum_not_moved(data: pd.DataFrame):
    """
    Check if all values in the DataFrame are nearly zero.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.

    Raises
    ------
    GenerateAnimationExceptionDataNearZero
        If all values in the DataFrame are nearly zero, indicating that the pendulum might not have moved.
    """
    if np.allclose(data.values, 0, atol=0.05):
        raise GenerateAnimationExceptionDataNearZero(
            "All values in the DataFrame are nearly zero. Did you move the pendulum?")


def validate_data_correctness(data: pd.DataFrame, overall_length: int):
    """
    Validate the correctness of the input data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    overall_length : int
        Desired overall length.

    Raises
    ------
    GenerateAnimationExceptionTypeNotCorrect
        If the input data is not a DataFrame.
    GenerateAnimationExceptionShapeNotCorrect
        If the shape of the DataFrame is not correct (at least 4 columns).
    GenerateAnimationExceptionLengthNotCorrect
        If the length of the DataFrame is not correct (at least the specified overall length).
    GenerateAnimationExceptionDataNearZero
        If all values in the DataFrame are nearly zero, indicating that the pendulum might not have moved.
    """
    validate_data_type(data)
    validate_data_quality(data)
    validate_data_shape(data)
    validate_data_length(data, overall_length)
    check_pendulum_not_moved(data)
