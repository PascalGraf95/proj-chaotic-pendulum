import numpy as np
import cv2


def reconstruct_radian(array: np.ndarray) -> np.ndarray:
    """
    Reconstruct radians from sine and cosine values.

    Parameters
    ----------
    array : np.ndarray
        Input array with sine and cosine values.

    Returns
    -------
    np.ndarray
        Reconstructed radians.
    """
    reconstructed1 = np.arctan2(array[:, 0], array[:, 1])
    reconstructed2 = np.arctan2(array[:, 2], array[:, 3])
    return np.column_stack((reconstructed1, reconstructed2))


def overlay_images(background, overlay):
    """
    Overlay two images with alpha channels.

    Parameters
    ----------
    background : np.ndarray
        Background image.
    overlay : np.ndarray
        Overlay image.

    Returns
    -------
    np.ndarray
        Resulting image after overlay.
    """
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:, :, 3] / 255.0
    alpha_overlay = overlay[:, :, 3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[:, :, color] = alpha_overlay * overlay[:, :, color] + \
                                  alpha_background * background[:, :, color] * (1 - alpha_overlay)

    # set adjusted alpha and denormalize back to 0-255
    background[:, :, 3] = (1 - (1 - alpha_overlay) * (1 - alpha_background)) * 255
    return background


def rotate_image(image, angle):
    """
    Rotate an image by a given angle.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated image.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def translate_image(image, x, y):
    """
    Translate an image by a given offset.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    x : int
        Horizontal translation offset.
    y : int
        Vertical translation offset.

    Returns
    -------
    np.ndarray
        Translated image.
    """
    rows, cols = image.shape[:2]
    mat = np.float32([[1, 0, y], [0, 1, x]])
    result = cv2.warpAffine(image, mat, (cols, rows))
    return result
