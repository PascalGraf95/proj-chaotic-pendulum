from camera_controller import IDSCameraController
import os.path
import numpy as np
from angle_detection import AngleDetector


def main():
    # initialize camera and angle detector object
    camera = IDSCameraController(param_file=r"../CameraParameters/cp_AngleDetection.ini")
    detection = AngleDetector(camera, definition=0)

    # initialize variables for counting frames and storing reference values
    frame_count = 0
    reference_values = np.zeros(2, dtype=int)

    print("Checking for references. Do not move the pendulum.")

    # loop until reference points are found
    while np.isnan(detection.pos_B).all() or np.isnan(detection.pos_C).all():
        frame_count += 1
        # check if frame count maximum is reached
        if frame_count >= 1000:
            camera.close_camera_connection()
            raise RuntimeError(f"Reference points not found after {frame_count} frames. Check conditions.")

        # use angle detection method to find positions of coloured dots
        detection.get_angle()

        # calculate vector between C and B
        vec_bc = detection.pos_C - detection.pos_B

        # Aresults from following the vector BC starting from point b in the opposite direction
        # since the arms of the pendulum are of equal length.
        pos_a = detection.pos_B - vec_bc

        # Vector between A and B is used as reference vector for the angle calculation
        ref_vec = detection.pos_B - pos_a

        # store A and reference vector in array
        reference_values = np.array((pos_a, ref_vec))

    # create file path for saving the values
    if not os.path.exists(r"../CalibrationData"):
        os.makedirs(r"../CalibrationData")

    # save the values as a numpy file
    np.save('../CalibrationData/ReferenceAxis.npy', reference_values)
    camera.close_camera_connection()

    print("Reference calibration finished successfully.")


if __name__ == '__main__':
    main()
