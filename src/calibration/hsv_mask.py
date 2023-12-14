import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, Button
import os

# Load an image
image_directory = r'C:\Users\Marco\dev\backups\2023-08-11_14-29-45_DemoFolder'
image_paths = os.listdir(image_directory)
current_idx = 0
original_img = cv2.imread(os.path.join(image_directory, image_paths[0]))
hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)


def update_mask(lower, upper):
    global original_img
    global hsv_img
    lower_hsv = np.array([lower[0], lower[1], lower[2]])
    upper_hsv = np.array([upper[0], upper[1], upper[2]])
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(original_img, original_img, mask=mask)
    cv2.imshow('Result', result)


def update_image():
    global current_idx
    global original_img
    global hsv_img
    original_img = cv2.imread(os.path.join(image_directory, image_paths[current_idx]))
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    current_idx += 1
    on_change(event=None)


def on_change(event):
    lower = [lower_h.get(), lower_s.get(), lower_v.get()]
    upper = [upper_h.get(), upper_s.get(), upper_v.get()]
    update_mask(lower, upper)


def on_image_change():
    update_image()


# Green
# 50, 55, 110
# 108 180 218

# Red
# 0, 79, 188
# 13, 225, 255

# Create a GUI window
root = tk.Tk()
root.title("HSV Masking")

# Create sliders for HSV ranges
lower_h = Scale(root, from_=0, to=179, orient="horizontal", label="Lower H")
lower_s = Scale(root, from_=0, to=255, orient="horizontal", label="Lower S")
lower_v = Scale(root, from_=0, to=255, orient="horizontal", label="Lower V")

upper_h = Scale(root, from_=0, to=179, orient="horizontal", label="Upper H")
upper_s = Scale(root, from_=0, to=255, orient="horizontal", label="Upper S")
upper_v = Scale(root, from_=0, to=255, orient="horizontal", label="Upper V")

next_image = Button(root, text="Click Me", command=on_image_change)

lower_h.bind("<Motion>", on_change)
lower_s.bind("<Motion>", on_change)
lower_v.bind("<Motion>", on_change)
upper_h.bind("<Motion>", on_change)
upper_s.bind("<Motion>", on_change)
upper_v.bind("<Motion>", on_change)

# Pack the sliders
lower_h.pack()
lower_s.pack()
lower_v.pack()
upper_h.pack()
upper_s.pack()
upper_v.pack()
next_image.pack()

# Initialize the GUI
root.mainloop()

cv2.destroyAllWindows()
