"""
    Access Webcam through OpenCV Library
    BioCapsule
    Edwin Sanchez

    Current Objective:
    - Get a frame from the webcam & save the image
"""

import os.path as pth
import cv2
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk


def main():
    # test frame extraction, display, and frame saving
    frame = get_frame_from_webcam()
    display_frame(frame=frame)
    save_img(frame=frame)


# gets a frame from the webcam
def get_frame_from_webcam(camera_id:int=0)-> np.ndarray:
    # access the a registered camera device, grab a frame
    vid = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    _, frame = vid.read()
    vid.release()
    return frame


# gets a frame from a video capture and converts it to a PhotoImage for tkinter
def get_tk_frame(video_capture)-> ImageTk.PhotoImage:
    # get last frame
    cv2img = cv2.cvtColor(video_capture.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2img)
    return ImageTk.PhotoImage(image=img)


# display camera feed to tkinter window
def display_camera(video_capture, video_label:tk.Label, update_delay_ms:int=20)-> None:
    imgtk = get_tk_frame(video_capture=video_capture)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(update_delay_ms, display_camera, video_capture, video_label)


# display a frame in a window. wait until user closes window
def display_frame(frame:np.ndarray)-> None:
    cv2.imshow('frame', frame) # show frame
    cv2.waitKey(0) # wait for the user to close the window
    cv2.destroyAllWindows() # make sure windows are destroyed afterwards


# save the image to a certain location
def save_img(frame:np.ndarray, dir:str="./", file_name:str="frame.png")->None:
    path = pth.join(dir, file_name)
    success = cv2.imwrite(filename=path, img=frame)
    if success == False:
        raise "Failed to save image!"


if __name__ == "__main__":
    main()
