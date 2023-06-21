"""
    Access Webcam through OpenCV Library
    BioCapsule
    Edwin Sanchez

    Current Objective:
    - Get a frame from the webcam & save the image
"""

import os.path as pth
import cv2
import numpy as np


def main():
    frame = get_frame_from_webcam()
    display_frame(frame=frame)
    save_img(frame=frame)


# gets a frame from the webcam
def get_frame_from_webcam(camera_id:int=0)-> np.ndarray:
    # access the first registered camera device, grab a frame
    vid = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    _, frame = vid.read()
    vid.release()
    return frame


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
