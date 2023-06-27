"""
    Master Setup - Live Active Auth. Testing
    BioCapsule
    Edwin Sanchez

    TODO:
    - Display live Camera Feed (DONE)
    - Capture individual frames w/ a button (display separately from live feed) (DONE)
    - Password + Face Auth.
    - Display authentication determination (current yes/no of authenticator)
    - Get Latency of the system
        - CPU/GPU
        - Non-BC/BC
    - Train Binary Classifier on my face

    Other TODO:
    * Add a README to project
"""

import tkinter as tk # basic gui for quick prototyping
import cv2
import webcam # convenience functions for accesssing the web cam

from typing import Tuple


# biocapsule & face recognition
#import src.biocapsule as bc # import biocapsule code
#import src.face as facerec # import face recognition book


def main():
    # create window
    window = tk.Tk(screenName="BioCapsule")

    # get first registered camera
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # setup parts of GUI
    setup_video_feed(window=window, video_capture=video_capture)
    capt_btn, frame_label = setup_frame_capture(window=window, video_capture=video_capture)

    # execute the main loop -> begin display
    window.mainloop() # blocks execution

    video_capture.release()


# setup frame capture using the GUI
def setup_frame_capture(window:tk.Tk, video_capture)-> Tuple[tk.Button, tk.Label]:
    # setup display for captured frame for tkinter window
    captured_frame_label = tk.Label(window)
    captured_frame_label.grid(row=0, column=1)

    # authentication square (green for authenticated, red otherwise)
    canvas = tk.Canvas(window, width=500, height=240)
    canvas.grid(row=1,column=0)
    rectangle = canvas.create_rectangle(100, 100, 400, 400, fill='blue')
    set_color(canvas=canvas, rectangle=rectangle)

    # what the button should do (in this case, capture a frame and update another label)
    frame = ()
    def btn_callback()-> None:
        # get img from webcam
        frame, imgtk = webcam.get_tk_and_normal_frames(video_capture=video_capture)
        
        # set captured frame label to new image
        captured_frame_label.imgtk = imgtk
        captured_frame_label.configure(image=imgtk)

        # perform authentication!!!! (and set color of auth square)
        set_color(canvas=canvas, rectangle=rectangle, is_authenticated=is_authorized())

        """
            1. Face Pre-processing
            2. Feat. Ext.
            3. Compute BioCapsule
            4. Run Auth.
        """

    # create a button to capture frames
    frame_capture_btn = tk.Button(
        master=window,
        text="Capture Frame",
        width=25,
        height=5,
        bg="green",
        fg="white",
        command=btn_callback # what the button should do
    ) # end button init
    frame_capture_btn.grid(row=1, column=1)

    return (frame_capture_btn, captured_frame_label)


# authenticate the user
import random
def is_authorized():
    return round(random.random()) >= 0.5 # random for now


# convenience function for changing the color of the authentication rectangle
def set_color(canvas:tk.Canvas, rectangle, is_authenticated:bool=False):
    if is_authenticated:
        canvas.itemconfig(rectangle, fill='green')
    else:
        canvas.itemconfig(rectangle, fill='red')



# perform process to setup video feed
def setup_video_feed(window:tk.Tk, video_capture)-> None:
    # setup display for video for tkinter window
    video_label = tk.Label(window)
    video_label.grid(row=0, column=0)

    # display camera feed to window
    webcam.display_camera(video_capture=video_capture, video_label=video_label)

if __name__ == "__main__":
    main()
