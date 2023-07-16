"""
    Master Setup - Live Active Auth. Testing
    BioCapsule
    Edwin Sanchez

    TODO:
    - Display live Camera Feed (DONE)
    - Capture individual frames w/ a button (display separately from live feed) (DONE)
    - Password + Face Auth.
    - Get Latency of the system
        - CPU/GPU
        - Non-BC/BC

    Other TODO:
    * Add a README to project
"""

import tkinter as tk # basic gui for quick prototyping
import cv2
import webcam # convenience functions for accesssing the web cam

from typing import Tuple

import pickle

import os
import time
import json
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, '../../src/')

# biocapsule & face recognition
import biocapsule as bc
import face as fr

def main():
    # test the latency of the system
    #get_latency()
    #return

    # create window
    window = tk.Tk(screenName="BioCapsule")

    # get first registered camera
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # load face model
    face = fr.ArcFace(0, "mtcnn")

    # load regression model
    model = load_classifier(pkl_file_path="./edwin_classifier.pkl")

    # create bc_generator
    bc_gen = bc.BioCapsuleGenerator()

    # load rs_feat
    rs_feature = get_rs_feature()

    # setup parts of GUI
    setup_video_feed(window=window, video_capture=video_capture)
    capt_btn, frame_label = setup_frame_capture(
        window=window, 
        video_capture=video_capture,
        feat_ext_model=face,
        classifier_model=model,
        bc_generator=bc_gen,
        rs_feat=rs_feature)

    # execute the main loop -> begin display
    window.mainloop() # blocks execution

    video_capture.release()


# setup frame capture using the GUI
def setup_frame_capture(window:tk.Tk, 
                        video_capture, 
                        classifier_model, 
                        feat_ext_model, 
                        bc_generator, 
                        rs_feat)-> Tuple[tk.Button, tk.Label]:
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

        # get features from user
        feature = feat_ext_model.extract(frame)

        # compute biocapsule
        biocapsule = bc_generator.biocapsule(user_feature=feature, rs_feature=rs_feat)
        # perform authentication!!!! (and set color of auth square)
        set_color(
            canvas=canvas, 
            rectangle=rectangle, 
            is_authenticated=is_authorized(feature=biocapsule, classifier_model=classifier_model))

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


# load classifier
def load_classifier(pkl_file_path):
    with open(pkl_file_path, "rb") as file:
        return pickle.load(file)


# authenticate the user
def is_authorized(feature, classifier_model)-> bool:
    feature = feature.reshape(1, -1)
    print(feature)
    print(feature.shape)
    pred = classifier_model.predict_proba(feature)
    print(pred)
    if pred[0][1] >= 0.5:
        return True
    else:
        return False


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


def get_rs_feature():
    """Return ArcFace features for 6 predetermined Reference Subjects
    (RSs) used in this experiment for reproducibility.

    """
    arcface = fr.ArcFace()

    rs_subjects = sorted(os.listdir("../../rs/"))
    rs_subjects = rs_subjects[4:]

    rs_features = np.zeros((6, 512))
    for s_id, subject in enumerate(rs_subjects):
        for image in os.listdir(f"../../rs/{subject}"):
            img = cv2.imread(f"../../rs/{subject}/{image}")
            feature = arcface.extract(img)
            rs_features[s_id] = feature

    return rs_features[0][:]


# gets the latency of the system: preprocessing, feat. ext., bc gen, and classification
def get_latency():
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # get first registered camera
    face = fr.ArcFace(0, "mtcnn") # load face model
    model = load_classifier(pkl_file_path="./edwin_classifier.pkl") # load regression model
    bc_gen = bc.BioCapsuleGenerator() # create bc_generator
    rs_feature = get_rs_feature() # load rs_feat


    
    # step 1: get image from camera
    start_time_frame_cap = time.time_ns()
    _, frame = video_capture.read()
    end_time_frame_cap = time.time_ns()

    # step 2: perform preprocessing and feat. ext.
    start_time_preprocessing = time.time_ns()    
    feat_vector = face.preprocess(frame)
    end_time_preprocessing = time.time_ns()

    start_time_feat_ext = time.time_ns()    
    feat_vector = face.extract(face_img=frame, align=False)
    end_time_feat_ext = time.time_ns()

    # step 3: generate biocapsule
    start_time_bc_gen = time.time_ns()
    biocapsule = bc_gen.biocapsule(user_feature=feat_vector, rs_feature=rs_feature)
    end_time_bc_gen = time.time_ns()

    # step 4: classify the biocapsule
    start_time_prediction = time.time_ns()
    biocapsule = biocapsule.reshape(1, -1)
    pred = model.predict(biocapsule)
    end_time_prediction = time.time_ns()

    # compute time difference in seconds
    frame_cap = comp_time_diff(end_time_frame_cap, start_time_frame_cap)
    preprocess = comp_time_diff(end_time_preprocessing, start_time_preprocessing)
    feat_ext = comp_time_diff(end_time_feat_ext, start_time_feat_ext)
    bc_gen_time = comp_time_diff(end_time_bc_gen, start_time_bc_gen)
    pred_time = comp_time_diff(end_time_prediction, start_time_prediction)

    total = frame_cap["time (sec)"] + preprocess["time (sec)"] + feat_ext["time (sec)"] + bc_gen_time["time (sec)"] + pred_time["time (sec)"]

    time_data = {
        "frame cap": frame_cap,
        "preprocess": preprocess,
        "feat_ext": feat_ext,
        "bc_gen_time": bc_gen_time,
        "pred": pred_time,
        "total (sec)": total
    }

    write_to_json(time_data, "time_test.json")


def comp_time_diff(end_time, start_time):
    times = ((end_time - start_time) / (10 ** 9), end_time - start_time)
    time_data = {
        "start time": start_time,
        "end time": end_time,
        "time (ns)": times[1],
        "time (sec)": times[0]
    } # end data dict
    return copy.deepcopy(time_data)


# writes a dictionary to a file as a json file
def write_to_json(data:dict, file_name:str)-> None:
    with open(file_name, "w") as file:
        file.write(json.dumps(data))


if __name__ == "__main__":
    main()
