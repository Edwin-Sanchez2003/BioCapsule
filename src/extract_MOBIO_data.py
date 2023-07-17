"""
    Generate Feature Vectors from a Dataset
    BioCapsule
    Edwin Sanchez

    Generates feature vectors from a set of images or videos.
    Stores the feature vectors in a JSON file with other metadata.

    Bottleneck: The bottle neck will be how many videos &
                models we can load into memory w/out crashing.
                Bottleneck also in the amount of available GPU 
                space to load models.

    PLEASE READ:
    - To run, make sure to set the PRESETS!!!!
    - this file MUST be ran from inside the BioCapsule directory! (references other files)
"""

# imports
import os
import json
import copy

import numpy as np
import cv2

# import module to do img processing
import face as fr

# set these to match the folder we're currently extracting from
class PRESETS:
    GPU = 0 # -1 for CPU, [0,n] for GPU(s)
    LOCATION = "but" # location where video was taken (MOBIO)
    PHASE = 1 # the phase: 1 or 2
    DEVICE = "laptop" # what device was used
    # DEVICE = "mobile/phone"
    USE_ARCFACE = True # whether to use ArcFace or FaceNet
    
    # set directories to the paths we want to extract from 
    input_dir = "../MOBIO/but_laptop/" # extracting dir
    output_dir = "./MOBIO_extracted/but/" # save dir


# class to organize info about the MOBIO dataset
class MOBIO:
    # MOBIO dataset locations
    LOCATIONS = [
        "but",
        "idiap",
        "lia",
        "uman",
        "unis",
        "uoulu"
    ] # end locations

    VIDEO_TYPES = [
        ".mp4",
        ".mov"
    ]


def main():
    # load face model w/ preprocessing model
    model = None
    if PRESETS.USE_ARCFACE == True:
        model = fr.ArcFace(PRESETS.GPU) # hardcoded to ArcFace w/MTCNN!
    else:
        model = fr.FaceNet(PRESETS.GPU) # hardcoded to FaceNet w/MTCNN!

    # make sure output dir exists
    if os.path.exists(PRESETS.output_dir) == False:
        os.makedirs(PRESETS.output_dir)

    # walk through a directory containing info
    for (dir_path, dir_names, file_names) in os.walk(PRESETS.input_dir, topdown=True):
        # loop over files found in dir
        for file_name in file_names:
            # check if file is a video file
            if is_in_list(os.path.splitext(os.path.basename(file_name))[1], MOBIO.VIDEO_TYPES):
                # perform extraction
                extract_video(
                    output_dir=PRESETS.output_dir,
                    file_path=os.path.join(dir_path, file_name),
                    model=model
                ) # extract video function


# extract important details from the video & save to dict
def extract_video(output_dir:str, file_path:str, model):
    # load video
    video = cv2.VideoCapture(file_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Video fps: {fps}")

    # loop over frames
    keepGoing = True
    frame_num = 0
    feature_vectors = [] # list to store feature vectors
    while keepGoing:
        # get a frame from the video
        ret, frame = video.read()

        # exit if we hit the last frame
        if ret == False:
            keepGoing = False
            continue

        # convert image to the format we need
        # should I do this? input for models say input should be BGR, but this still works...
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # pass img through preprocessing
        img_pre_processed:np.ndarray = model.preprocess(face_img=frame)

        # pass img through feature extraction
        feature_vector:np.ndarray = model.extract(face_img=img_pre_processed, align=False)

        # the feature vector used for every frame of the video
        feature_vector_container = {
            "frame_num": frame_num, # the index of the frame in the video, starting from 0
            "time_stamp (milisec)": video.get(cv2.CAP_PROP_POS_MSEC), # time stamp from video at the frame in miliseconds
            "num_faces_detected": 1, # how many faces detected (hard coded) - default to 1 for now
            "preprocessing_tensor": img_pre_processed.tolist(), # img ndarray from MTCNN
            "feature_vector": feature_vector.tolist() # feature vector from feat ext model (ie. FaceNet or ArcFace)
        } # end example feature vector

        # copy feature vector container to array
        # deep copy to avoid ptr problems
        feature_vectors.append(copy.deepcopy(feature_vector_container))

        # increment frame number
        frame_num += 1

    # get parts of the path to add as metadata to json file
    base_name = os.path.basename(file_path)
    subject_ID = base_name[:4] # first 4 char of video file name
    session_ID = base_name[5:7] # next 2 char from video file name, after underscore
    gender = base_name[:1] # first character of the video file name
    recording_num = base_name[9:11] # char 10 & 11 - recording number

    # data dict format for output json files
    data = {
        "MOBIO": {
            "location": PRESETS.LOCATION,
            "phase": PRESETS.PHASE,
            "device": PRESETS.DEVICE,
            "subject_ID": subject_ID, # string (ex: 'f404')
            "session_ID": session_ID, # string from 1 - 12
            "gender": gender, # string (ex: 'f' for female, 'm' for male)
            "recording_num": recording_num, # which recording during the session  - number from 1 to 21
        }, # end MOBIO specific data for video
        "file_name": base_name, # original video file name
        "total_frames": frame_num, # frame count of video
        "video_fps": fps, # fps of video
        "pre_proc_model": "MTCNN", # model used for preprocessing - default MTCNN
        "feat_ext_model": "ArcFace", # model used for feature extraction - default ArcFace
        "feat_vect_length": 512, # length of the feature vectors - default 512
        "feature_vectors": feature_vectors # the feature vectors and other data pulled from each frame of the video
    } # end data dict

    # write data dict to file
    name = f"{os.path.splitext(base_name)[0]}.json"
    output_path = os.path.join(output_dir, name)
    write_to_json(file_path=output_path, data=data)


# tests data from a video to make sure its use-able
def test_data_json(file_path:str):
    data = load_json_file(file_path=file_path)
    feat_vect = data["feature_vectors"][0]["feature_vector"]
    feat_vect_np = np.array(feat_vect)
    print(feat_vect)
    print("")
    print(feat_vect_np.shape)
    print(len(feat_vect_np))


# writes a dictionary to a json file
def write_to_json(file_path:str, data:dict)-> None:
    with open(file_path, "w") as file:
        file.write(json.dumps(data))


# loads a json file into a python dictionary
def load_json_file(file_path:str)-> dict:
    with open(file_path, "r") as file:
        return json.load(file)
    

# check a value against a list. returns true if exists in the list
def is_in_list(item:str, data:"list[str]")-> bool:
    for data_item in data:
        if item.lower() == data_item.lower():
            return True
    return False


if __name__ == "__main__":
    main()
