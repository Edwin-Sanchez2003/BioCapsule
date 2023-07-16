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
"""

# Imports
import os # used for opening & reading files.
import json # used for storing the feature vector files.
import copy # used to copy data from pointers to new variables
import numpy as np # for n dimensional arrays

import multiprocessing # for speeding up processing.

import face as fr # relies on face.py being in the src folder

import cv2 # for video & image processing

import argparse # used for passing args to this file.
parser = argparse.ArgumentParser(
    prog="Generate Feature Vectors",
    description="This file is designed to generate feature vectors from videos in the MOBIO dataset and store them in JSON files with their corresponding metadata."
) # end parser init

# add arguments for file
parser.add_argument("-c", "--config_file_path", help="The path to the configuration JSON file.")

# parse for arguments
args = parser.parse_args()


# list of allowed preprocessors
pre_processors = [
    "MTCNN"
] # end preprocessors

# list of allowed feature extractors
feature_extractors = {
    "FaceNet": fr.FaceNet,
    "ArcFace": fr.ArcFace
} # end feature extractors


# runs when this file is called directly
def main():
    """
    performs extraction on one video.
    stores all data collected into a single json file.
    used as the job for a single process.

        load config file and parameters
        decide how many processes to create
            load preprocessing model
            load in feat ext model
            load a video
            bust into frames
                pass each frame to models & get feature vector
                store data into dict
            store meta data into dict
            store dict into json file on disk
    """
    #test_data_json("./test/f401_1_laptop_file_0.json")
    #return

    # load the config file detailing parameters
    config_data, template_data = load_configurations(config_file_path=args.config_file_path)
    input = {
        "file_name": "",
        "index": 0,
        "data": copy.deepcopy(template_data),
        "output_dir": copy.deepcopy(config_data["output_dir"])
    } # end template input

    # decide how many processes to generate
    max_num_processes = 1
    if config_data["is_file"] == False: # if false, then it is a directory of vids
        # the maximum number of processes to generate at a given time
        max_num_processes = config_data["max_num_processes"]
        
        # get a list videos from the directory
        files_in_dir = os.listdir(config_data["input_path"])

        # get all of the video paths into params to be passed to processes
        params_list = []
        for i, file_name in enumerate(files_in_dir):
            if file_name.lower().endswith(tuple(config_data["file_extensions"])):
                new_input = copy.deepcopy(input)
                new_input["index"] = i
                new_input["file_name"] = os.path.join(config_data["input_path"], file_name)
                params_list.append(new_input)
        
        # print the number of files to extract from
        print(f"Extracting {len(params_list)} file(s)...")

        # generate processes to do feature extraction
        print(f"Creating {max_num_processes} process(es)...")
        with multiprocessing.Pool(max_num_processes) as p:
            p.map(process_feat_ext, params_list)
    else: # only a single video - only need to do proc_feat_ext once
        input["file_name"] = config_data["input_path"]
        process_feat_ext(input=input)


# loads the config file params into an easy to read format
# returns the config dict and a starting point for the output JSON file
def load_configurations(config_file_path:str)-> "tuple[dict,dict]":
    # open JSON into a python dict
    config_data = load_json_file(file_path=config_file_path)
    
    # create starting dict formatted to store data
    data = load_json_file(config_data["json_template_path"])

    # check if we should ignore the pre-sets
    if config_data["ignore_pre_sets"]:
        data["location"] = None
        data["subject_ID"] = None
        data["session_ID"] = None
        data["phase"] = None
        data["device"] = None
    else:
        data["location"] = config_data["pre_sets"]["location"]
        data["subject_ID"] = config_data["pre_sets"]["subject_ID"]
        data["session_ID"] = config_data["pre_sets"]["session_ID"]
        data["phase"] = config_data["pre_sets"]["phase"]
        data["device"] = config_data["pre_sets"]["device"]

    # check if input path is a directory or a single video
    # store for later decisions
    config_data["is_file"] = os.path.isfile(config_data["input_path"])

    # set other meta data details
    if is_in_list(config_data["pre_proc_model"], pre_processors) == False:
        raise Exception(
            f"preprocessing model not available. Available models for preprocessing: {pre_processors}")
    if is_in_list(config_data["feat_ext_model"], feature_extractors) == False:
        raise Exception(
            f"feature extraction model not available. Available models for feature extraction: {feature_extractors}")

    data["pre_proc_model"] = config_data["pre_proc_model"].lower()
    data["feat_ext_model"] = config_data["feat_ext_model"].lower()

    # create folder for output files, if it doesn't exist already
    if os.path.exists(config_data["output_dir"]) == False:
        os.makedirs(config_data["output_dir"])

    return config_data, data


# do the work of one process
def process_feat_ext(input:dict)-> None:
    """
    load preprocessing model
    load in feat ext model
    load video
    bust into frames
        pass each frame to models & get feature vector
        store data into dict
    store meta data into dict
    store dict into json file on disk
    """
    print(f"Beginning feature extraction on: {input['file_name']}")

    data:dict = input["data"]

    # load face model w/ preprocessing model
    model = fr.ArcFace(0) # hardcoded to ArcFace w/MTCNN!
    

    # load video
    video = cv2.VideoCapture(input["file_name"])
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Video fps: {fps}")

    # create default feature container to copy from
    feature_container = copy.deepcopy(data["feature_vectors"][0])

    # clear out feature vector default in data dict
    data["feature_vectors"] = []
    
    # loop over frames
    keepGoing = True
    frame_num = 0
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
        img_pre_processed = model.preprocess(face_img=frame)

        # pass img through feature extraction
        feature_vector = model.extract(face_img=img_pre_processed, align=False)

        # store in array
        feature_container["frame_num"] = frame_num
        feature_container["time_stamp"] = (frame_num / fps)
        feature_container["num_faces_detected"] = None # not sure how to measure right now...
        feature_container["preprocessing_tensor"] = img_pre_processed.tolist()
        feature_container["feature_vector"] = feature_vector.tolist()

        """
        print("Img Pre Processed:")
        print(img_pre_processed)
        print("Feature Vector:")
        print(feature_vector)
        print("to_list version")
        print(feature_vector.tolist())
        """
        data["feature_vectors"].append(copy.deepcopy(feature_container))

        # increment frame num
        frame_num += 1
    
    # once finished, add in last details about video to dict
    data["total_frames"] = frame_num
    data["video_fps"] = fps
    data["file_name"] = os.path.basename(input["file_name"])

    # get details to name new file
    sub_ID = data["subject_ID"]
    ses_ID = data["session_ID"]
    device = data["device"]
    i = input["index"]

    # check if user chose to ignore pre sets
    if sub_ID == None:
        print("User has chosen to ignore pre sets.")
        print("Storing output as matching file name of original video")
        file_to_create = os.path.join(input["output_dir"], os.path.splitext(os.path.basename(input["file_name"]))[0])
    else:
        file_to_create = os.path.join(input["output_dir"], f"{sub_ID}_{ses_ID}_{device}_file_{i}")
    
    # write json file to disk
    write_to_json(file_path=f"{file_to_create}.json", data=data)


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