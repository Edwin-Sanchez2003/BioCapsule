"""
    Generate Feature Vectors from a Dataset
    BioCapsule
    Edwin Sanchez

    Generates feature vectors from a set of images or videos.
    Stores the feature vectors in a JSON file with other metadata.

    Bottleneck: GPU Size/Memory Constraints

    PLEASE READ:
    - To run, make sure to set the PRESETS!!!!
    - this file MUST be ran from inside the BioCapsule directory! (references other files)
"""

# imports
import os
import json
import copy
import gzip

import numpy as np
import cv2

from multiprocessing import Pool
import time

# import module to do img processing
import face as fr

path_to_mobio = "../MOBIO/"
extraction_dest_dir = "./MOBIO_extracted/one_sec_intervals/"

# list of presets to run through. set them to extract from different folders in MOBIO.
presets_list = [
    # [ GPU, LOCATION, PHASE, DEVICE, USE_ARCFACE, input_dir, output_dir]
    # three per location, for 'phase 1 laptop', 'phase 1 mobile', and 'phase 2 mobile' folders

    # but location
    [ 0, "but", 1, "laptop", True, f"{path_to_mobio}but_laptop/", f"{extraction_dest_dir}but/"],
    [ 0, "but", 1, "mobile", True, f"{path_to_mobio}but_phase1/", f"{extraction_dest_dir}but/"],
    [ 0, "but", 2, "mobile", True, f"{path_to_mobio}but_phase2/", f"{extraction_dest_dir}but/"],

    # idiap location
    [ 0, "idiap", 1, "laptop", True, f"{path_to_mobio}idiap_laptop/", f"{extraction_dest_dir}idiap/"],
    [ 0, "idiap", 1, "mobile", True, f"{path_to_mobio}idiap_phase1/", f"{extraction_dest_dir}idiap/"],
    [ 0, "idiap", 2, "mobile", True, f"{path_to_mobio}idiap_phase2/", f"{extraction_dest_dir}idiap/"],

    # lia location
    [ 0, "lia", 1, "laptop", True, f"{path_to_mobio}lia_laptop/", f"{extraction_dest_dir}lia/"],
    [ 0, "lia", 1, "mobile", True, f"{path_to_mobio}lia_phase1/", f"{extraction_dest_dir}lia/"],
    [ 0, "lia", 2, "mobile", True, f"{path_to_mobio}lia_phase2/", f"{extraction_dest_dir}lia/"],

    # uman location
    [ 0, "uman", 1, "laptop", True, f"{path_to_mobio}uman_laptop/", f"{extraction_dest_dir}uman/"],
    [ 0, "uman", 1, "mobile", True, f"{path_to_mobio}uman_phase1/", f"{extraction_dest_dir}uman/"],
    [ 0, "uman", 2, "mobile", True, f"{path_to_mobio}uman_phase2/", f"{extraction_dest_dir}uman/"],

    # unis location
    [ 0, "unis", 1, "laptop", True, f"{path_to_mobio}unis_laptop/", f"{extraction_dest_dir}unis/"],
    [ 0, "unis", 1, "mobile", True, f"{path_to_mobio}unis_phase1/", f"{extraction_dest_dir}unis/"],
    [ 0, "unis", 2, "mobile", True, f"{path_to_mobio}unis_phase2/", f"{extraction_dest_dir}unis/"],

    # uoulu location
    [ 0, "uoulu", 1, "laptop", True, f"{path_to_mobio}uoulu_laptop/", f"{extraction_dest_dir}uoulu/"],
    [ 0, "uoulu", 1, "mobile", True, f"{path_to_mobio}uoulu_phase1/", f"{extraction_dest_dir}uoulu/"],
    [ 0, "uoulu", 2, "mobile", True, f"{path_to_mobio}uoulu_phase2/", f"{extraction_dest_dir}uoulu/"]
] # end presets list


# set these to match the folder we're currently extracting from
class PRESETS:
    WAIT_SEC_FOR_EXT = 1.00 # wait every 1 second to perform extraction
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

    # types of videos to extract from. MOBIO only has .mov and .mp4 files.
    VIDEO_TYPES = [
        ".mp4",
        ".mov"
    ] # end allowed video types


def main():
    # loop over the presets list
    for preset in presets_list:
        # set PRESETS to match the given preset
        PRESETS.GPU = preset[0]
        PRESETS.LOCATION = preset[1]
        PRESETS.PHASE = preset[2]
        PRESETS.DEVICE = preset[3]
        PRESETS.USE_ARCFACE = preset[4]
        PRESETS.input_dir = preset[5]
        PRESETS.output_dir = preset[6]

        print("Current Preset Settings:")
        print(f"GPU:{PRESETS.GPU} LOCATION:{PRESETS.LOCATION} PHASE:{PRESETS.PHASE} DEVICE:{PRESETS.DEVICE}")
        print(f"USE_ARCFACE:{PRESETS.USE_ARCFACE} input_dir:{PRESETS.input_dir} output_dir:{PRESETS.output_dir}")

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
            # loop over files found in dir, adding files to the session list
            session_file_names = [] # videos all in the same folder (MOBIO)
            for file_name in file_names:
                # check if file is a video file
                if is_in_list(os.path.splitext(os.path.basename(file_name))[1], MOBIO.VIDEO_TYPES):
                    # avoid bad files
                    if file_name[:2] != "._":
                        f_name = os.path.join(dir_path, file_name)
                        session_file_names.append(f_name)
                    else: # remove bad files generated by Mac OS
                        f_name = os.path.join(dir_path, file_name)
                        os.remove(f_name)
                        print(f"Removed: {f_name}")
            
            
            # check if there are any videos to extract from
            if len(session_file_names) > 0:
                # sort the files to be in the right order
                session_file_names = put_files_in_order(session_file_names)
                # pass the videos found in the session folder
                # to the extraction function
                print(f"Extracting files in dir {os.path.dirname(session_file_names[0])} from {PRESETS.input_dir} ({PRESETS.LOCATION})")
                extract_video(
                    output_dir=PRESETS.output_dir,
                    file_paths=session_file_names,
                    model=model
                ) # end extract video function


# puts the files in order, from 1 to 21
def put_files_in_order(old_list:"list[str]")-> "list[str]":
    # loop over file names and get their corresponding recording_num
    tmp_list = []
    for file_name in old_list:
        base_name = os.path.basename(file_name)
        recording_num = int(base_name[9:11])
        tmp_list.append((file_name, recording_num))

    # sort 
    sorted_list:"list[str]" = []
    keepGoing = True
    sentry = 1
    while keepGoing:
        for file_name, recording_num in tmp_list:
            if recording_num == sentry:
                sorted_list.append(file_name)
                sentry += 1
        if len(sorted_list) == len(old_list):
            keepGoing = False
    return sorted_list


# extract important details from the video & save to dict
def extract_video(output_dir:str, file_paths:"list[str]", model):
    # loop over every video given (from the session)
    current_time = float(0.0)
    frame_data = [] # list to store data from each frame extracted
    for file_path in file_paths:
        tic = time.perf_counter()
        recording_num = os.path.basename(file_path)[9:11]
        # load video
        video = cv2.VideoCapture(file_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # loop over frames
        keepGoing = True
        frame_num = 0
        print(f"Processing video. Recording_num: {recording_num}")
        while keepGoing:
            # get a frame from the video
            ret, frame = video.read()

            # exit if we hit the end of the video
            if ret == False:
                keepGoing = False
                continue
            
            # increment time
            current_time += float(1/fps)
            if current_time >= PRESETS.WAIT_SEC_FOR_EXT: #check if we hit the 10 second mark, then process the frame
                current_time = 0.0

                # pass img through preprocessing
                pre_proc_data:"tuple[np.ndarray, int]" = model.preprocess(face_img=frame)
                img_pre_processed = pre_proc_data[0]
                num_faces = pre_proc_data[1]

                # pass img through feature extraction
                feature_vector:np.ndarray = model.extract(face_img=img_pre_processed, align=False)

                # the feature vector used for every frame of the video
                feature_vector_container = {
                    "frame_num": frame_num, # the index of the frame in the video, starting from 0
                    "time_stamp (milisec)": video.get(cv2.CAP_PROP_POS_MSEC), # time stamp from video at the frame in miliseconds
                    "recording_num": recording_num, # the video that this came from (1-21)
                    "frame_num": frame_num, # the frame that this came from in the video (0 for first frame)
                    "video_name": file_path, # name of the video this feature came from
                    "fps": fps, # video's fps
                    "num_faces_detected": num_faces, # how many faces detected
                    "preprocessing_tensors": {
                        "mtcnn": img_pre_processed.tolist() # img ndarray from MTCNN
                    }, # end preprocessing tensors
                    "feature_vectors": {
                        "arcface": feature_vector.tolist() # feature vector from feat ext model (ie. FaceNet or ArcFace)
                    }, # end feature vectors
                } # end example feature vector

                # copy feature vector container to array
                # deep copy to avoid ptr problems
                frame_data.append(copy.deepcopy(feature_vector_container))

            # increment frame number
            frame_num += 1
        # end while loop for one video

        toc = time.perf_counter()
        print(f"Time to extract the video: {toc - tic:0.4f} seconds")
    # end for loop over videos

    tic = time.perf_counter()
    # get parts of the path to add as metadata to json file
    base_name = os.path.basename(file_paths[0])
    subject_ID = base_name[:4] # first 4 char of video file name
    session_ID = base_name[5:7] # next 2 char from video file name, after underscore
    gender = base_name[:1] # first character of the video file name

    # data dict format for output json files
    data = {
        "MOBIO": {
            "location": PRESETS.LOCATION,
            "phase": PRESETS.PHASE,
            "device": PRESETS.DEVICE,
            "subject_ID": subject_ID, # string (ex: 'f404')
            "session_ID": session_ID, # string from 1 - 12
            "gender": gender # string (ex: 'f' for female, 'm' for male)
        }, # end MOBIO specific data for video
        "file_names": file_paths, # original video file names (in order)
        "feat_vect_length": 512, # length of the feature vectors - default 512
        "frame_data": frame_data, # the feature vectors and other data pulled from each frame of the video
        "extraction_rate_sec": PRESETS.WAIT_SEC_FOR_EXT # how many seconds to wait in each video before extracting
    } # end data dict

    # create name of new file
    name = f"{PRESETS.LOCATION}_{PRESETS.DEVICE}_{PRESETS.PHASE}_{subject_ID}_{session_ID}.json.gz"
    output_path = os.path.join(output_dir, name)

    # compresss & write data to a file
    write_to_json_gz(file_path=output_path, data=data)
    toc = time.perf_counter()
    print(f"Time to write to file: {toc - tic:0.4f} seconds")


# tests data from a video to make sure its use-able
def test_data_json_gz(file_path:str):
    data = load_json_gz_file(file_path=file_path)
    print(data["MOBIO"])
    print(data["frame_data"][0])
    feat_vect = data["frame_data"][0]["feature_vectors"]["arcface"]
    feat_vect_np = np.array(feat_vect)
    print(feat_vect)
    print("")
    print(feat_vect_np.shape)
    print(len(feat_vect_np))
    print("")
    for features in data["feature_vectors"]:
        print(features["frame_num"])


# jsonify data dictionary, then gzip to make it more compressed
# after compression, write to the file
def write_to_json_gz(file_path:str, data:dict, comp_lvl:int=6)-> None:
    json_data = json.dumps(data)
    encoded_data = json_data.encode('utf-8')
    with open(file_path, "wb") as file:
        file.write(gzip.compress(data=encoded_data, compresslevel=comp_lvl))
    print("Finished writing file!")


# wrapper for write_to_json_gz
def compress_wrapper(file_path:str)-> None:
    data = load_json_file(file_path=file_path)
    new_name = f"{file_path}.gz"
    write_to_json_gz(file_path=new_name, data=data)


# creates a bunch of processes to zip up the files
def compress_json_files():
    # get files to compress
    base_path = "./MOBIO_extracted/but/"
    files_to_compress = os.listdir(base_path)
    file_paths = []
    for file_name in files_to_compress:
        file_paths.append(os.path.join(base_path, file_name))

    # generate processes to handle compression
    with Pool() as p:
        p.map(compress_wrapper, file_paths)


# writes a dictionary to a json file
def write_to_json(file_path:str, data:dict)-> None:
    with open(file_path, "w") as file:
        file.write(json.dumps(data))


# loads a json file into a python dictionary
def load_json_file(file_path:str)-> dict:
    with open(file_path, "r") as file:
        return json.load(file)
    

# loads a json file into a python dictionary
def load_json_gz_file(file_path:str)-> dict:
    with gzip.open(file_path, "rb") as file:
        return json.load(file)
    

# check a value against a list. returns true if exists in the list
def is_in_list(item:str, data:"list[str]")-> bool:
    for data_item in data:
        if item.lower() == data_item.lower():
            return True
    return False


if __name__ == "__main__":
    main()
