"""
    Test Generated Data
    BioCapsule
    Edwin Sanchez

    Test the data generated from extract_MOBIO_data.py
    and add_to_MOBIO_data.py. Make sure the data makes
    sense.

    - randomly select samples to check
    - print out all metadata
    - check time intervals
    - display preprocessed images for visual check
    - double-check features vectors
"""

# imports
from extract_MOBIO_data import load_json_gz_file

import os
import random
import time

import face as fr
import numpy as np

from PIL import Image

NUM_FILES_TO_CHECK = 3
NUM_SAMPLES_PER_FILE = 3


def main():
    # get folder to randomly check data integrity
    base_dir = "./MOBIO_extracted/one_sec_intervals/"
    sub_dirs = [
        "but",
        "idiap",
        "lia",
        "uman",
        "unis",
        "uoulu"
    ] # end sub dirs

    # loop over location sub directories
    for sub_dir in sub_dirs:
        input_dir = os.path.join(base_dir, sub_dir)
        file_names = os.listdir(input_dir)
        num_files = len(file_names)

        # randomly select files to test
        for file_num in range(NUM_FILES_TO_CHECK):
            random.seed()
            i = random.randint(0, num_files-1)
            file_path = os.path.join(input_dir, file_names[i])
            run_tests(file_path=file_path)
            print(f"Finished file {file_num}")
            input("Press [Enter] to continue to next file.")


def run_tests(file_path:str):
    # load file 
    data = load_json_gz_file(file_path=file_path)

    # print out metadata for visual test
    print_meta_data(data=data)

    # view random frames data
    view_random_frames(frames_data=data["frame_data"])


# print out the meta data for the whole session
def print_meta_data(data:dict)->None:
    print(data["MOBIO"])
    print(data["file_names"])
    print(f"Num Files: {len(data['file_names'])}")
    print(f"Extraction rate (sec):{data['extraction_rate_sec']}")


# sample from frames
def view_random_frames(frames_data:dict):
    num_frames = len(frames_data)
    for frame_num in range(NUM_SAMPLES_PER_FILE):
        random.seed()
        i = random.randint(0, num_frames-1)
        # print out metadata from  random frame
        frame = frames_data[i]
        print(f"time_stamp: {frame['time_stamp (milisec)']}")
        print(f"recording_num: {frame['recording_num']}")
        print(f"frame_num: {frame['frame_num']}")
        print(f"video_name: {frame['video_name']}")
        print(f"fps: {frame['fps']}")
        print(f"num_faces_detected: {frame['num_faces_detected']}")
        
        # view preprocessed image
        img_tensor = np.array(frame["preprocessing_tensors"]["mtcnn"])
        view_preprocessed_image(img_tensor)
        
        # double-check feature vectors
        double_check_feature_vectors()
        print(f"Finished frame {frame_num}")
        input("Press [Enter] to continue to the next frame.")


# opens a window with the preprocessed image displayed
def view_preprocessed_image(pre_processed_tensor:list)-> None:
    pre_processed_tensor = np.uint8(pre_processed_tensor)
    pre_processed_tensor = np.rollaxis(pre_processed_tensor, 0, 3)
    img = Image.fromarray(pre_processed_tensor)
    img.show("preprocessed image")


# verifies that the feature vector generated is correct by using
# the original file
def double_check_feature_vectors()-> None:
    pass


if __name__ == "__main__":
    main()
