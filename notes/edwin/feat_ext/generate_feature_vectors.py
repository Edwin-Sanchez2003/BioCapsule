"""
    Generate Feature Vectors from a Dataset
    BioCapsule
    Edwin Sanchez

    Generates feature vectors from a set of images or videos.
    Stores the feature vectors in a JSON file with other metadata.

    Bottleneck: The bottle neck will be how many videos &
                models we can load into memory w/out crashing.
"""

# Imports
import os # used for opening & reading files.
import json # used for storing the feature vector files.

import multiprocessing # for speeding up processing.

import argparse # used for passing args to this file.
parser = argparse.ArgumentParser(
    prog="Generate Feature Vectors",
    description="This file is designed to generate feature vectors from videos in the MOBIO dataset and store them in JSON files with their corresponding metadata."
) # end parser init

# add arguments for file
parser.add_argument("-c", "--config_file_path", help="The path to the configuration JSON file.")

# parse for arguments
args = parser.parse_args()

print("I ran")
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

    # load the config file detailing parameters
    config_data, template_data = load_configurations(config_file_path=args.config_file_path)
    
    # decide how many processes to generate
    max_num_processes = 1
    if config_data["is_file"] == False:
        max_num_processes = config_data["max_num_processes"]

    



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
    data["pre_proc_model"] = config_data["pre_proc_model"]
    data["feat_ext_model"] = config_data["feat_ext_model"]
    data["pre_proc_model"] = config_data["pre_proc_model"]
    data["pre_proc_model"] = config_data["pre_proc_model"]

    # create folder for output files, if it doesn't exist already
    if os.path.exists(config_data["output_dir"]) == False:
        os.makedirs(config_data["output_dir"])

    return config_data, data


# creates processes to speed up the feature extraction process.
# CPU-Bound. Bottleneck: how many models, videos, and dictionaries
# of data we can have in memory during execution without crashing.
def generate_processes(num_processes):
    pass


# tests data from a video to make sure its useable
def test_data_json():
    pass


# gets a feature vector from a single image
def get_feat_vect_from_img(image, pre_proc_model, feat_ext_model)-> "list[float]":
    """
    This function extracts features from a frame of the video.

    Inputs
    ----------
    - image: an ndarray representing an image to perform 
    preprocessing and extraction on.
    - pre_proc_model: the model to use for preprocessing.
    - feat_ext_model: the model to use for feature extraction.
    
    Outputs
    ----------
    - feature_vector: the feature vector generated from the 
    feature extraction model.
    """
    pass


# writes a dictionary to a json file
def write_to_json(file_path:str, data:dict)-> None:
    with open(file_path, "w") as file:
        file.write(json.dumps(data))


# loads a json file into a python dictionary
def load_json_file(file_path:str)-> dict:
    with open(file_path, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    main()