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
import copy # used to copy data from pointers to new variables

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
    if config_data["is_file"] == False: # if false, then it is a directory of vids
        # the maximum number of processes to generate at a given time
        max_num_processes = config_data["max_num_processes"]
        
        # get a list videos from the directory
        files_in_dir:list[str] = os.listdir(config_data["input_path"])

        # get all of the video paths into params to be passed to processes
        params_list = []
        input = {
            "file_name": "",
            "data": copy.deepcopy(template_data)
        } # end template input
        for file_name in files_in_dir:
            if file_name.lower().endswith(tuple(config_data["file_extensions"])):
                new_input = copy.deepcopy(input)
                new_input["file_name"] = os.path.join(config_data["input_path"], file_name)
                params_list.append(new_input)
        
        # print the number of files to extract from

        print(f"Extracting {len(params_list)} files...")

        # generate processes to do feature extraction
        print(f"Creating {max_num_processes} processes...")
        with multiprocessing.Pool(max_num_processes) as p:
            p.map(process_feat_ext, params_list)
    else: # only a single video - only need to do proc_feat_ext once
        input = {
            "file_name": "",
            "data": copy.deepcopy(template_data)
        } # end template input
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
    data["pre_proc_model"] = config_data["pre_proc_model"]
    data["feat_ext_model"] = config_data["feat_ext_model"]


    # create folder for output files, if it doesn't exist already
    if os.path.exists(config_data["output_dir"]) == False:
        os.makedirs(config_data["output_dir"])

    return config_data, data


# do the work of one process
def process_feat_ext(input:dict)-> None:
    print(f"Beginning feature extraction on: {input['file_name']}")
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


# tests data from a video to make sure its use-able
def test_data_json():
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