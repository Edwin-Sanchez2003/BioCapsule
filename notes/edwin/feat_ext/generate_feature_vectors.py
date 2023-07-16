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
import json # used for storing the feature vector files.
import multiprocessing

def main():
    # load the config file detailing parameters
    # for feature extraction
    pass


# creates processes to speed up the feature extraction process.
# CPU-Bound. Bottleneck: how many models, videos, and dictionaries
# of data we can have in memory during execution without crashing.
def generate_processes(num_processes):
    pass


# performs extraction on one video.
# stores all data collected into a single json file.
# used as the job for a single process.
def extract_from_video():
    pass
    # load preprocessing model
    # load in feat ext model
    # load a video
    # bust into frames
        # pass each frame to models & get feature vector
        # store data into dict
    # store meta data into dict
    # store dict into json file on disk


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
