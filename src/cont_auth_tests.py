"""
    Continuous Authentication Testing
    BioCapsule
    
    Description:
    This file contains tests for Continuous Authentication,
    with a primary goal of testing the system with and
    without biocapsule involved.

    TODO:
    - k-fold across sessions (13 fold validation)
    - set time parameter (1 second intervals)
    - window size for the authentication decision (how many scores to average)
      from previous decisions
        - what averaging algorithm to use
    - with and w/out biocapsule
        - graph feat. vectors for an individual with and without biocapsule
    - CA-MMOC Classifiers vs. F-MMOC Classifiers
    - quantized vs. non-quantized
    - MOBIO and YouTube Faces
    - FaceNet vs. ArcFace
    - Measure speed at some point?
    - Single Platform vs. Multi Platform

    MOBIO Notes:
    - OMIT f210 from unis!!! Only 1 session exists for this individual
    - the rest have all 13 sessions (12 mobile, 1 laptop)
"""


# imports
import os

import biocapsule as bc

from test_enums import *
from data_tools import *


MOBIO_LOCATIONS = [
    "but/",
    "idiap/",
    "lia/",
    "uman/",
    "unis/",
    "uoulu/"
] # end mobio locations


def main():
    """
    Set Params:
    - bc or no bc
    - time for sampling (only for test???)
    - window size
        - averaging method
        - maybe use the averaging method introduced
          in operating systems???
    - single vs multi platform
        - multi platform can't do k-fold val
    - FaceNet vs ArcFace
    - quantized vs non-quantized
    - dataset to use MOBIO vs. YouTubeFaces
    - classifier type (for authentication decision)
    - play w/feature vector size???

    # pick a location. iterate over the location
        # pick a participant. get all of their sessions
        # tracked as the compressed files
        
        # get the number of sessions they have
        # separate out laptop version
        # that will be our k in k-fold validation

        # for k, select one session for training
        # and the rest for testing
            # train a classifier on the train
            # data. test on the test data.
            
            # store output in an easy to aggregate manner
    """

    # Params #
    base_dir = "../MOBIO_extracted/one_sec_intervals/"

    # get all of the MOBIO locations as input directories
    input_dirs = []
    for loc_path in MOBIO_LOCATIONS:
        input_dirs.append(os.path.join(base_dir, loc_path))

    # loop over input dirs
    for input_dir in input_dirs:
        # loop over participants
        for part_id in os.listdir(input_dir):
            # run a test generate a file containing
            # data on the test. include test params
            run_test()


# run a test for a single participant,
# given the input params
def run_test(participant_dir:str,
             use_bc:bool,
             platform:Enum,
             model_type:Enum,
             use_quantization:bool,
             dataset_name:Enum,
             classifier_type:Enum,
             window_size:int,
             wait_time:int,
             use_k_fold:bool)-> None:
    
    # the id of the subject (participant id)
    subj_id = os.path.basename(participant_dir)

    # the data to be saved as a json file
    data = {
        "dataset": dataset_name,
        "origin_path": participant_dir,
        "subject_id": subj_id,
        "use_bc": use_bc,
        "platform": platform,
        "pre_proc_model": "mtcnn",
        "feat_ext_model": model_type,
        "use_quantization": use_quantization,
        "classifier": classifier_type,
        window_size: window_size,
        wait_time: wait_time,
        "results": []
    } # end test dict

    # check platform




# accumulate results for a single location
# using generated test files from individual participants
def aggregate_results():
    pass


if __name__ == "__main__":
    main()
