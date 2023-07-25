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

    Output Data:
    - organize into a folder for each test,
      for each location, and for each subject 
      in each location
    - should generate a json file that contains
      aggregated results at the root of each
      individual test
    - document if any subjects were skipped during testing

    Questions:
    - how many users should I use for negative samples?
    - all of the other users? Use the same session for
    all other users for negative samples for training,
    then use all other sessions from negative users for
    testing
    - or should I only use negative samples from the subject's
      location? Then use the rest of the locations as negative
      samples?
    Thoughts:
    - If I use all of the other users as negative samples during 
      training, then that would be poluting the test results!
    Decision:
    - Load single session of user for positive samples
        - the rest of the sessions are test sessions
    - Load half of all other users for negative samples
        - only use 1 session for training negative samples,
        - the rest for testing negative samples
    - Test on ALL other people, but organize results to
      distinguish between users used for training and users
      used for testing

    Dataset wide statistics:
    - how far apart are the feature embeddings of all of the samples?
    - what is the avg. euclidean dist betw. feature embeddings of users?
    - std. dev? (always have error bars)
    - how about after applying bc scheme? will give great insights to 
      how bc changes the representation of the feature embeddings

    MOBIO Notes:
    - OMIT f210 from unis!!! Only 1 session exists for this individual
    - the rest have all 13 sessions (12 mobile, 1 laptop)
"""


# imports
import os
import copy
from typing import Union
import random

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

 # Params #
BASE_DIR = "./MOBIO_extracted/one_sec_intervals/"


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

    # get all of the MOBIO locations as input directories
    input_dirs = []
    for loc_path in MOBIO_LOCATIONS:
        input_dirs.append(os.path.join(BASE_DIR, loc_path))

    # loop over input dirs
    for input_dir in input_dirs:
        # loop over participants
        for part_id in os.listdir(input_dir):
            # run a test generate a file containing
            # data on the test. include test params
            out_data = run_test(
                participant_dir=os.path.join(input_dir, part_id),
                location=os.path.basename(input_dir),
                use_bc=False,
                platform=Platform.MULTI,
                model_type=Model_Type.ARCFACE,
                use_quantization=False,
                dataset_name=Dataset.MOBIO,
                classifier_type=Classifier.LOGISTIC_REGRESSION,
                window_size=1,
                wait_time=10,
                use_k_fold=False
            ) # end run_test function

            ## aggregate results here from out_data


# run a test for a single participant,
# given the input params
def run_test(participant_dir:str,
             location:str,
             use_bc:bool,
             platform:Enum,
             model_type:Enum,
             use_quantization:bool,
             dataset_name:Enum,
             classifier_type:Enum,
             window_size:int,
             wait_time:int,
             use_k_fold:bool)-> dict:
    
    # the id of the subject (participant id)
    subj_id = os.path.basename(participant_dir)

    # the data to be saved as a json file
    out_subj_data = {
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

    # get list of sessions collected during
    # data collection stage
    session_file_names = os.listdir(participant_dir)
    session_file_names.sort()
    print(f"Num of sessions: {len(session_file_names)}")
    print(session_file_names)

    # setup the dict used for each train/test split
    results = {
        "train_session": "",
        "num_train_sessions": 1,
        "test_sessions": [],
        "num_test_sessions": 0,
        "num_train_samples": 0,
        "num_test_samples": 0,
        "num_pos_train_samples": 0,
        "num_neg_train_samples": 0,
        "num_pos_test_samples": 0,
        "num_neg_test_samples": 0,
        "num_users_neg_train": 0,
        "num_users_neg_test": 0,
        "test_train_ratio": 0, # num_test_samples divided by the num_train_samples
    } # end results dict

    # if multi-platform, then only perform a single test
    # using laptop for train, the rest for test
    training_files = []
    if platform == Platform.MULTI:
        file_name = get_file_with_subcomponent(session_file_names, "laptop")
        if file_name != None:
            file_path = os.path.join(participant_dir, file_name)
            training_files.append(file_path)
        else:
            pass
            # need to make note that
            # subject did not have a laptop version
            # exclude from results
    else:
        # check if use_k_fold is set to true
        if use_k_fold:
            for file_name in session_file_names:
                training_files.append(
                    os.path.join(participant_dir, file_name)
                ) # end append to training_files
        else:
            training_files = session_file_names
    
    # loop over training_files
    # for multi-platform & non-k-fold this is only 1 file,
    # but for the rest, this gives each session a chance
    # as the train data (12-fold validation)
    for training_path in training_files:
        # get the list for the test paths
        test_paths = get_test_file_paths(training_path, session_file_names)
        # positive samples for training & testing
        train_pos_samples, test_pos_samples = get_pos_samples(
            training_path=training_path, 
            test_paths=test_paths,
            t_inverval=wait_time,
            model_type=model_type
        ) # end get pos samples from subject

        # load samples that are going to be our
        # negative data during training.
        # load samples from all other locations
        train_neg_samples, test_neg_samples = get_neg_samples(
            subj_id=subj_id,
            subj_loc=location,
            platform=platform
        ) # end get_neg_samples

        # combine train pos & train neg samples
        # create labels for binary classification
        # combine test pos & test neg samples
        # create labels for binary classification

        # apply biocapsule, if applicable

        # train classifier for this subject

        # run classifier on test data
        # generate performance metrics
        # store data into file


# get pos samples, in train/test split
def get_pos_samples(training_path:str, 
                    test_paths:"list[str]",
                    t_inverval:int,
                    model_type:str,
                    )-> "tuple[list, list]":
    # get training positive samples
    # load data file from training_path
    data = load_json_gz_file(file_path=training_path)
    train_pos_samples = get_feature_data(data=data,
                                         t_interval=t_inverval,
                                         model_type=model_type)
    test_pos_samples = []
    for test_path in test_paths:
        data = load_json_gz_file(file_path=test_path)
        test_pos_samples.extend(
            get_feature_data(data=data, 
                             t_interval=t_inverval, 
                             model_type=model_type)
        ) # end extend test_pos_samples
    # return a tuple containing the train & test samples
    return (train_pos_samples, test_pos_samples)


# get negative samples for both training & testing
def get_neg_samples(subj_id:str,
                    subj_loc:str,
                    t_interval:int,
                    model_type:str,
                    platform:str)-> "tuple[list, list]":
    # get a list of all other subject's directories
    other_subj_paths = []
    # loop over folders for each location
    for loc in MOBIO_LOCATIONS:
        loc_dir = os.path.join(BASE_DIR, loc)
        # get a list of dirs in a location (correspond to subjects)
        subj_dirs = os.listdir(loc_dir)
        subj_dirs = add_back_loc_to_path(loc=loc, subj_dirs=subj_dirs)
        # remove the training subject if in this location
        if loc == subj_loc:
            subj_dirs = remove_training_subj(
                subj_id=subj_id, 
                subj_dirs=subj_dirs
            ) # end remove_training_subj call
        other_subj_paths.extend(subj_dirs)
    
    # shuffle subject dirs, split into
    # complete holdout & semi-holdout groups
    rand_gen = random.Random(x=42)
    rand_gen.shuffle(other_subj_paths)
    # split as close as possible to 50/50
    index_to_split_at = len(other_subj_paths) // 2
    complete_holdouts = other_subj_paths[:index_to_split_at]
    semi_holdouts = other_subj_paths[index_to_split_at:]

    # NOTE: MAKE SURE TO EXCLUDE/INCLUDE LAPTOP
    #       DEPENDING ON PLATFORM PARAM!!!
    train_neg_samples = []
    test_neg_samples = []
    # for complete holdouts, take all sessions (use platform param)
    # and put into test_neg_samples
    comp_holdout_paths, _ = get_subject_file_paths_from_dirs(
        subj_dirs=complete_holdouts,
        semi_holdout=False
    ) # end get_subject_file_paths_from_dirs
    for path in comp_holdout_paths:
        data = load_json_gz_file(file_path=path)
        test_neg_samples.extend(
            get_feature_data(
                data=data,
                t_interval=t_interval,
                model_type=model_type
            ) # end get_feature_data
        ) # end extend test_neg_samples

    # for semi, split one sess for train_neg_samples
    # the rest go into test_neg_samples
    test_paths, train_paths = get_subject_file_paths_from_dirs(
        subj_dirs=semi_holdouts,
        semi_holdout=True
    ) # end get_subject_file_paths_from_dirs
    for path in test_paths:
        data = load_json_gz_file(file_path=path)
        test_neg_samples.extend( # add to test
            get_feature_data(
                data=data,
                t_interval=t_interval,
                model_type=model_type
            ) # end get_feature_data
        ) # end extend test_neg_samples
    for path in train_paths:
        data = load_json_gz_file(file_path=path)
        train_neg_samples.extend( # add to train
            get_feature_data(
                data=data,
                t_interval=t_interval,
                model_type=model_type
            ) # end get_feature_data
        ) # end extend test_neg_samples
    return (train_neg_samples, test_neg_samples)


# get a list of data files to load from a list of dirs to search through
def get_subject_file_paths_from_dirs(
        subj_dirs:"list[str]",
        semi_holdout:bool
    )-> "tuple[list[str],list[str]]":
    file_paths = []
    train_paths = []
    # check if we shouldn't holdout anthing
    if semi_holdout == False:
        for subj_dir in subj_dirs:
            file_names = os.listdir(subj_dir)
            for file_name in file_names:
                file_paths.append(os.path.join(subj_dir, file_name))
    else: # we should holdout one for train
        for subj_dir in subj_dirs:
            file_names = os.listdir(subj_dir)
            file_names.sort()
            train_paths.append(os.path.join(subj_dir, file_names[0]))
            for file_name in file_names[1:]:
                file_paths.append(os.path.join(subj_dir, file_name))
    return (file_paths, train_paths)


# helper - adds a location back to the elements of a list
def add_back_loc_to_path(loc:str, subj_dirs:str)-> "list[str]":
    new_list = []
    for subj_dir in subj_dirs:
        loc_path = os.path.join(loc, subj_dir)
        new_list.append(os.path.join(BASE_DIR, loc_path))
    return new_list


# remove the subject dir of the training subject
def remove_training_subj(subj_id:str, subj_dirs:"list[str]")-> "list[str]":
    new_list = []
    for subj_dir in subj_dirs:
        if os.path.basename(subj_dir) != subj_id:
            new_list.append(subj_dir)
    if len(subj_dirs)-1 != len(new_list):
        raise Exception("Failed to remove the subject from the location...")
    return new_list


# returns the feature vectors selected for training/testing
# at a given interval and for a specific model
def get_feature_data(data:dict, t_interval:int, model_type:str)-> "list[list[int]]":
    # assemble all of the feature vectors from a given model
    # into an array, using the integer for seconds time intervals
    feature_vectors = []
    num_features = len(data["frame_data"])
    for t_index in range(0, num_features, t_interval):
        feature_vectors.append(
            data["frame_data"][t_index][model_type]
        ) # end append feature vector to list of feature vectors
    return feature_vectors


# get test files as a separate list
def get_test_file_paths(train_path:str, file_paths:"list[str]")-> "list[str]":
    test_paths = []
    for file_path in file_paths:
        if file_path != train_path:
            test_paths.append(file_path)
    if len(file_paths)-1 != len(test_paths):
        print(test_paths)
        print(train_path)
        raise Exception("Failed to get test paths correctly...")
    return test_paths


# parse file name to get the file with the given subcomponent
def get_file_with_subcomponent(file_names:"list[str]", word_to_find:str)-> Union[str, None]:
    for f_name in file_names:
        for start in range(0, len(f_name)-(len(word_to_find)-1)):
            name_slice = f_name[start:start+len(word_to_find)]
            if name_slice == word_to_find:
                return f_name
    return None


# accumulate results for a single location
# using generated test files from individual participants
def aggregate_results():
    pass


if __name__ == "__main__":
    main()
