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
    - confusion matrix between the 150 people!!!! (That would be so cool)

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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from test_enums import *
from data_tools import *

# Params #
model_type = Model_Type.ARCFACE

# load rs features for later bc use
rs_data = load_json_gz_file(
    file_path="./MOBIO_extracted/one_sec_intervals/XX_removed_from_exp_XX/f210/unis_laptop_1_f210_01.json.gz")
ref_subj_feat_vect = rs_data["frame_data"][0][model_type]

# locations in MOBIO dataset
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
    subjects_data = []
    for input_dir in input_dirs:
        # loop over participants
        for part_id in os.listdir(input_dir):
            # get out the subject data for every subject
            # and load the data into memory.
            # only the data we need to run the tests
            subject_data = get_subj_data(
                subj_dir=os.path.join(input_dir, part_id),
                platform=Platform.MULTI,
                t_interval=10,
                model_type=Model_Type.ARCFACE
            ) # end get_subj_data call
            subjects_data.append(subject_data)
        # end loop over participants in a location
    # end loop over all directories to search through for data
    
    # run a test generate a file containing
    # data on the test. include test params
    out_data = run_test(
        subjects_data=subjects_data,
        use_bc=False,
        window_size=1,
        classifier_type=Classifier.LOGISTIC_REGRESSION,
        use_k_fold=False
    ) # end run_test
# end main


# run a test for a single participant,
# given the input params
# store the data generated for each invididual test for later
def run_test(subjects_data:"list[dict]",
             use_bc:bool,
             window_size:int,
             classifier_type:Classifier,
             use_k_fold:bool)-> Union[dict, None]:
    # loop over every element in the subjects_data list
    # the whole subjects_data array will be used in every iteration,
    # but the classifier & test will be done with respect to the current
    # subject data dict being iterated over by the for loop
    for subj_index, subject_data in enumerate(subjects_data):
        # the id of the current subject (participant id)
        subj_id = subjects_data["subject_ID"]

        # extract out the train & val data from the current sample
        train_pos_samples = subject_data["train_samples"]["train_split_samples"]
        val_pos_samples = subject_data["train_samples"]["val_split_samples"]

        # extract out the train & val data from every other subject

        # create labels for train & val data

        # combine all data together

        # train classifier for this subject
        # data is shuffled with a constant 'random' number
        # training is rebalanced based on the ratio of pos to neg samples
        classifier = LogisticRegression(
            class_weight="balanced", random_state=42
        ).fit(train_samples, train_labels)
        # store classifier???

        # perform threshold tuning using validation set
        threshold_tuning()
        
        # iterate over all subjects, using their test samples
        this_subj_test_results = []
        all_other_subj_test_results = []
        for test_subj_index, test_subj_data in enumerate(subjects_data):
            for session in test_subj_data["test_samples"]:
                session_test()
                if subj_index == test_subj_index:
                    pass
                else:
                    pass
            # end loop over a single session for testing
        # end loop over all other subjects for testing
    # end loop focusing on each subject as the positive case
    # store overall test performance data into file
# end run_test


# applies the bc scheme, if applicable
def apply_bc_scheme(bc_gen:bc.BioCapsuleGenerator,
                    samples:"list[list[float]]",
                    reference_subject:"list[float]"
                    )-> "list[list[float]]":
    pass


# session test - does a test over a single session &
# reports the performance metrics
def session_test():
    # apply biocapsule, if applicable
    if use_bc == True:
        bc_gen = bc.BioCapsuleGenerator()
        train_samples = apply_bc_scheme(
            bc_gen=bc_gen,
            samples=train_samples,
            reference_subject=ref_subj_feat_vect
        ) # end apply_bc_scheme to train_samples
        test_samples = apply_bc_scheme(
            bc_gen=bc_gen,
            samples=test_samples,
            reference_subject=ref_subj_feat_vect
        ) # end apply_bc_scheme to train_samples

    # run classifier on test data
    mean_acc = classifier.score(test_samples, test_labels)
    # get confusion matrix to extract data
    pred_test_labels = classifier.predict(test_samples)
    conf_matrix = confusion_matrix(test_labels, pred_test_labels)
    tn, fp, fn, tp = conf_matrix.ravel()
    #far
    far = fp / (fp + fn)
    #frr
    frr = fn / (tp + tn)
    #eer
    # generate performance metrics
    # store data for single subject into file


# creates a list containing all of one type of sample
def combine_samples(group_A:list, # negative samples list
                    group_B:list, # positive samples list
                    )-> "tuple[list[list[float]], list[int]]":
    len_A = len(group_A)
    len_B = len(group_B)
    labels_A = [0]*len_A
    labels_B = [1]*len_B
    labels = labels_A.extend(labels_B)
    samples = group_A.extend(group_B)
    return (samples, labels)


# get the samples from a subject as train, validation, and test splits
# for a single subject
def get_subj_data(subj_dir:str,
                     platform:Platform,
                     t_interval:int,
                     model_type:str,
                    )-> "tuple[dict, dict]":
    # get list of sessions collected during
    # data collection stage
    session_file_names = os.listdir(subj_dir)
    session_file_names.sort()
    print(f"Num of sessions: {len(session_file_names)}")
    print(session_file_names)
    if len(session_file_names) != 13:
        print(f"Number of session file names is not the right number: {len(session_file_names)}")

    # pull out the laptop file from the rest
    laptop_file_name = get_file_with_subcomponent(
        file_names=session_file_names,
        word_to_find="laptop"
    ) # end_get_file_with_subcomponent
    if laptop_file_name == None:
        print("No Laptop data... If using Multi-Platform, it won't work.")

    # separate out the train/test sessions, based on platform
    # if multi-platform, then only perform a single test
    # using laptop for train, the rest for test

    # if using multi platform, then train is the
    # laptop file path the rest are for testing
    # remove either way - we want it separate in both cases
    pos_train_file = None
    pos_test_files = None
    mobile_sess_names = remove_file_from_list(
        file_to_remove=laptop_file_name,
        file_paths=session_file_names
    ) # end get_test_file_paths
    mobile_sess_names.sort()

    # load data based off of platform param
    if platform == Platform.SINGLE:
        # train is the first session from mobile
        # (due to potential file name issues,
        # sort the sessions and select the first one)
        pos_train_file = mobile_sess_names[0]
        pos_test_files = remove_file_from_list(
            file_to_remove=pos_train_file,
            file_paths=mobile_sess_names
        ) # end remove_file_from_list
        assert len(pos_test_files) == (len(mobile_sess_names)-1)
    else: # set up for multi
        pos_train_file = laptop_file_name
        pos_test_files = mobile_sess_names
        assert len(pos_test_files) == (len(session_file_names)-1)

    # collect pos & negative samples for training & testing

    # add back directory path for pos sample files
    pos_train_path = os.path.join(subj_dir, pos_train_file)
    pos_test_paths = add_back_path(subj_dir, pos_test_files)
   
    # get positive train & test samples from pos files
    train_samples, test_samples = get_samples(
        training_path=pos_train_path,
        test_paths=pos_test_paths,
        t_inverval=t_interval,
        model_type=model_type
    ) # end get_pos_samples

    # organize subject's data into a dict
    subject_data = {
        "subject_ID": os.path.basename(subj_dir),
        "train_samples": train_samples,
        "test_samples": test_samples
    } # end subject data
    return subject_data
    

# get samples, in train/val/test split
def get_samples(training_path:str, 
                test_paths:"list[str]",
                t_inverval:int,
                model_type:str,
                train_split_percentage:float=0.8,
                rand_shuffle_seed:int=42
                )-> "tuple[dict, dict]":
    # get training positive samples
    # load data file from training_path
    data = load_json_gz_file(file_path=training_path)
    samples = get_feature_data(data=data,
                                         t_interval=t_inverval,
                                         model_type=model_type)
    # split the training samples into train & val sets
    train_split_samples, val_split_samples, split_index = get_train_test_split(
        samples=samples,
        train_split_percentage=train_split_percentage,
        rand_shuffle_seed=rand_shuffle_seed
    ) # end get_train_test_split call
    train_samples = {
        "file_path": test_path,
        "session_ID": data["MOBIO"]["session_ID"],
        "split_index": split_index,
        "suffle_seed": rand_shuffle_seed,
        "train_split_percentage": train_split_percentage,
        "train_split_samples": train_split_samples,
        "val_split_samples": val_split_samples
    } # end train samples dict

    # grab the test samples from the remaining sessions
    # keep track of sessions - we will score based off of
    # sessions.
    test_samples = []
    test_paths.sort()
    for test_path in test_paths:
        data = load_json_gz_file(file_path=test_path)
        test_session = {
            "file_path": test_path,
            "session_ID": data["MOBIO"]["session_ID"],
            "features": get_feature_data(data=data, 
                            t_interval=t_inverval, 
                            model_type=model_type)
        } # end dict for a single test session
        test_samples.append(test_session)
    # end for over test session
    
    # return a tuple containing the train, val, & test samples
    return (train_samples, test_samples)


# split training samples into test & val sets
# default to 80-20 split
def get_train_test_split(samples:"list[list[float]]", 
                         train_split_percentage:float=0.8,
                         rand_shuffle_seed:int=42
                         )-> "tuple[list[list[float]], list[list[float]], int]":
    # shuffle the samples for training before splitting
    random.Random(x=rand_shuffle_seed).shuffle(samples)
    # split train pos samples into train & val sets
    split_index = int(len(samples) * train_split_percentage)
    train_split_samples = samples[:split_index]
    val_split_samples = samples[split_index:]
    return (train_split_samples, val_split_samples, split_index)


# returns the feature vectors selected for training/testing
# at a given interval and for a specific model
def get_feature_data(data:dict, t_interval:int, model_type:str)-> "list[list[float]]":
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
def remove_file_from_list(file_to_remove:str, file_paths:"list[str]")-> "list[str]":
    test_paths = []
    for file_path in file_paths:
        if file_path != file_to_remove:
            test_paths.append(file_path)
    return test_paths


# parse file name to get the file with the given subcomponent
def get_file_with_subcomponent(file_names:"list[str]", word_to_find:str)-> Union[str, None]:
    for f_name in file_names:
        for start in range(0, len(f_name)-(len(word_to_find)-1)):
            name_slice = f_name[start:start+len(word_to_find)]
            if name_slice == word_to_find:
                return f_name
    return None


# add back a directory to a list of file names
def add_back_path(base_dir:str,  file_names:"list[str]")->"list[str]":
    new_list = []
    for file_name in file_names:
        new_list.append(os.path.join(base_dir, file_name))
    return new_list


# accumulate results for a single subject
def aggregate_results_test():
    pass

# accumulate the results across the entire dataset
def aggregate_results_final():
    pass


if __name__ == "__main__":
    main()
