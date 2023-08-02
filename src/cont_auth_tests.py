"""
    Continuous Authentication Testing
    BioCapsule
    
    Description:
    This file contains tests for Continuous Authentication,
    with a primary goal of testing the system with and
    without biocapsule involved.

    Dataset wide statistics:
    - how far apart are the feature embeddings of all of the samples?
    - what is the avg. euclidean dist betw. feature embeddings of users?
    - std. dev? (always have error bars)
    - how about after applying bc scheme? will give great insights to 
      how bc changes the representation of the feature embeddings
    - confusion matrix between the 150 people!!!! (That would be so cool)

    MOBIO Notes:
    - OMIT f210 from unis!!! Only 1 session exists for this individual
    - OMIT f218 !!! No laptop data
    - the rest have all 13 sessions (12 mobile, 1 laptop)


    ISSUE!!!!
     - Threshold tuning doesn't move very far before being locked in!!
      - ~ 0.07-0.08 (meaning any probabilities above these values pass as true positives)
            ^^^ That is shit!

    TODO:
     - write code to store only needed data for a test in uncompressed
       json format, with only necessary data. Loading time is taking too long. (~21 minutes)
       - prob not useful in the long run - need to generate differently for diff tests... (t_interval)
"""


# imports
import os
import copy
import math
import time
from typing import Union
import random

import biocapsule as bc

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from test_enums import *
from data_tools import *

# Params #
BASE_DIR = "./MOBIO_extracted/one_sec_intervals/"
OUT_DIR = "./MOBIO_extracted/test_results/"
T_INTERVAL = 10
USE_BC = False
MODEL_TYPE = Model_Type.ARCFACE.value
PLATFORM = Platform.MULTI.value # ERROR - NOT ALL SUBJECTS HAVE LAPTOP VIDEOS!!!
WINDOW_SIZE = 1
CLASSIFIER_TYPE = Classifier.LOGISTIC_REGRESSION.value
USE_K_FOLD = False

SAME_RS = True

# load rs features for later bc use
rs_data = load_json_gz_file(
    file_path="./MOBIO_extracted/one_sec_intervals/XX_removed_from_exp_XX/f210/unis_laptop_1_f210_01.json.gz")
ref_subj_feat_vect = rs_data["frame_data"][0]["feature_vectors"][MODEL_TYPE]
# locations in MOBIO dataset
MOBIO_LOCATIONS = [
    "but/",
    "idiap/",
    "lia/",
    "uman/",
    "unis/",
    "uoulu/"
] # end mobio locations

def main():
    # get all of the MOBIO locations as input directories
    input_dirs = []
    for loc_path in MOBIO_LOCATIONS:
        input_dirs.append(os.path.join(BASE_DIR, loc_path))

    # loop over input dirs
    print("Retrieving Subject Data...")
    long_tic = time.perf_counter()
    subjects_data = []
    subj_collect_count = 1
    for input_dir in input_dirs:
        # loop over participants
        for part_id in os.listdir(input_dir):
            # get out the subject data for every subject
            # and load the data into memory.
            # only the data we need to run the tests
            print(f"Count: {subj_collect_count}, Retrieving Subject {part_id}")
            tic = time.perf_counter()
            subject_data = get_subj_data(
                subj_dir=os.path.join(input_dir, part_id),
                platform=PLATFORM,
                t_interval=T_INTERVAL,
                model_type=MODEL_TYPE
            ) # end get_subj_data call
            if subject_data != None:
                subjects_data.append(subject_data)
            subj_collect_count += 1
            toc = time.perf_counter()
            print(f"Time : {toc - tic:0.4f} seconds")
        # end loop over participants in a location
    # end loop over all directories to search through for data
    long_toc = time.perf_counter()
    print("Finished retrieving data!")
    print(f"Time : {long_toc - long_tic:0.4f} seconds")
    # run a test generate a file containing
    # data on the test. include test params
    print("Running Test with the Following Params:")
    print(f"USE_BC: {USE_BC}")
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    print(f"PLATFORM: {PLATFORM}")
    print(f"T_INTERVAL: {T_INTERVAL}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"CLASSIFIER_TYPE: {CLASSIFIER_TYPE}")
    print(f"USE_K_FOLD: {USE_K_FOLD}")
    
    # print out all parameters set by the user here.
    print("Beginning Testing...")
    tic = time.perf_counter()
    out_data = run_test(
        subjects_data=subjects_data,
        use_bc=USE_BC,
        window_size=WINDOW_SIZE,
        classifier_type=CLASSIFIER_TYPE,
        use_k_fold=USE_K_FOLD
    ) # end run_test
    toc = time.perf_counter()
    print("Finished Testing!")
    print(f"Time : {toc - tic:0.4f} seconds")

    # print out results to see
    print(f"TP: {out_data['tp']}")
    print(f"FP: {out_data['fp']}")
    print(f"TN: {out_data['tn']}")
    print(f"FN: {out_data['fn']}")
    print(f"FAR: {out_data['far']}")
    print(f"FRR: {out_data['frr']}")

    # make sure output dir exists
    if os.path.isdir(OUT_DIR) == False:
        os.makedirs(OUT_DIR)

    # store out data in a file
    print("Writing results to file...")
    tic = time.perf_counter()
    out_file_name = f"{USE_BC}_{MODEL_TYPE}_{PLATFORM}_0.json"
    out_file_path = os.path.join(OUT_DIR, out_file_name)
    keepGoing = True
    sentry = 0
    while keepGoing:
        sentry += 1
        if os.path.isfile(out_file_path):
            out_file_name = f"{USE_BC}_{MODEL_TYPE}_{PLATFORM}_{sentry}.json"
            out_file_path = os.path.join(OUT_DIR, out_file_name)
        else:
            keepGoing = False
    write_to_json(out_file_path, out_data)
    toc = time.perf_counter()
    print("Finished writing to file!")
    print(f"Time : {toc - tic:0.4f} seconds")
# end main


# run a test for a single participant,
# given the input params
# store the data generated for each invididual test for later
def run_test(subjects_data:"list[dict]", # list of all user data split into train, val, and test
             use_bc:bool,
             window_size:int,
             classifier_type:Classifier,
             use_k_fold:bool)-> Union[dict, None]:
    # loop over every element in the subjects_data list
    # the whole subjects_data array will be used in every iteration,
    # but the classifier & test will be done with respect to the current
    # subject data dict being iterated over by the for loop
    g_tp, g_fp, g_tn, g_fn = 0,0,0,0 # global tp, fp, tn, fn
    pos_subj_results = []
    num_subjects = len(subjects_data)
    for subj_index, subject_data in enumerate(subjects_data):
        g_tic = time.perf_counter()
        print(f"Starting test for a single subject... {subj_index+1} of {num_subjects}")
        # the id of the current subject (participant id)
        subj_id = subject_data["subject_ID"]

        # extract out the train & val data from the current sample
        print("Extracting pos data...")
        train_pos_samples = subject_data["train_samples"]["train_split_samples"]
        val_pos_samples = subject_data["train_samples"]["val_split_samples"]

        # extract out the train & val data from every other subject
        print("Extracting neg data...")
        ext_tic = time.perf_counter()
        train_neg_samples = []
        val_neg_samples = []
        for neg_subj_index, neg_subj_data in enumerate(subjects_data):
            if neg_subj_index != subj_index:
                train_neg_samples.extend(neg_subj_data["train_samples"]["train_split_samples"])
                val_neg_samples.extend(neg_subj_data["train_samples"]["val_split_samples"])
            # end if
        # end for over train & val data collection
        ext_toc = time.perf_counter()
        print("Finished extracting neg data!")
        print(f"Time : {ext_toc - ext_tic:0.4f} seconds")

        # create labels for train & val data
        # combine all data together
        train_samples, train_labels = combine_samples(train_neg_samples, train_pos_samples)
        val_samples, val_labels = combine_samples(val_neg_samples, val_pos_samples)

        # check if we should use bc
        bc_gen = bc.BioCapsuleGenerator()
        if use_bc:
            print("Creating BioCapsules...")
            bc_tic = time.perf_counter()
            train_samples = apply_bc_scheme(
                bc_gen=bc_gen,
                samples=train_samples,
                reference_subject=ref_subj_feat_vect
            ) # end apply_bc_scheme
            val_samples = apply_bc_scheme(
                bc_gen=bc_gen,
                samples=val_samples,
                reference_subject=ref_subj_feat_vect
            ) # end apply_bc_scheme
            bc_toc = time.perf_counter()
            print("Finished creating BCs!")
            print(f"Time : {bc_toc - bc_tic:0.4f} seconds")
        # train classifier for this subject
        # data is shuffled with a constant 'random' number
        # training is rebalanced based on the ratio of pos to neg samples

        print("Training Classifier...")
        c_tic = time.perf_counter()
        classifier = LogisticRegression(
            class_weight="balanced", random_state=42
        ).fit(train_samples, train_labels)
        c_toc = time.perf_counter()
        print(f"Time : {c_toc - c_tic:0.4f} seconds")
        print("Classifier trained!")
        # store classifier???

        # perform threshold tuning using validation set

        threshold_data = threshold_tuning(classifier=classifier,
                         val_samples=val_samples,
                         val_labels=val_labels
        ) # end threhold tuning
        
        # iterate over all subjects, using their test samples
        this_subj_test_results = None
        all_other_subj_test_results = []
        s_tp, s_fp, s_tn, s_fn = 0,0,0,0 # per-classifier-subject tp, fp, tn, fn
        for test_subj_index, test_subj_data in enumerate(subjects_data):
            # get a performance on each session using the threshold
            subj_results = {
                "subj_id_train": subj_id,
                "subj_id_test": test_subj_data["subject_ID"],
                "threshold_tuning_data": threshold_data,
                "sess_results": []
            } # subject results and metadata
            tp, fp, tn, fn = 0,0,0,0
# test labels should be parameterized for whether its a positive subject or negative subject!!!
            for session in test_subj_data["test_samples"]:
                test_labels = None
                if subj_results["subj_id_test"] == subj_results["subj_id_train"]:
                    test_labels = [1]*len(session["features"])
                else:
                    test_labels = [0]*len(session["features"])

                # results from a single session from a single user
                results = session_test(
                    classifier=classifier,
                    threshold=threshold_data["threshold"],
                    bc_gen=bc_gen,
                    use_bc=use_bc,
                    test_samples=session["features"],
                    test_labels=test_labels
                ) # end test for single session
                tp += int(results["tp"])
                fp += int(results["fp"])
                tn += int(results["tn"])
                fn += int(results["fn"])
                s_tp += int(results["tp"])
                s_fp += int(results["fp"])
                s_tn += int(results["tn"])
                s_fn += int(results["fn"])
                g_tp += int(results["tp"])
                g_fp += int(results["fp"])
                g_tn += int(results["tn"])
                g_fn += int(results["fn"])

                subj_results["sess_results"].append(results)
            # end loop over sessions for testing

            # get per-test-subject results
            subj_results["tp"] = int(tp)
            subj_results["fp"] = int(fp)
            subj_results["tn"] = int(tn)
            subj_results["fn"] = int(fn)

            # where to put test results
            if subj_results["subj_id_test"] == subj_results["subj_id_train"]:
                this_subj_test_results = subj_results
            else:
                print("Finished Negative Test")
                all_other_subj_test_results.append(subj_results)
        # end loop over all other subjects for testing
        # per-classifier-subject results
        pos_subj_result = {
            "subj_id": subj_id,
            "pos_subj_results": this_subj_test_results,
            "neg_subjs_restults": all_other_subj_test_results,
            "tp": int(s_tp),
            "fp": int(s_fp),
            "tn": int(s_tn),
            "fn": int(s_fn),
        } # end pos_subj_results
        pos_subj_results.append(pos_subj_result)
        g_toc = time.perf_counter()
        print(f"Time : {g_toc - g_tic:0.4f} seconds")
    # end loop focusing on each subject as the positive case

    # get final results over every subject's data
    print(f"USE_BC: {USE_BC}")
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    print(f"PLATFORM: {PLATFORM}")
    print(f"T_INTERVAL: {T_INTERVAL}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"CLASSIFIER_TYPE: {CLASSIFIER_TYPE}")
    print(f"USE_K_FOLD: {USE_K_FOLD}")
    
    # create a dict to hold the important data for the final results of the test
    final_results = {
        "use_bc": use_bc,
        "model_type": MODEL_TYPE,
        "platform": PLATFORM,
        "t_interval": T_INTERVAL,
        "window_size": window_size,
        "classifier": classifier_type,
        "use_k_fold": use_k_fold,
        "tp": int(g_tp),
        "fp": int(g_fp),
        "tn": int(g_tn),
        "fn": int(g_fn),
        "far": get_far(fp=g_fp, tn=g_tn),
        "frr": get_frr(fn=g_fn, tp=g_tp),
        "pos_subj_results": pos_subj_results
    } # end results for all subjects' performance

    # return data
    return final_results
# end run_test

# False Acceptance Rate
def get_far(fp, tn)-> float:
    if (fp == 0) and (tn == 0):
        return None
    return float(fp / (fp + tn))


# False Rejection Rate
def get_frr(fn, tp)-> float:
    if (fn == 0) and (tp == 0):
        return None
    return float(fn / (fn + tp))


# return the threshold between pos and neg samples
# that reaches an eer rate. returns the f
def threshold_tuning(classifier:LogisticRegression,
                     val_samples:"list[list[float]]",
                     val_labels:"list[int]",
                     start:int=1, # start at 1 percent confidence
                     stop:int=100, # stop before 100 percent confidence
                     step:int=1, # walk up from start to stop
                     precision:float=0.01)-> "dict":
    # run classifier on validation samples to get probability scores
    preds = classifier.predict_proba(val_samples)
    # loop over thresholds
    # walk from the start precentage confidence down to the stop precision confidence,
    # checking whether or not to stop as we've hit an eer rate
    #for i in range(start, stop, step):
    thresh = 0.5
    #print(f'Threshold: {thresh}')
    (tp, fp, tn, fn), conf_matrix = get_tp_fp_tn_fn(
        thresh=thresh,
        gt_labels=val_labels,
        preds=preds
    ) # end get_tp_fp_tn_fn
    
    #far
    far = get_far(fp=fp, tn=tn)
    #frr
    frr = get_frr(fn=fn, tp=tp)
    #print(f"far: {far}, frr: {frr}")
    #if is_equal_error_rate(far=far, frr=frr):
    #print(f"EER has been found! -> far: {far}, frr: {frr} | threshold: {thresh}")
    print(f"Threshold Performance -> far: {far}, frr: {frr} | threshold: {thresh}")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "far": far,
        "frr": frr,
        "threshold": 0.5 # hard-code threshold for now
    } # return threshold data dict
    # end loop over thresholds to test
    raise Exception("Did not find an equal error rate - was the validation data setup correctly?")
# threshold tuning


# get tp, fp, tn, fn from pred probabilities and ground truth
def get_tp_fp_tn_fn(thresh:float,
                    gt_labels:"list[int]",
                    preds:"list[tuple[int, int]]"
                    )->"tuple[int, int, int, int]":
    predicted_labels = []
    for pred_prob in preds:
        # check the positive class probability
        if pred_prob[1] >= thresh:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    # end loop accumulating predicted labels

    # get confusion matrix
    conf_matrix = confusion_matrix(y_true=gt_labels, y_pred=predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    return (int(tp), int(fp), int(tn), int(fn)), conf_matrix

"""
- double check labels look correct (gt & prediction)

# (tn + tp) / (tn + fp + fn + tp)
acc = (conf_mat[0] + conf_mat[3]) / np.sum(conf_mat)
# fp / (tn + fp)
far = conf_mat[1] / (conf_mat[0] + conf_mat[1])
# fn / (fn + tp)
frr = conf_mat[2] / (conf_mat[2] + conf_mat[3])
"""


# checks if far & frr are equal, with a given precision
def is_equal_error_rate(far:float, frr:float, precision:float=0.01)-> bool:
    # if the difference is less than or equal to the precision, then return true
    # if the difference is greater than the precision, then return false
    if (far == None) or (frr == None):
        return False
    return (abs(far-frr) <= precision)


# applies the bc scheme, if applicable
def apply_bc_scheme(bc_gen:bc.BioCapsuleGenerator,
                    samples:"list[list[float]]",
                    reference_subject:"list[float]"
                    )-> "list[list[float]]":
    bc_samples = []
    for sample in samples:
        bc_samples.append(
            bc_gen.biocapsule(np.array(sample), np.array(reference_subject))
        ) # end append to bc_samples
    return bc_samples


# session test - does a test over a single session &
# reports the performance metrics
def session_test(classifier:LogisticRegression,
                 threshold:float,
                 bc_gen:bc.BioCapsuleGenerator,
                 use_bc:bool,
                 test_samples:"list[list[float]]",
                 test_labels:"list[int]"
                 )->dict:
    # apply biocapsule, if applicable
    if use_bc == True:
        test_samples = apply_bc_scheme(
            bc_gen=bc_gen,
            samples=test_samples,
            reference_subject=ref_subj_feat_vect
        ) # end apply_bc_scheme to train_samples

    # run classifier on test data
    preds = classifier.predict_proba(test_samples)
    # get confusion matrix to extract data
    (tp, fp, tn, fn), conf_matrix = get_tp_fp_tn_fn(
        thresh=threshold,
        gt_labels=test_labels,
        preds=preds
    ) # end get_tp_fp_tn_fn
    # store data for single subject into dict
    results = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold":int(threshold),
        "conf_matrix": conf_matrix
    } # end results
    return results


# creates a list containing all of one type of sample
# uses shortest list to decide which samlpes are posit
def combine_samples(group_A:list, # negative samples list
                    group_B:list, # positive samples list
                    group_A_neg:bool=True)-> "tuple[list[list[float]], list[int]]":
    len_A = len(group_A)
    len_B = len(group_B)
    if group_A_neg:
        labels_A = [0]*len_A
        labels_B = [1]*len_B
    else:
        labels_A = [1]*len_A
        labels_B = [0]*len_B
    labels_A.extend(labels_B)
    group_A.extend(group_B)
    labels = labels_A
    samples = group_A
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
    if platform == Platform.SINGLE.value:
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
        if len(pos_test_files) != (len(session_file_names)-1):
            print("This subject does not have laptop data... subject will be ignored")
            print(f"subject_id: {os.path.basename(subj_dir)}")
            return None
        #assert len(pos_test_files) == (len(session_file_names)-1)

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
        "file_path": training_path,
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
            data["frame_data"][t_index]["feature_vectors"][model_type]
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


if __name__ == "__main__":
    main()
