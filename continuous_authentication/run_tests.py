"""
    BioCapsule
    Run Tests

    This file is used to run testing for Continuous Authentication
    performance of the BioCapsule (BC) system with Face Authentication.
"""

import random
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from load_mobio_data import load_MOBIO_dataset, SessionData, SubjectData

# Params #
EXTRACTED_MOBIO_DIR = ""
TIME_INTERVAL = 10
FEATURE_EXTRACTION_MODEL = "arcface"
TRAINING_PLATFORM = "single"
USE_BC = False
MULTI_RS = False


def main():
    # check TRAININ_PLATFORM param. the other params are checked later
    if TRAINING_PLATFORM != "single":
        if TRAINING_PLATFORM != "multi":
            raise Exception("Training Platform is not valid. Must be either 'single' or 'multi'")

    # load data into a simple to use format
    subjects = load_MOBIO_dataset(
        extracted_MOBIO_dir=EXTRACTED_MOBIO_DIR,
        time_interval=TIME_INTERVAL,
        feature_extraction_model=FEATURE_EXTRACTION_MODEL,
        use_bc=USE_BC,
        multi_rs=MULTI_RS
    ) # end load_MOBIO_dataset function call

    # perform tests per user
    print("Performing Tests...")
    for i in range(len(subjects)):
        print(f"Test with Positive subject as subject {subjects[i].get_subject_id()}")
        single_user_test()



    # collect information from test
    # store the probability classifications so we can extract more data later
    # make sure to document the parameters
    # used for each test


# performs a single test with a given index for which user to use
def single_user_test(
        subjects:"list[SubjectData]",
        subject_index:int,
        training_platform:str,
        window_size:int=1,
    )-> dict:
    subject = subjects[subject_index]

    # get positive train, val, and test samples and labels
    # do deep copy for safety - we will be extending the list later
    # we don't want to extend the class variable itself, so we need to deepcopy
    # can be ignored for labels as labels are generated on the fly
    pos_train_samples = None
    pos_train_labels = None
    if training_platform == "single":
        pos_train_samples = copy.deepcopy(subject.get_mobile_session_one().get_feature_vectors())
        pos_train_labels = subject.get_mobile_session_one().get_labels(classification=1)
    else: # multi/cross platform (train with laptop data)
        pos_train_samples = copy.deepcopy(subject.get_laptop_session().get_feature_vectors())
        pos_train_labels = subject.get_laptop_session().get_labels(classification=0)
    # end get pos train data

    # split into train & validation sets
    (pos_train_samples, 
     pos_train_labels, 
     pos_val_samples, 
     pos_val_labels) = get_train_test_split(
        samples=pos_train_samples,
        labels=pos_train_labels
    ) # end get_train_test_split

    # get negative train & validation samples
    for i, test_subject in enumerate(subjects):
        if i == subject_index: # make sure we're not getting this subject's data again
            continue

    
        neg_train_samples = None
        neg_train_labels = None
        if training_platform == "single":
            neg_train_samples = copy.deepcopy(subject.get_mobile_session_one().get_feature_vectors())
            neg_train_labels = subject.get_mobile_session_one().get_labels(classification=1)
        else: # multi/cross platform (train with laptop data)
            neg_train_samples = copy.deepcopy(subject.get_laptop_session().get_feature_vectors())
            neg_train_labels = subject.get_laptop_session().get_labels(classification=0)
        # end get pos train data

    # train the classifier
    print("Training Classifier...")
    classifier = LogisticRegression(
        class_weight="balanced", random_state=42
    ).fit(train_samples, train_labels)
    c_toc = time.perf_counter()
    print("Classifier trained!")

    # get test data
    # loop over every subject, running the 



    # loop over every subject again
    for i, test_subject in enumerate(subjects):
        # get test, val, and train samples into single arrays
        pass


    # end for testing on each subject
# end singel_user_test function


# split training data into train and validation sets
def get_train_test_split(samples:"list[list[float]]",
                         labels:"list[int]", 
                         train_split_percentage:float=0.8,
                         rand_shuffle_seed:int=42
                         )-> tuple:
    # shuffle the samples and labels for training before splitting
    random.Random(x=rand_shuffle_seed).shuffle(samples)
    random.Random(x=rand_shuffle_seed).shuffle(labels)

    # split train pos samples into train & val sets
    split_index = int(len(samples) * train_split_percentage)
    train_split_samples = samples[:split_index]
    val_split_samples = samples[split_index:]
    train_split_labels = labels[:split_index]
    val_split_labels = labels[split_index:]
    return (
        train_split_samples, 
        train_split_labels,
        val_split_samples, 
        val_split_labels
    ) # end return tuple


# takes a list of a list of samples, a list of list of labels,
# and puts them all into a single list for each using list.extend()
def combine_samples_and_labels(
        samples:"list[list[list[float]]]", 
        labels:"list[list[int]]"
    )-> "tuple[list[list[float]], list[int]]":
    out_samples = []
    out_labels = []

    for sample_list in samples:
        out_samples.extend(sample_list)
    
    for label_list in labels:
        out_labels.extend(label_list)

    return (out_samples, out_labels)
# end combine_samples_and_labels


if __name__ == "__main__":
    main()
