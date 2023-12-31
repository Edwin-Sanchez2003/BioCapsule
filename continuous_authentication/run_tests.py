"""
    BioCapsule
    Run Tests

    This file is used to run testing for Continuous Authentication
    performance of the BioCapsule (BC) system with Face Authentication.
"""

import copy
import os
import random
import time

import tools
import window
from load_mobio_data import SubjectData, load_MOBIO_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

"""
# implement arguments at some point...
import argparse

parser = argparse.ArgumentParser(
    prog='Continuous Authentication Testing',
    description="This file runs tests for BioCapsule Continuous Authentication Testing."
) # end ArgumentParser construction

# arguments to be passed in by the user via the command line/terminal
parser.add_argument('-bd', '--background_images_dir', help="The directory containing background images to add objects to.")
parser.add_argument('-od', '--object_images_dir', help="The images that have objects to cut & paste onto background images.")
parser.add_argument('-a', '--annotations_file_path', help="The MSCOCO annotations for the images that have objects in object_images_dir.")
parser.add_argument('-o', '--output_dir', default="./generated_patch_data/", help="The directory to put the generated annotations and images. Created if it doesn't already exist.")
parser.add_argument('-n', '--make_noise_control', action="store_true", help="Whether or not to make a control version of the data using noise patches. See paper for more details.")

# get arguments from user
args = parser.parse_args()
"""

# Params #
EXTRACTED_MOBIO_DIR = "./MOBIO_extracted/one_sec_intervals/"
REFERENCE_SUBJ_DATA_DIR = (
    "./MOBIO_extracted/rs_features.json"  # LFW faces as reference subjects
)
OUT_DIR = "./MOBIO_extracted/test_results/"

USE_BC = True
FEATURE_EXTRACTION_MODEL = "facenet"  # "facenet"
TRAINING_PLATFORM = "multi"  # "multi"
TIME_INTERVAL = 10
WINDOW_SIZE = 1
MULTI_RS = True  # only relevant for biocapsule

# the function to use for averaging windows
AVERAGING_METHOD = ("simple_average", window.simple_average)

# extra name for testing code
EXTRA_NAME = "multi-rs"


def main():
    # check TRAINING_PLATFORM param. the other params are checked later
    if TRAINING_PLATFORM != "single":
        if TRAINING_PLATFORM != "multi":
            raise Exception(
                "Training Platform is not valid. Must be either 'single' or 'multi'"
            )

    # load data into a simple to use format
    subjects = load_MOBIO_dataset(
        extracted_MOBIO_dir=EXTRACTED_MOBIO_DIR,
        time_interval=TIME_INTERVAL,
        feature_extraction_model=FEATURE_EXTRACTION_MODEL,
        use_bc=USE_BC,
        multi_rs=MULTI_RS,
        ref_subj_data_dir=REFERENCE_SUBJ_DATA_DIR,
    )  # end load_MOBIO_dataset function call

    # perform tests per user
    print("Performing Tests...")
    tp, fp, tn, fn = 0, 0, 0, 0
    thresholds_list = []
    threshold_avg = 0
    per_subject_results = []
    for i in range(len(subjects)):
        print(f"Test for subject: {subjects[i].get_subject_id()}")
        results = single_user_test(
            subjects=subjects,
            subject_index=i,
            training_platform=TRAINING_PLATFORM,
            window_size=WINDOW_SIZE,
        )  # end single_user_test
        # accumulate tp, fp, tn, fn for all subjects
        tp += results["tp"]
        fp += results["fp"]
        tn += results["tn"]
        fn += results["fn"]
        threshold_avg += results["threshold"]
        thresholds_list.append(results["threshold"])
        per_subject_results.append(copy.deepcopy(results))
    # end for loop over subjects

    # get average threshold
    threshold_avg = threshold_avg / len(subjects)

    # get the non-face-count
    total_bad_detections = get_bad_detection_count(
        subjects=subjects, training_platform=TRAINING_PLATFORM
    )

    # collect information from testing
    # store the probability classifications so we can extract more data later
    out_data = {
        "use_bc": USE_BC,
        "feature_extraction_model": FEATURE_EXTRACTION_MODEL,
        "training_platform": TRAINING_PLATFORM,
        "time_interval": TIME_INTERVAL,
        "window_size": WINDOW_SIZE,
        "averaging_method": AVERAGING_METHOD[0],
        "multi_rs": MULTI_RS,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "far": get_far(fp=fp, tn=tn),
        "frr": get_frr(fn=fn, tp=tp),
        "total_bad_detections": total_bad_detections,
        "average_threshold": threshold_avg,
        "thresholds_list": thresholds_list,
        "per_subject_results": per_subject_results,
    }  # end out data

    # make sure output dir exists
    if os.path.isdir(OUT_DIR) == False:
        os.makedirs(OUT_DIR)

    # store out data in a file
    print("Writing results to file...")
    out_file_name = f"{USE_BC}_{FEATURE_EXTRACTION_MODEL}_{TRAINING_PLATFORM}_{EXTRA_NAME}_0.json"
    out_file_path = os.path.join(OUT_DIR, out_file_name)
    keepGoing = True
    sentry = 0
    while keepGoing:
        sentry += 1
        if os.path.isfile(out_file_path):
            out_file_name = f"{USE_BC}_{FEATURE_EXTRACTION_MODEL}_{TRAINING_PLATFORM}_{EXTRA_NAME}_{sentry}.json"
            out_file_path = os.path.join(OUT_DIR, out_file_name)
        else:
            keepGoing = False
    tools.write_to_json(out_file_path, out_data)
    print("Finished writing to file!")


# end main function


# performs a single test with a given index for which user to use
def single_user_test(
    subjects: "list[SubjectData]",
    subject_index: int,
    training_platform: str,
    window_size: int = 1,
) -> dict:
    subject = subjects[subject_index]

    # get positive train, val, and test samples and labels
    # do deep copy for safety - we will be extending the list later
    # we don't want to extend the class variable itself, so we need to deepcopy
    # can be ignored for labels as labels are generated on the fly
    pos_train_samples = None
    pos_train_labels = None
    # classification is 1 since these are positive samples
    if training_platform == "single":
        pos_train_samples = copy.deepcopy(
            subject.get_mobile_session_one().get_feature_vectors()
        )
        pos_train_labels = subject.get_mobile_session_one().get_labels(
            classification=1
        )
        pos_train_samples.extend(
            copy.deepcopy(
                subject.get_mobile_session_one().get_flipped_feature_vectors()
            )
        )
        pos_train_labels.extend(
            subject.get_mobile_session_one().get_labels(classification=1)
        )
    else:  # multi/cross platform (train with laptop data)
        pos_train_samples = copy.deepcopy(
            subject.get_laptop_session().get_feature_vectors()
        )
        pos_train_labels = subject.get_laptop_session().get_labels(
            classification=1
        )
        pos_train_samples.extend(
            copy.deepcopy(
                subject.get_laptop_session().get_flipped_feature_vectors()
            )
        )
        pos_train_labels.extend(
            subject.get_laptop_session().get_labels(classification=1)
        )
    # end get pos train data

    # split into train & validation sets
    (
        pos_train_samples,
        pos_train_labels,
        pos_val_samples,
        pos_val_labels,
    ) = get_train_test_split(
        samples=pos_train_samples, labels=pos_train_labels
    )  # end get_train_test_split

    # get negative train & validation samples
    neg_train_samples = []
    neg_train_labels = []
    for i, test_subject in enumerate(subjects):
        if (
            i == subject_index
        ):  # make sure we're not getting this subject's data again
            continue

        # get the negative data
        # classification is 0 since these are negative samples
        if training_platform == "single":
            neg_train_samples.extend(
                copy.deepcopy(
                    test_subject.get_mobile_session_one().get_feature_vectors()
                )
            )
            neg_train_labels.extend(
                test_subject.get_mobile_session_one().get_labels(
                    classification=0
                )
            )
        else:  # multi/cross platform (train with laptop data)
            neg_train_samples.extend(
                copy.deepcopy(
                    test_subject.get_laptop_session().get_feature_vectors()
                )
            )
            neg_train_labels.extend(
                test_subject.get_laptop_session().get_labels(classification=0)
            )
        # end get neg train data
    # end for loop over other subjects to get negative train & validation data

    # split into train & validation sets
    (
        neg_train_samples,
        neg_train_labels,
        neg_val_samples,
        neg_val_labels,
    ) = get_train_test_split(
        samples=neg_train_samples, labels=neg_train_labels
    )  # end get_train_test_split

    # combine train pos & neg
    train_samples, train_labels = combine_samples_and_labels(
        samples=[pos_train_samples, neg_train_samples],
        labels=[pos_train_labels, neg_train_labels],
    )  # end combine_samples_and_labels

    # combine val pos & neg
    val_samples, val_labels = combine_samples_and_labels(
        samples=[pos_val_samples, neg_val_samples],
        labels=[pos_val_labels, neg_val_labels],
    )  # end combine_samples_and_labels

    # train the classifier
    print("Training Classifier...")
    classifier = LogisticRegression(
        class_weight="balanced", random_state=42
    ).fit(train_samples, train_labels)
    print("Classifier trained!")

    # perform threshold tuning
    threshold = tune_threshold(
        classifier=classifier,
        val_samples=val_samples,
        val_labels=val_labels,
        window_size=window_size,
    )  # end threshold tuning

    # loop over every subject again
    tp, fp, tn, fn = 0, 0, 0, 0
    per_test_results = []
    for i, test_subject in enumerate(subjects):
        # get test data for this subject
        per_session_results = []
        subj_tp, subj_fp, subj_tn, subj_fn = 0, 0, 0, 0
        for session in test_subject.get_mobile_sessions():
            temp_fn = 0
            temp_tn = 0
            # this is the same subject, set classifcation accordingly
            test_samples = None
            test_labels = None
            if i == subject_index:  # same subject
                test_samples = session.get_feature_vectors()
                test_labels = session.get_labels(classification=1)
                # account for bad samples
                temp_fn += session.get_bad_detection_count()
            else:  # different subject
                test_samples = session.get_feature_vectors()
                test_labels = session.get_labels(classification=0)
                # account for bad samples
                temp_tn += session.get_bad_detection_count()
            # end if check for positive or negative subject during testing

            # run through classifier, get results
            (
                session_tn,
                session_fp,
                session_fn,
                session_tp,
                preds,
            ) = get_test_results(
                classifier=classifier,
                test_samples=test_samples,
                test_labels=test_labels,
                threshold=threshold,
                window_size=window_size,
            )  # end get_test_results
            session_tn += temp_tn
            session_fn += temp_fn

            # accumulate tp, fp, tn, fn for test subject
            subj_tp += session_tp
            subj_fp += session_fp
            subj_tn += session_tn
            subj_fn += session_fn

            # get per session results
            per_session_results.append(
                {
                    "session": session.get_session_file_path(),
                    "tp": session_tp,
                    "fp": session_fp,
                    "tn": session_tn,
                    "fn": session_fn,
                    "ground_truth_labels": copy.deepcopy(test_labels),
                    "predicted_probabilities_pos": copy.deepcopy(preds),
                }
            )
        # end for over each subject's session

        # accumulate tp, fp, tn, fn for this subject
        tp += subj_tp
        fp += subj_fp
        tn += subj_tn
        fn += subj_fn

        per_test_results.append(
            {
                "subject_id": test_subject.get_subject_id(),
                "tp": subj_tp,
                "fp": subj_fp,
                "tn": subj_tn,
                "fn": subj_fn,
                "per_session_results": copy.deepcopy(per_session_results),
            }
        )
    # end for loop over each subject

    # assemble collected data for this subject
    subject_results = {
        "subject_id": subject.get_subject_id(),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "test_results": copy.deepcopy(per_test_results),
    }
    return subject_results


# end single_user_test function


# get test results
def get_test_results(
    classifier: LogisticRegression,
    test_samples: "list[list[float]]",
    test_labels: "list[int]",
    threshold: float,
    window_size: int,
) -> "tuple[int, int, int, int, list]":
    # get predicted probability
    preds = classifier.predict_proba(test_samples)
    preds = preds[:, 1]

    # apply window to preds
    if window_size > 1:
        preds = window.apply_window_to_probabilities(
            window_size=window_size, preds=preds, avg_fn=AVERAGING_METHOD[1]
        )  # end apply_window_to_probabilities
    # end if check for window size

    # get classes using the threshold
    pred_labels = []
    for pred in preds:
        # check the classification = 1 probability
        if pred >= threshold:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    # end for over predicted probabilities

    # compare to test labels and get performance
    conf_matrix = confusion_matrix(
        y_true=test_labels, y_pred=pred_labels, labels=[0, 1]
    )
    tn, fp, fn, tp = conf_matrix.ravel()
    return (tn, fp, fn, tp, preds)


# end get_test_results function


# split training data into train and validation sets
def get_train_test_split(
    samples: "list[list[float]]",
    labels: "list[int]",
    train_split_percentage: float = 0.8,
    rand_shuffle_seed: int = 42,
) -> tuple:
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
        val_split_labels,
    )  # end return tuple


# tune the threshold for the classifier
def tune_threshold(
    classifier: LogisticRegression,
    val_samples: "list[list[float]]",
    val_labels: "list[int]",
    window_size: int,
) -> float:
    # get predicted probability
    preds = classifier.predict_proba(val_samples)
    preds = preds[:, 1]  # get only pos predictions

    # apply window to preds
    if window_size > 1:
        preds = window.apply_window_to_probabilities(
            window_size=window_size, preds=preds, avg_fn=AVERAGING_METHOD[1]
        )  # end apply_window_to_probabilities
    # end if check for window size

    step_size = 0.01  # the step size for the threshold
    target = 0.1  # the target percentage for far

    # loop over possible thresholds
    threshold = 0
    for i in range(50, 0, -1):
        threshold = i * step_size
        # get classes using the threshold
        pred_labels = []
        for pred in preds:
            # check the classification = 1 probability
            if pred >= threshold:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        # end for over predicted probabilities

        # get metrics based on thresh
        conf_matrix = confusion_matrix(
            y_true=val_labels, y_pred=pred_labels, labels=[0, 1]
        )
        tn, fp, fn, tp = conf_matrix.ravel()
        far = get_far(fp=fp, tn=tn)

        # check if the far is greater than target
        # if yes, then stop tuning
        # multiply by 100 to get a percentage
        if (far * 100) >= target:
            print(f"Threshold set to: {threshold}")
            return threshold
    # end for over test thesholds
    print("Did not find far... set to default threshold of 0.5!")
    return 0.5


# end tune_threshold


# takes a list of predicted probabilities and converts
# them to 1s or 0s based on a threshold value
# assumes preds are the probability for the "yes" authentication
# probabilities
def binarize_predictions(preds: "list[float]", threshold: int) -> "list[int]":
    pred_labels = [0] * len(preds)
    for i, pred in enumerate(preds):
        if pred >= threshold:
            pred_labels[i] = 1
        else:
            pred_labels[i] = 0
    # end for loop over predictions
    return pred_labels


# end binarize_predictions


# remove no authentication probability
# returns a list of single probabilities of "yes" authentication
def rem_extra_prob(preds: "list[list[float]]") -> "list[float]":
    slim_pred = [0] * (len(preds))
    for i, pred in enumerate(preds):
        slim_pred[i] = pred[1]
    return slim_pred


# takes a list of a list of samples, a list of list of labels,
# and puts them all into a single list for each using list.extend()
def combine_samples_and_labels(
    samples: "list[list[list[float]]]",
    labels: "list[list[int]]",
    rand_shuffle_seed: int = 42,
) -> "tuple[list[list[float]], list[int]]":
    # combine lists
    out_samples = []
    for sample_list in samples:
        out_samples.extend(sample_list)

    # combine labels
    out_labels = []
    for label_list in labels:
        out_labels.extend(label_list)

    # shuffle the samples and labels for training before splitting
    random.Random(x=rand_shuffle_seed).shuffle(out_samples)
    random.Random(x=rand_shuffle_seed).shuffle(out_labels)

    return (out_samples, out_labels)


# end combine_samples_and_labels


# gets the number of non-face samples during testing
def get_bad_detection_count(
    subjects: "list[SubjectData]", training_platform: str
) -> int:
    total_non_faces = 0
    for subject in subjects:
        # get non faces for the selected training set
        if training_platform == "single":  # get mobile train non-faces
            total_non_faces += (
                subject.get_laptop_session().get_bad_detection_count()
            )
        else:  # get laptop train non-faces
            total_non_faces += (
                subject.get_mobile_session_one().get_bad_detection_count()
            )

        # get non faces for the rest of the subject's sessions
        for session in subject.get_mobile_sessions():
            total_non_faces += session.get_bad_detection_count()
    # end for loop over subjects
    return total_non_faces


# end get_non_face_count


# False Acceptance Rate
def get_far(fp, tn) -> float:
    if (fp == 0) and (tn == 0):
        return None
    return float(fp / (fp + tn))


# False Rejection Rate
def get_frr(fn, tp) -> float:
    if (fn == 0) and (tp == 0):
        return None
    return float(fn / (fn + tp))


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Time to run experiment: {toc-tic:04f} seconds")
