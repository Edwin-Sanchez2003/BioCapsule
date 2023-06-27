"""
    BioCapsule Enrollment Phase
    BioCapsule
    Edwin Sanchez

    Generates BioCapsules for training a Binary Classifier.
    Store Classifier to use during active Auth.
"""

import sys
sys.path.insert(0, '../../src/')

import os

import biocapsule as bc
import face as fr

import cv2
import numpy as np

from sklearn.linear_model import LogisticRegression

import pickle

# Set random seed for reproducibility
np.random.seed(42)

def main():
    train_classifier(extract=False)


def train_classifier(extract=False):
    """
        Train Classifier
        ------
        Trains a classifier on biocapsules. The classifier is trained
        to make a 'yes' classification to authenticate a user, and a 
        'no' classification to deny a user.

        Input:
        ------
        - x_train: the ndarray containing the images of the dataset
        - y_train: the labels for the training data

        Output:
        ------
        - classifier: a trained classifier to perform binary classification
        for a given user.
    """

    # extract the dataset features (& store)
    if extract:
        fr.extract_dataset("lfw", gpu=0, add_new=True)

    # load features from file
    features = np.load(f"../../data/lfw_arcface_mtcnn_feat_edwin.npz")["arr_0"]
    
    # load rs feature for bc generation
    rs_feature = get_rs_feature()

    # generate biocapsules
    bc_gen = bc.BioCapsuleGenerator()

    x_train = []
    for feature in features:
        feature = feature[:-1]
        x_train.append(bc_gen.biocapsule(user_feature=feature, rs_feature=rs_feature))

    # generate labels for bcs
    y_train = np.zeros(shape=len(x_train))

    # the index of the images where mine are
    edwin_index_begin = len(x_train) - 14
    for i in range(14):
        y_train[(edwin_index_begin + i)] = 1

    # train model
    edwin_classifier = LogisticRegression(
        class_weight="balanced", random_state=42
    ).fit(x_train, y_train)

    # save model to pickle file
    pkl_classifier_file_name = "edwin_classifier.pkl"
    with open(pkl_classifier_file_name, "wb") as file:
        pickle.dump(edwin_classifier, file)


def get_rs_feature():
    """Return ArcFace features for 6 predetermined Reference Subjects
    (RSs) used in this experiment for reproducibility.

    """
    arcface = fr.ArcFace()

    rs_subjects = sorted(os.listdir("../../rs/"))
    rs_subjects = rs_subjects[4:]

    rs_features = np.zeros((6, 512))
    for s_id, subject in enumerate(rs_subjects):
        for image in os.listdir(f"../../rs/{subject}"):
            img = cv2.imread(f"../../rs/{subject}/{image}")
            feature = arcface.extract(img)
            rs_features[s_id] = feature

    return rs_features[0][:]


if __name__ == "__main__":
    main()
