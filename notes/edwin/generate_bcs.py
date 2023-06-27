"""
    BioCapsule Enrollment Phase
    BioCapsule
    Edwin Sanchez

    Generates BioCapsules for training a Binary Classifier.
    Store Classifier to use during active Auth.
"""

import src.biocapsule as bc
import src.face_models as fr

import tensorflow as tf

import cv2
import numpy as np


# Set random seed for reproducibility
np.random.seed(42)

def main():
    pass


def train_classifier(x_train, y_train):
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



    clf = LogisticRegression(
        class_weight="balanced", random_state=42
    ).fit(X_train_bianry, y_train_binary)

    y_pred = clf.predict(X_test_bianry)

if __name__ == "__main__":
    main()
