"""
    Enumerations for Testing Continuous Authentication
    BioCapsule

    Description:
    These are the enumeration classes that define certain
    aspects of testing for the bc cont. auth. testing.
    The categories are:
    - platform
    - model type
    - dataset name
    - classifier type
"""

from enum import Enum

# platform decision for the test
class Platform(Enum):
    SINGLE = "single"
    MULTI = "multi"

# the type of model feature vectors being used
class Model_Type(Enum):
    FACENET = "facenet"
    ARCFACE = "arcface"

# the name of the dataset being used to perform the test
class Dataset(Enum):
    MOBIO = "mobio"

# the type of classifier used to do authentication
class Classifier(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"

