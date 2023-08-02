"""
    BioCapsule
    Load MOBIO Data

    This script is meant to load the extracted data from the
    MOBIO dataset. The data is processed and organized to be easily
    used for running the continuous authentication tests.
"""

"""
    Important info:
    - distinguishing subjects
    - counting if frame is none or multiple faces
"""

import tools
import copy

import biocapsule as bc


# Params #

# the type of model to use for the feature extraction
# features have already been extracted, so the options are
# 'facenet' or 'arcface'
FEATURE_EXTRACTION_MODEL = "facenet" # "arcface"
# the time interval from which to extract frames Must be
# at least 1 second, and an integer value.
TIME_INTERVAL = 10
# the platform from which to perform training.
# the options are to train and validate with the
# 'laptop' session or the 'mobile' session
TRAINING_PLATFORM = "multi" # single


# loads the data from the mobio dataset into an easy to
# use format for testing
def load_MOBIO_data(mobio_extr_data_dir:str)->dict:
    pass


# loads the data for a single subject in the MOBIO dataset.
# loads the individual data from each session and then packages
# into train, validation, and test splits for all needed cases
def get_single_subject_data(mobio_data_dir:str,
        subject_location:str,
        subject_id:str
        )-> dict:
    """
    Inputs
    -------
    mobio_data_dir : str
        The directory containing data extracted from the MOBIO dataset.
        Assumes data has been extracted using 'extract_MOBIO_data.py'

    subject_location : str
        The location in which the subject participated in the data
        collections process for MOBIO.

    subject_id : str
        The unique id of the subject, designated by the MOBIO dataset.
    """
    pass


# class to neatly store session information for a subject
class SessionData(object):
    def __init__(self, 
                 session_file_path:str,
                 time_interval:int=10,
                 feature_extraction_model:str="arcface",
                 use_bc:bool=False,
                 rs_feature_vector:"list[float]"=None):
        self.__session_file_path = session_file_path
        self.__time_interval = time_interval
        self.__feature_extraction_model = feature_extraction_model
        self.__use_bc = use_bc
        self.__rs_feature_vector = rs_feature_vector

        # load the feature vectors and the indices corresponding
        # to the feature vectors with non-faces
        self.__feature_vectors = None
        self.__non_face_indices = None
        self.__load_single_subject_session_data(session_file_path=session_file_path)
        
        # if using biocapsule, apply transformations and store back
        if use_bc:
            assert rs_feature_vector != None
            self.__make_biocapsules()


    # loads the data from a single session of a subject,
    # extracting the most important information to be accumulated
    def __load_single_subject_session_data(
            self,
            session_file_path:str
        )-> "tuple[list[list[float]], list[int]]":
        """
        Loads the data for a single subject for a single session.
        Extracts only the data needed for testing and returns a
        python dictionary containing that data.

        Inputs
        -------
        session_file_path : int
            The session of given subject from which to retrieve data for.
            There are 13 sessions, 1 laptop, and 12 mobile.

        Outputs
        -------
        sets class variables for features and non_faces_indices
        """

        # load the session file from the given path
        session_data = tools.load_json_gz_file(session_file_path)

        # list to store the features of this subject
        self.__feature_vectors = []

        # loops over each frame dict in the extracted data file
        # loop at the given time interval to get frames at
        # TIME_INTERVAL seconds, which correspond to the order
        # of the loaded list
        num_features = len(session_data["frame_data"])
        self.__non_face_indices = []
        for t_index in range(0, num_features, self._time_interval):
            # add feature to feature vector
            self.__feature_vectors.append(
                session_data["frame_data"][t_index]["feature_vectors"][FEATURE_EXTRACTION_MODEL]
            ) # end append feature vector to list of feature vectors

            # check if there's more than one face or no faces
            # if so, add to non_face_indices for this session
            if int(session_data["frame_data"][t_index]["num_faces_detected"]) != 1:
                # append the index for this feature vector.
                # this will be the current length of the feature vector
                # list minus one, which will correspond to the current last item
                # in the list
                self.__non_face_indices.append(len(self.__feature_vectors)-1)
            # end if check for non_faces_indices
        # end for loop over feature vectors
    # end self.__load_single_subject_session_data


    # make the stored feature vectors biocapsules
    def __make_biocapsules(self):
        # create biocapsule object
        bc_gen = bc.BioCapsuleGenerator()
        
        pass


# generates labels for a single subject
def gen_labels_for_subject()-> "list[int]":
    pass