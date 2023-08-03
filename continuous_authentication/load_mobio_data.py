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
    - counting if frame is none or multiple faces (abstracted into SessionData class)
"""

import copy
import os

import numpy as np
import tools
from load_rs_features import yield_reference_subject

import biocapsule as bc

# a list of all locations in mobio
MOBIO_LOCATIONS = [
    "but/",
    "idiap/",
    "lia/",
    "uman/",
    "unis/",
    "uoulu/",
]  # end MOBIO_LOCATIONS


# used for testing
def main():
    # sessionDataTest()
    # subjectDataTest()
    subjects = load_MOBIO_dataset(
        extracted_MOBIO_dir="./MOBIO_extracted/one_sec_intervals/",
        time_interval=10,
        feature_extraction_model="arcface",
        use_bc=False,
        multi_rs=False,
    )  # end load_MOBIO_dataset
    print(len(subjects))
    print(subjects[0].get_laptop_session().get_feature_vectors()[0])


# end main function tests


def sessionDataTest():
    # test SessionData Class
    session_data = SessionData(
        session_file_path="./MOBIO_extracted/one_sec_intervals/but/f401/but_laptop_1_f401_01.json.gz",
        time_interval=10,
        feature_extraction_model="arcface",
        use_bc=True,
        rs_feature_vector=np.random.rand(512).tolist(),
    )  # end init SessionData
    print(session_data.get_feature_vectors())
    print(len(session_data.get_feature_vectors()))
    print(session_data.get_labels(0))
    print(session_data.get_labels(1))
    print(len(session_data.get_labels(0)))


# end sessionDataTest


def subjectDataTest():
    # test SubjectData Class
    subject_data = SubjectData(
        subject_dir="./MOBIO_extracted/one_sec_intervals/but/f401/",
        time_interval=10,
        feature_extraction_model="arcface",
        use_bc=True,
        rs_feature_vector=np.random.rand(512).tolist(),
    )  # end init SessionData

    print(subject_data.get_laptop_session().get_feature_vectors())
    print(len(subject_data.get_laptop_session().get_feature_vectors()))
    print(subject_data.get_laptop_session().get_labels(0))
    print(subject_data.get_laptop_session().get_labels(1))
    print(len(subject_data.get_laptop_session().get_labels(0)))

    subject_data = SubjectData(
        subject_dir="./MOBIO_extracted/one_sec_intervals/but/f401/",
        time_interval=10,
        feature_extraction_model="facenet",
        use_bc=False,
        rs_feature_vector=None,
    )  # end init SessionData

    print(subject_data.get_laptop_session() != None)
    print(subject_data.get_mobile_session_one() != None)
    print(len(subject_data.get_mobile_sessions()))


# end subjectDataTest


# loads the entire MOBIO dataset given input parameters
def load_MOBIO_dataset(
    extracted_MOBIO_dir: str,
    time_interval: int,
    feature_extraction_model: str,
    use_bc: bool,
    multi_rs: bool,
    ref_subj_data_dir: str = None,
) -> "list[SubjectData]":
    # check inputs #

    # time_interval check
    if type(time_interval) != int:
        raise Exception("time_invterval MUST be a positive integer")
    if time_interval <= 0:
        raise Exception("time_interval MUST be a positive integer")

    # feature_extraction_model check
    if feature_extraction_model != "arcface":
        if feature_extraction_model != "facenet":
            raise Exception(
                "feature_extraction_model MUST be either 'facenet' or 'arcface'"
            )

    # use_bc check
    if type(use_bc) != bool:
        raise Exception(
            "use_bc must be either TRUE or FALSE. If TRUE, you MUST also specify single or mutliple reference subjects"
        )

    # multi_rs check
    rs_file_name = None
    rs_feature_vector = None
    if use_bc:
        if multi_rs == False:
            rs_file_name = "f210"
            rs_feature_vector = load_single_rs_feature_vector(
                file_path="./MOBIO_extracted/one_sec_intervals/XX_removed_from_exp_XX/f210/unis_laptop_1_f210_01.json.gz",
                feature_extraction_model=feature_extraction_model,
            )  # end load_single_rs_feature_vector
        # end check if use multiple reference subjects
    # end check if use_bc

    # get all of the files to load
    rs_gen = yield_reference_subject(
        file_path=ref_subj_data_dir,
        feature_extraction_model=feature_extraction_model,
    )  # end generator init

    subjects = []
    for location in MOBIO_LOCATIONS:
        # get all subjects at this location
        location_path = os.path.join(extracted_MOBIO_dir, location)
        subject_dirs = os.listdir(location_path)
        for subject_folder_name in subject_dirs:
            # check if we should use the same reference subject or not
            if multi_rs:
                rs_data = next(rs_gen)
                rs_file_name = rs_data[0]
                rs_feature_vector = copy.deepcopy(rs_data[1])
            subject_path = os.path.join(location_path, subject_folder_name)
            subjects.append(
                SubjectData(
                    subject_dir=subject_path,
                    time_interval=time_interval,
                    feature_extraction_model=feature_extraction_model,
                    use_bc=use_bc,
                    rs_feature_vector=rs_feature_vector,
                    rs_file_name=rs_file_name,
                )  # end subject data construction
            )  # end append to subjects list
        # end for loop over all subjects at this location
    # end loop over all locations in MOBIO_LOCATIONS
    return subjects


# end load_MOBIO_dataset function


# loads a single rs feature vector
def load_single_rs_feature_vector(
    file_path: str, feature_extraction_model: str
) -> "list[float]":
    session_data = tools.load_json_gz_file(file_path=file_path)
    return session_data["frame_data"][0]["feature_vectors"][
        feature_extraction_model
    ]


# class to neatly store session information for a subject
class SessionData(object):
    def __init__(
        self,
        session_file_path: str,
        time_interval: int = 10,
        feature_extraction_model: str = "arcface",
        use_bc: bool = False,
        rs_feature_vector: "list[float]" = None,
    ):
        self.__session_file_path = session_file_path
        self.__time_interval = time_interval
        self.__feature_extraction_model = feature_extraction_model
        self.__use_bc = use_bc
        self.__rs_feature_vector = rs_feature_vector

        # load the feature vectors and the indices corresponding
        # to the feature vectors with non-faces
        self.__feature_vectors = []
        self.__flipped_feature_vectors = []
        self.__bad_detections = []
        self.__load_single_subject_session_data(
            session_file_path=self.__session_file_path
        )
        assert self.__feature_vectors != None

        # if using biocapsule, apply transformations and store back
        if self.__use_bc:
            assert rs_feature_vector != None
            self.__make_biocapsules()
        # end if

    # end __init__ for SessionData

    # get the session's file_path
    def get_session_file_path(self) -> str:
        return self.__session_file_path

    # get the flipped feature vectors, if facenet
    def get_flipped_feature_vectors(self) -> "list[list[float]]":
        return self.__flipped_feature_vectors

    # function for retrieving feature vectors for running a test
    def get_feature_vectors(self) -> "list[list[float]]":
        return self.__feature_vectors

    # function for retrieving the labels. specifically made to avoid
    # issues with bad detections
    def get_labels(self, classification: int) -> "list[int]":
        # generate the labels for this session, based off of
        # given classification * the length of the feature vectors
        labels = [classification] * len(self.__feature_vectors)
        return labels

    # get the number of non_face_vectors
    def get_bad_detection_count(self) -> int:
        return len(self.__bad_detections)

    # loads the data from a single session of a subject,
    # extracting the most important information to be accumulated
    def __load_single_subject_session_data(
        self, session_file_path: str
    ) -> "tuple[list[list[float]], list[int]]":
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
        self.__bad_detections = []
        self.__flipped_feature_vectors = []

        # loops over each frame dict in the extracted data file
        # loop at the given time interval to get frames at
        # TIME_INTERVAL seconds, which correspond to the order
        # of the loaded list
        num_features = len(session_data["frame_data"])
        for t_index in range(0, num_features, self.__time_interval):
            # check if there's no faces
            if (
                int(session_data["frame_data"][t_index]["num_faces_detected"])
                == 0
            ):
                self.__bad_detections.append(
                    session_data["frame_data"][t_index]["feature_vectors"][
                        self.__feature_extraction_model
                    ]
                )  # end append to non face feature vectors
            else:
                # add feature to feature vector
                self.__feature_vectors.append(
                    session_data["frame_data"][t_index]["feature_vectors"][
                        self.__feature_extraction_model
                    ]
                )  # end append feature vector to list of feature vectors

                # get the flipped version of the model's feature vector
                self.__flipped_feature_vectors.append(
                    session_data["frame_data"][t_index]["feature_vectors"][
                        f"{self.__feature_extraction_model}_flip"
                    ]
                )
            # end if check for non_faces_indices
        # end for loop over feature vectors
        print(f"Num Good Detections found: {len(self.__feature_vectors)}")
        print(f" Num Bad Detections found: {len(self.__bad_detections)}")

    # end self.__load_single_subject_session_data

    # make the stored feature vectors biocapsules
    def __make_biocapsules(self):
        # create biocapsule object
        bc_gen = bc.BioCapsuleGenerator()

        # loop over feature vectors, creating a biocapsule using the
        # given reference subject and each feature vector in the session
        bc_vectors = []
        for feature_vector in self.__feature_vectors:
            bc_vectors.append(
                bc_gen.biocapsule(
                    user_feature=np.array(feature_vector),
                    rs_feature=np.array(self.__rs_feature_vector),
                ).tolist()  # end biocapsule generation function
            )  # end append to bc_vectors

        # set feature vectors to be biocapsule list
        self.__feature_vectors = copy.deepcopy(bc_vectors)

    # end __make_biocapsules


# end SessionData class


# class to neatly store data for a single subject
# subdivides data into train & test sections as needed
class SubjectData(object):
    def __init__(
        self,
        subject_dir: str,
        time_interval: int = 10,
        feature_extraction_model: str = "arcface",
        use_bc: bool = False,
        rs_feature_vector: "list[float]" = None,
        rs_file_name: str = None,
    ) -> None:
        self.__subject_dir = subject_dir
        self.__subject_id = os.path.basename(subject_dir)
        self.__laptop_session_one = None
        self.__mobile_session_one = None
        self.__mobile_sessions = None
        self.__time_interval = time_interval
        self.__feature_extraction_model = feature_extraction_model
        self.__use_bc = use_bc
        self.__rs_feature_vector = rs_feature_vector
        self.__rs_file_name = rs_file_name

        # load all session data for this subject
        print(f"Loading data for subject '{self.__subject_id}'...")
        self.__load_sessions()
        assert self.__laptop_session_one != None
        assert self.__mobile_session_one != None
        assert self.__mobile_sessions != None
        print("Subject Data Loaded!")

    # end __init__ for SubjectData

    # getters
    def get_laptop_session(self) -> SessionData:
        return self.__laptop_session_one

    def get_mobile_session_one(self) -> SessionData:
        return self.__mobile_session_one

    def get_mobile_sessions(self) -> "list[SessionData]":
        return self.__mobile_sessions

    def get_subject_id(self) -> str:
        return self.__subject_id

    # load all session data. divide into train, validation, and test sets
    # based on multi or single platform
    def __load_sessions(self):
        # get all of the session files
        session_file_paths = []
        for file_name in os.listdir(self.__subject_dir):
            session_file_paths.append(
                os.path.join(self.__subject_dir, file_name)
            )
        # end for

        # sort file paths. we can use this order to help us
        session_file_paths.sort()

        self.__mobile_sessions = []
        # create SessionData objects. use to get feature vectors
        for i, session_file_path in enumerate(session_file_paths):
            if i == 0:  # index of the laptop session (only true if sorted)
                self.__laptop_session_one = SessionData(
                    session_file_path=session_file_path,
                    time_interval=self.__time_interval,
                    feature_extraction_model=self.__feature_extraction_model,
                    use_bc=self.__use_bc,
                    rs_feature_vector=self.__rs_feature_vector,
                )  # end SessionData construction
            elif (
                i == 1
            ):  # index of the first mobile session (only true if sorted)
                self.__mobile_session_one = SessionData(
                    session_file_path=session_file_path,
                    time_interval=self.__time_interval,
                    feature_extraction_model=self.__feature_extraction_model,
                    use_bc=self.__use_bc,
                    rs_feature_vector=self.__rs_feature_vector,
                )  # end SessionData construction
            else:  # all other mobile sessions
                self.__mobile_sessions.append(
                    SessionData(
                        session_file_path=session_file_path,
                        time_interval=self.__time_interval,
                        feature_extraction_model=self.__feature_extraction_model,
                        use_bc=self.__use_bc,
                        rs_feature_vector=self.__rs_feature_vector,
                    )  # end SessionData contstruction
                )  # end append to self.__mobile_sessions
            # end if check sorting SessionData
        # end for loop over session_file_paths

    # end __load_sessions function

    # parse file name to get the file with the given subcomponent
    def __check_for_subcomponent(self, f_name: str, word_to_find: str) -> bool:
        f_name = os.path.basename(f_name)
        for start in range(0, len(f_name) - (len(word_to_find) - 1)):
            name_slice = f_name[start : start + len(word_to_find)]
            if name_slice == word_to_find:
                return True
        return False

    # end __check_for_subcomponent


# end SubjectData class


if __name__ == "__main__":
    main()
